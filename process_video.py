"""
Scale OCR Video Processing Script

Processes video files to extract weight readings from seven-segment displays.
Outputs CSV with frame-by-frame weights and applies temporal smoothing.

Usage:
    python process_video.py --video path/to/video.mp4 --output results.csv
    python process_video.py --video video.mp4 --roi 100,50,400,150 --smoothing-window 5
    python process_video.py --video video.mp4 --batch-size 16 --checkpoint-every 1000 --save-video
    python process_video.py --video video.mp4 --ground-truth gt.csv --gt-tolerance 0.0
    python process_video.py --video video.mp4 --resume  # Resume from checkpoint
    python process_video.py --video video.mp4 --no-strict  # Disable aggressive flagging

Recommendation:
    For new or unprocessed videos (no ground-truth available), run with `--conservative`
    to maximize recall (the script will flag any frame that isn't extremely certain).
    Example:
        python process_video.py --video new_video.mp4 --output new_weights.csv --conservative
"""
import argparse, cv2, torch, pandas as pd, numpy as np
import os
import sys
# Fix for GoPro/high-res videos with multiple streams
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import json
from pathlib import Path
from scipy.signal import medfilt
from tqdm import tqdm
from model import create_model
from dataset import get_transforms

# Add training_scripts to path for CropLabel import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_scripts'))

# ============================================================
# ROI SELECTION WINDOW
# ============================================================
def select_roi_interactively(video_path, start_frame=None, end_frame=None):
    """
    Open a GUI window to let the user select the ROI (scale display region) and frame range.
    Returns ((x, y, w, h), (start_frame, end_frame)) tuple or None if cancelled.
    """
    try:
        from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                                     QHBoxLayout, QVBoxLayout, QSlider, QGroupBox, QAction)
        from PyQt5.QtGui import QPixmap, QImage
        from PyQt5.QtCore import Qt, QTimer
        from qtrangeslider import QRangeSlider
        from CropLabel import CNNCropLabel
        from functools import partial
    except ImportError as e:
        print(f"Error: PyQt5 not installed or CropLabel not found: {e}")
        return None

    class ROISelectionWindow(QWidget):
        def __init__(self, video_path, init_start_frame=None, init_end_frame=None):
            super().__init__()
            self.video_path = video_path
            self.video = cv2.VideoCapture(video_path)
            if not self.video.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            
            # Read first frame to get dimensions
            success, first_frame = self.video.read()
            if success:
                self.original_h = first_frame.shape[0]
                self.original_w = first_frame.shape[1]
            else:
                self.original_h = 720
                self.original_w = 1280
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.current_frame_index = 0
            self.last_loaded_frame = -1
            self.current_raw_frame = None
            self.crop_coords = None
            self.result = None  # Will be set on confirm
            self._confirmed = False
            
            # Frame range state
            self.range_start = init_start_frame if init_start_frame is not None else 0
            self.range_end = init_end_frame if init_end_frame is not None else self.total_frames - 1
            
            # Zoom/pan state
            self.zoom_level = 1.0
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            
            # Debounce timer for slider
            self.slider_timer = QTimer()
            self.slider_timer.setSingleShot(True)
            self.slider_timer.setInterval(100)
            self.pending_slider_action = None
            self.slider_timer.timeout.connect(self._execute_pending_slider_action)
            
            self.setWindowTitle(f"Select ROI - {os.path.basename(video_path)}")
            
            # Calculate optimal window size
            target_video_h = max(400, min(800, int(self.original_h * 0.7)))
            target_video_w = int(target_video_h * (self.original_w / self.original_h))
            controls_width = 220
            ui_overhead = 180
            optimal_width = target_video_w + controls_width
            optimal_height = target_video_h + ui_overhead
            self.resize(optimal_width, optimal_height)
            
            main_layout = self._setup_main_ui()
            self.setLayout(main_layout)
            
            if success:
                self.video_label.set_original_size(self.original_w, self.original_h)
                self.video_label.reset_crop_to_standard()
            
            self.load_frame_from_video()
        
        def _debounced_slider(self, slider, value_changed_func):
            self.slider_timer.stop()
            self.pending_slider_action = lambda: value_changed_func(slider.value())
            self.slider_timer.start()
        
        def _execute_pending_slider_action(self):
            if self.pending_slider_action:
                self.pending_slider_action()
                self.pending_slider_action = None

        def _setup_main_ui(self):
            main_layout = QHBoxLayout()
            video_layout = QVBoxLayout()
            
            # Video label with crop overlay
            self.video_label = CNNCropLabel()
            self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_label.setMinimumSize(600, 300)
            self.video_label.crop_selected.connect(self.on_crop_selected)
            video_layout.addWidget(self.video_label)
            
            # Navigation buttons
            nav_layout = QHBoxLayout()
            self.prev_button = QPushButton("Previous")
            self.prev_button.clicked.connect(self.previous_frame)
            self.prev_button.setToolTip("Go to previous frame (Left arrow)")
            self.prev_action = QAction("Previous frame", self)
            self.prev_action.setShortcut("Left")
            self.prev_action.triggered.connect(self.previous_frame)
            self.addAction(self.prev_action)
            
            self.frame_info_label = QLabel(f"Frame: 0 / {self.total_frames}")
            self.frame_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.next_button = QPushButton("Next")
            self.next_button.clicked.connect(self.next_frame)
            self.next_button.setToolTip("Go to next frame (Right arrow)")
            self.next_action = QAction("Next frame", self)
            self.next_action.setShortcut("Right")
            self.next_action.triggered.connect(self.next_frame)
            self.addAction(self.next_action)
            
            nav_layout.addWidget(self.prev_button)
            nav_layout.addWidget(self.frame_info_label)
            nav_layout.addWidget(self.next_button)
            video_layout.addLayout(nav_layout)
            
            # Range slider (trim)
            range_layout = QHBoxLayout()
            self.range_label = QLabel("Range:")
            self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
            self.range_slider.setStyleSheet("QRangeSlider {qproperty-barColor: #447;}")
            self.range_slider.setRange(0, self.total_frames - 1)
            self.range_slider.setValue((self.range_start, self.range_end))
            self.range_slider.setToolTip("Drag handles to set valid frame range")
            self.range_slider.valueChanged.connect(self._on_range_slider_changed)
            range_layout.addWidget(self.range_label)
            range_layout.addWidget(self.range_slider)
            video_layout.addLayout(range_layout)
            
            # Frame scrub slider
            scrub_layout = QHBoxLayout()
            self.scrub_label = QLabel("Frame:")
            self.frame_slider = QSlider(Qt.Orientation.Horizontal)
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setToolTip("Drag to navigate to a specific frame")
            self.frame_slider.valueChanged.connect(
                partial(self._debounced_slider, self.frame_slider, self.seek_to_frame)
            )
            scrub_layout.addWidget(self.scrub_label)
            scrub_layout.addWidget(self.frame_slider)
            video_layout.addLayout(scrub_layout)
            
            # Keyboard shortcuts
            self.jump_back_action = QAction("Jump Back 10 Frames", self)
            self.jump_back_action.setShortcut("Ctrl+Left")
            self.jump_back_action.triggered.connect(
                lambda: self.seek_to_frame(max(0, self.current_frame_index - 10))
            )
            self.addAction(self.jump_back_action)
            
            self.jump_forward_action = QAction("Jump Forward 10 Frames", self)
            self.jump_forward_action.setShortcut("Ctrl+Right")
            self.jump_forward_action.triggered.connect(
                lambda: self.seek_to_frame(min(self.total_frames - 1, self.current_frame_index + 10))
            )
            self.addAction(self.jump_forward_action)
            
            self.pan_up_action = QAction("Pan Up", self)
            self.pan_up_action.setShortcut("Up")
            self.pan_up_action.triggered.connect(lambda: self.pan_view(0, -50))
            self.addAction(self.pan_up_action)
            
            self.pan_down_action = QAction("Pan Down", self)
            self.pan_down_action.setShortcut("Down")
            self.pan_down_action.triggered.connect(lambda: self.pan_view(0, 50))
            self.addAction(self.pan_down_action)
            
            self.pan_left_action = QAction("Pan Left", self)
            self.pan_left_action.setShortcut("Shift+Left")
            self.pan_left_action.triggered.connect(lambda: self.pan_view(-50, 0))
            self.addAction(self.pan_left_action)
            
            self.pan_right_action = QAction("Pan Right", self)
            self.pan_right_action.setShortcut("Shift+Right")
            self.pan_right_action.triggered.connect(lambda: self.pan_view(50, 0))
            self.addAction(self.pan_right_action)
            
            main_layout.addLayout(video_layout, 3)
            controls_layout = self._setup_controls()
            main_layout.addLayout(controls_layout, 0)
            return main_layout

        def _setup_controls(self):
            controls_layout = QVBoxLayout()
            controls_layout.setSpacing(6)
            controls_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
            
            group_box_style = (
                "QGroupBox { border: 1px solid #555; border-radius: 3px; margin-top: 6px; padding-top: 6px; font-weight: bold; }"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }"
            )
            
            # Info label
            info_label = QLabel(f"{self.original_w}x{self.original_h} | {self.fps:.1f} fps")
            info_label.setStyleSheet(
                "background-color: #2a4a6a; color: white; padding: 4px; border-radius: 3px; font-size: 10px;"
            )
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            controls_layout.addWidget(info_label)
            
            # Shortcuts help
            shortcuts_group = QGroupBox("Shortcuts")
            shortcuts_group.setStyleSheet(group_box_style)
            shortcuts_layout = QVBoxLayout()
            shortcuts_layout.setSpacing(2)
            shortcuts_layout.setContentsMargins(6, 6, 6, 6)
            shortcuts_text = QLabel(
                "← → Frame nav\n"
                "Ctrl+← → Jump 10\n"
                "Wheel: Zoom\n"
                "Drag: Pan (when zoomed)\n"
                "↑↓ Shift+←→ Pan (when zoomed)"
            )
            shortcuts_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
            shortcuts_text.setStyleSheet("font-size: 10px;")
            shortcuts_layout.addWidget(shortcuts_text)
            shortcuts_group.setLayout(shortcuts_layout)
            controls_layout.addWidget(shortcuts_group)

            # ROI Selection group
            crop_group = QGroupBox("ROI Selection")
            crop_group.setStyleSheet(group_box_style)
            crop_layout = QVBoxLayout()
            crop_layout.setSpacing(2)
            crop_layout.setContentsMargins(6, 6, 6, 6)
            self.crop_dims_label = QLabel("Crop: Default")
            self.crop_dims_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.crop_dims_label.setStyleSheet("font-size: 10px;")
            crop_layout.addWidget(self.crop_dims_label)
            
            self.reset_crop_button = QPushButton("Reset to Standard")
            self.reset_crop_button.setToolTip("Reset crop to default standard position")
            self.reset_crop_button.clicked.connect(self.reset_roi)
            self.reset_crop_button.setStyleSheet("font-size: 10px; padding: 2px;")
            crop_layout.addWidget(self.reset_crop_button)
            
            crop_group.setLayout(crop_layout)
            controls_layout.addWidget(crop_group)
            
            # Zoom/Pan group
            zoom_group = QGroupBox("Zoom/Pan")
            zoom_group.setStyleSheet(group_box_style)
            zoom_main_layout = QHBoxLayout()
            zoom_main_layout.setSpacing(4)
            zoom_main_layout.setContentsMargins(6, 6, 6, 6)
            
            self.zoom_info_label = QLabel("Zoom: 1.0x\nPan: (0, 0)")
            self.zoom_info_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            self.zoom_info_label.setStyleSheet("font-size: 10px;")
            zoom_main_layout.addWidget(self.zoom_info_label, 1)
            
            zoom_btn_layout = QVBoxLayout()
            zoom_btn_layout.setSpacing(1)
            self.zoom_in_button = QPushButton("+")
            self.zoom_in_button.setToolTip("Zoom in (Mouse wheel up)")
            self.zoom_in_button.setMaximumHeight(18)
            self.zoom_in_button.setMaximumWidth(35)
            self.zoom_in_button.setStyleSheet("font-size: 10px; padding: 0px;")
            self.zoom_in_button.clicked.connect(lambda: self.apply_zoom(self.zoom_level * 1.5))
            self.zoom_reset_button = QPushButton("⟲")
            self.zoom_reset_button.setToolTip("Reset zoom to 1x")
            self.zoom_reset_button.setMaximumHeight(18)
            self.zoom_reset_button.setMaximumWidth(35)
            self.zoom_reset_button.setStyleSheet("font-size: 10px; padding: 0px;")
            self.zoom_reset_button.clicked.connect(lambda: self.apply_zoom(1.0, reset_pan=True))
            self.zoom_out_button = QPushButton("-")
            self.zoom_out_button.setToolTip("Zoom out (Mouse wheel down)")
            self.zoom_out_button.setMaximumHeight(18)
            self.zoom_out_button.setMaximumWidth(35)
            self.zoom_out_button.setStyleSheet("font-size: 10px; padding: 0px;")
            self.zoom_out_button.clicked.connect(lambda: self.apply_zoom(self.zoom_level / 1.5))
            
            zoom_btn_layout.addWidget(self.zoom_in_button)
            zoom_btn_layout.addWidget(self.zoom_reset_button)
            zoom_btn_layout.addWidget(self.zoom_out_button)
            zoom_main_layout.addLayout(zoom_btn_layout)
            zoom_group.setLayout(zoom_main_layout)
            controls_layout.addWidget(zoom_group)
            
            controls_layout.addStretch()
            
            # Action buttons
            self.confirm_btn = QPushButton("CONFIRM ROI")
            self.confirm_btn.setStyleSheet(
                "background-color: #4a7a4a; color: white; padding: 10px; font-weight: bold;"
            )
            self.confirm_btn.clicked.connect(self.confirm_selection)
            controls_layout.addWidget(self.confirm_btn)
            
            self.cancel_btn = QPushButton("CANCEL")
            self.cancel_btn.setStyleSheet(
                "background-color: #7a4a4a; color: white; padding: 5px;"
            )
            self.cancel_btn.clicked.connect(self.cancel_selection)
            controls_layout.addWidget(self.cancel_btn)
            
            return controls_layout

        def on_crop_selected(self, points):
            # Expect a quad: list of 4 [x,y] points
            if not points or len(points) != 4:
                return
            pts = [(int(p[0]), int(p[1])) for p in points]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w = max(1, max_x - min_x)
            h = max(1, max_y - min_y)
            self.crop_coords = pts  # Store the quad
            actual_ratio = w / h if h > 0 else 0
            self.crop_dims_label.setText(
                f"Crop: {w}x{h} pixels\nRatio: {actual_ratio:.2f}:1\nPosition: ({min_x}, {min_y})\nQuad: {pts}"
            )
            self.update_display_frame()

        def reset_roi(self):
            self.video_label.reset_crop_to_standard()

        def _on_range_slider_changed(self, values):
            new_start, new_end = values
            # Determine which handle moved and seek to it
            if new_start != self.range_start:
                self.range_start = new_start
                self.seek_to_frame(new_start)
            elif new_end != self.range_end:
                self.range_end = new_end
                self.seek_to_frame(new_end)
            else:
                self.range_start, self.range_end = new_start, new_end
            self._update_frame_info_style()

        def _update_frame_info_style(self):
            """Highlight frame info label red if current frame is outside selected range."""
            if self.current_frame_index < self.range_start or self.current_frame_index > self.range_end:
                self.frame_info_label.setStyleSheet(
                    "background-color: #7a3a3a; color: white; padding: 2px; border-radius: 3px;"
                )
            else:
                self.frame_info_label.setStyleSheet("")

        def seek_to_frame(self, frame_num):
            self.current_frame_index = max(0, min(frame_num, self.total_frames - 1))
            self.load_frame_from_video()

        def load_frame_from_video(self):
            if self.last_loaded_frame != self.current_frame_index:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                success, frame = self.video.read()
                
                if success:
                    self.current_raw_frame = frame
                    self.last_loaded_frame = self.current_frame_index
                else:
                    print(f"Failed to read frame {self.current_frame_index}")
                    self.current_raw_frame = None

            if self.current_raw_frame is not None:
                self.prev_button.setEnabled(self.current_frame_index > 0)
                self.next_button.setEnabled(self.current_frame_index < self.total_frames - 1)
                self.frame_info_label.setText(
                    f"Frame: {self.current_frame_index} / {self.total_frames}"
                )
                self._update_frame_info_style()
                self.update_display_frame()
            
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)

        def update_display_frame(self):
            if self.current_raw_frame is None:
                self.video_label.setText("No frame loaded")
                return
            self._update_image(self.current_raw_frame.copy())

        def _update_image(self, frame):
            if self.zoom_level > 1.0:
                h, w = frame.shape[:2]
                visible_w = int(w / self.zoom_level)
                visible_h = int(h / self.zoom_level)
                x1 = int(self.pan_offset_x)
                y1 = int(self.pan_offset_y)
                x2 = min(x1 + visible_w, w)
                y2 = min(y1 + visible_h, h)
                frame = frame[y1:y2, x1:x2]
            
            qimage = self._cv2_to_qimage(frame)
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)

        def _cv2_to_qimage(self, cv_img):
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = channel * width
            return QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        def next_frame(self):
            if self.current_frame_index < self.total_frames - 1:
                self.current_frame_index += 1
                self.load_frame_from_video()

        def previous_frame(self):
            if self.current_frame_index > 0:
                self.current_frame_index -= 1
                self.load_frame_from_video()

        def handle_wheel_event(self, event):
            delta = event.angleDelta().y()
            if delta > 0:
                new_zoom = self.zoom_level * 1.1
            else:
                new_zoom = self.zoom_level / 1.1
            mouse_pos = event.pos()
            self.apply_zoom(new_zoom, zoom_center=mouse_pos)

        def apply_zoom(self, new_zoom, reset_pan=False, zoom_center=None):
            old_zoom = self.zoom_level
            self.zoom_level = max(1.0, min(new_zoom, 10.0))
            
            if reset_pan or self.zoom_level == 1.0:
                self.pan_offset_x = 0
                self.pan_offset_y = 0
            elif zoom_center and self.original_w and self.original_h:
                params = self.video_label._get_display_params()
                if params:
                    scale, off_x, off_y, visible_w, visible_h = params
                    img_x = (zoom_center.x() - off_x) / scale + self.pan_offset_x
                    img_y = (zoom_center.y() - off_y) / scale + self.pan_offset_y
                    new_visible_w = self.original_w / self.zoom_level
                    new_visible_h = self.original_h / self.zoom_level
                    mouse_rel_x = (zoom_center.x() - off_x) / (visible_w * scale)
                    mouse_rel_y = (zoom_center.y() - off_y) / (visible_h * scale)
                    self.pan_offset_x = img_x - (mouse_rel_x * new_visible_w)
                    self.pan_offset_y = img_y - (mouse_rel_y * new_visible_h)
                    max_pan_x = max(0, self.original_w - new_visible_w)
                    max_pan_y = max(0, self.original_h - new_visible_h)
                    self.pan_offset_x = max(0, min(self.pan_offset_x, max_pan_x))
                    self.pan_offset_y = max(0, min(self.pan_offset_y, max_pan_y))
            
            self.video_label.set_zoom(self.zoom_level, self.pan_offset_x, self.pan_offset_y)
            self.zoom_info_label.setText(
                f"Zoom: {self.zoom_level:.1f}x\nPan: ({int(self.pan_offset_x)}, {int(self.pan_offset_y)})"
            )
            is_zoomed = self.zoom_level > 1.0
            self.pan_up_action.setEnabled(is_zoomed)
            self.pan_down_action.setEnabled(is_zoomed)
            self.pan_left_action.setEnabled(is_zoomed)
            self.pan_right_action.setEnabled(is_zoomed)
            self.update_display_frame()

        def pan_view(self, dx, dy):
            if self.zoom_level <= 1.0:
                return
            new_pan_x = self.pan_offset_x + dx
            new_pan_y = self.pan_offset_y + dy
            visible_w = self.original_w / self.zoom_level
            visible_h = self.original_h / self.zoom_level
            max_pan_x = max(0, self.original_w - visible_w)
            max_pan_y = max(0, self.original_h - visible_h)
            self.pan_offset_x = max(0, min(new_pan_x, max_pan_x))
            self.pan_offset_y = max(0, min(new_pan_y, max_pan_y))
            self.video_label.set_zoom(self.zoom_level, self.pan_offset_x, self.pan_offset_y)
            self.zoom_info_label.setText(
                f"Zoom: {self.zoom_level:.1f}x\nPan: ({int(self.pan_offset_x)}, {int(self.pan_offset_y)})"
            )
            self.update_display_frame()

        def on_pan_drag(self, pan_x, pan_y):
            self.pan_offset_x = pan_x
            self.pan_offset_y = pan_y
            self.zoom_info_label.setText(
                f"Zoom: {self.zoom_level:.1f}x\nPan: ({int(self.pan_offset_x)}, {int(self.pan_offset_y)})"
            )
            self.update_display_frame()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.update_display_frame()

        def confirm_selection(self):
            if self.crop_coords:
                self.result = (self.crop_coords, (self.range_start, self.range_end))
                self._confirmed = True
            self.video.release()
            self.close()

        def cancel_selection(self):
            self._confirmed = False
            self.video.release()
            self.close()

        def closeEvent(self, event):
            if self.video.isOpened():
                self.video.release()
            super().closeEvent(event)

    # Create or get QApplication instance
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        window = ROISelectionWindow(video_path, start_frame, end_frame)
        window.show()
        
        # Run event loop until window closes
        while window.isVisible():
            app.processEvents()
        
        if window._confirmed and window.result:
            return window.result
        return None
    except Exception as e:
        print(f"Error in ROI selection window: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# CONFIGURATION
# ============================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=False, help='Path to video file')
    parser.add_argument('--output', default=None, help='Output CSV path')
    parser.add_argument('--model', default='models/best_model.pth', help='Model path')
    parser.add_argument('--roi', default=None, help='ROI as x,y,w,h (e.g., 100,50,400,150)')
    parser.add_argument('--confidence-threshold', type=float, default=0.3, 
                       help='Min confidence threshold (default: 0.3, lower = stricter)')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Batch size for inference (default: 16, higher = faster but more RAM)')
    parser.add_argument('--checkpoint-every', type=int, default=1000, 
                       help='Save checkpoint every N frames (default: 1000, 0 = no checkpoints)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from last checkpoint if available')
    # strict is default True; provide --no-strict to disable
    parser.add_argument('--no-strict', dest='strict', action='store_false',
                       help='Disable aggressive flagging (by default strict mode is ON)')
    parser.set_defaults(strict=True)
    parser.add_argument('--smoothing-window', type=int, default=None,
                       help='Explicit smoothing window; overrides default behavior')
    parser.add_argument('--ground-truth', default=None,
                       help='Path to ground-truth CSV with columns `frame_num` and `actual_weight` (optional)')
    parser.add_argument('--gt-tolerance', type=float, default=0.0,
                       help='Tolerance (kg) when comparing predictions to ground-truth. 0.0 = any difference flagged')
    parser.add_argument('--gt-match-by', choices=['frame', 'timestamp'], default='frame',
                       help='Whether to match ground-truth rows to predictions by `frame` number or `timestamp`')
    parser.add_argument('--conservative', action='store_true',
                       help='Flag any frame unless model is extremely certain (high recall, low precision)')
    parser.add_argument('--pass-confidence', type=float, default=0.99,
                       help='Confidence threshold to consider a prediction "certain" when --conservative is used')
    parser.add_argument('--pass-entropy', type=float, default=0.01,
                       help='Entropy upper bound to consider a prediction "certain" when --conservative is used')
    parser.add_argument('--pass-delta', type=float, default=0.02,
                       help='Max absolute difference between raw and smoothed to be considered "certain" in conservative mode (kg)')
    return parser.parse_args()

# ============================================================
# LOAD MODEL
# ============================================================
def load_model(model_path, device):
    """Load trained model from checkpoint (handles wrapped state_dict prefixes)."""
    print(f"Loading model from {model_path}...")

    model = create_model(device=device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        raw_state = checkpoint['model_state_dict']
    else:
        raw_state = checkpoint

    # Detect and strip common wrapper prefixes (e.g. '_orig_mod.' or 'module.')
    prefix_candidates = ('_orig_mod.', 'module.')
    needs_fix = any(k.startswith(prefix_candidates) for k in raw_state.keys())
    if needs_fix:
        fixed_state = {}
        for k, v in raw_state.items():
            new_k = k
            for p in prefix_candidates:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    break
            fixed_state[new_k] = v
        state_to_load = fixed_state
        print("  Stripped wrapper prefix from state_dict keys")
    else:
        state_to_load = raw_state

    try:
        model.load_state_dict(state_to_load)
        print("  Model loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"  Warning: load_state_dict strict=True failed: {e}")
        print("    Attempting load with strict=False (will ignore unmatched keys)...")
        model.load_state_dict(state_to_load, strict=False)
        print("  Model loaded with strict=False (check for missing/unexpected keys)")

    model.eval()
    return model

# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================
def get_checkpoint_path(video_path):
    """Generate checkpoint file path for a video"""
    base_name = os.path.splitext(video_path)[0]
    return f"{base_name}_processing_checkpoint.json"

def save_checkpoint(checkpoint_path, results, frame_num):
    """Save processing checkpoint"""
    checkpoint_data = {
        'last_frame': frame_num,
        'results': results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"  Checkpoint saved at frame {frame_num}")

def load_checkpoint(checkpoint_path):
    """Load processing checkpoint if exists"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            print(f"  Found checkpoint from {data.get('timestamp', 'unknown time')}")
            print(f"  Resuming from frame {data['last_frame']}")
            return data['results'], data['last_frame']
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            return [], 0
    return [], 0

def cleanup_checkpoint(checkpoint_path):
    """Remove checkpoint file after successful completion"""
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("  Checkpoint file cleaned up")

# ============================================================
# VIDEO PROCESSING
# ============================================================
def get_video_metadata(video_path):
    """Get video metadata without loading frames"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    
    cap.release()
    return metadata

def get_roi(frame, roi_coords=None):
    if roi_coords:
        try:
            # Expect a quad: list of 4 [x,y] points
            if isinstance(roi_coords, (list, tuple)) and len(roi_coords) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in roi_coords):
                # Warp to CNN size (256x64)
                src_pts = np.float32(roi_coords)
                dst_pts = np.float32([[0, 0], [256, 0], [256, 64], [0, 64]])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (256, 64), flags=cv2.INTER_LINEAR)
                return warped
            else:
                print(f"Warning: Expected quad ROI [[x,y],...], got {roi_coords!r}. Using full frame.")
                return frame
        except Exception as e:
            print(f"Warning: Error processing quad ROI: {e}. Using full frame.")
            return frame
    return frame

def apply_clahe(frame):
    """Apply CLAHE preprocessing to match training data pipeline."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bgr_enhanced

def preprocess_frame(frame, transform):
    """Preprocess frame for model input with CLAHE enhancement."""
    enhanced = apply_clahe(frame)
    image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented['image']
    return image_tensor

# ============================================================
# INFERENCE
# ============================================================
def predict_weight_batch(model, frame_tensors, device):
    """
    Run model inference on a batch of frames.
    Args:
        model: trained model
        frame_tensors: list of tensors (C, H, W)
        device: torch device
    Returns:
        list of (predicted_weight, confidence) tuples
    """
    if len(frame_tensors) == 0:
        return []
    
    # Stack into batch: list of (C,H,W) -> (B, C, H, W)
    batch_tensor = torch.stack(frame_tensors).to(device)
    
    with torch.no_grad():
        # Forward pass
        log_probs, _ = model(batch_tensor)

        # Decode predictions using model's built-in decoder
        # log_probs shape: (seq_len, batch, num_classes)
        decoded_list = model.decode_predictions(log_probs, enforce_format=True)

        # Calculate confidence scores
        probs = torch.exp(log_probs)
        max_probs, _ = torch.max(probs, dim=2)  # (seq_len, batch)
        confidences = torch.mean(max_probs, dim=0)  # (batch,)

        # Calculate per-sample entropy across time steps
        eps = 1e-9
        entropy_per_timestep = - (probs * torch.log(probs + eps)).sum(dim=2)  # (seq_len, batch)
        entropies = torch.mean(entropy_per_timestep, dim=0)  # (batch,)

        # Pair predictions with confidences and entropies
        results = [(pred, conf.item(), ent.item()) for pred, conf, ent in zip(decoded_list, confidences, entropies)]

    return results

def process_video_streaming_batched(video_path, model, transform, roi_coords, device, 
                                    batch_size=8, checkpoint_every=1000, resume=False,
                                    frame_range=None):
    """
    Process video frame-by-frame with batch inference and checkpoint saving.
    Memory usage: O(batch_size) regardless of video length.
    
    Args:
        video_path: path to video file
        model: trained model
        transform: preprocessing transform
        roi_coords: ROI coordinates or None
        device: torch device
        batch_size: number of frames to process in one batch
        checkpoint_every: save checkpoint every N frames (0 = no checkpoints)
        resume: whether to resume from checkpoint
        frame_range: optional (start_frame, end_frame) tuple to limit processing
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Apply frame range limits
    range_start = 0
    range_end = total_frames - 1
    if frame_range is not None:
        range_start, range_end = frame_range
        range_start = max(0, range_start)
        range_end = min(total_frames - 1, range_end)
        print(f"Processing frame range: {range_start} - {range_end} ({range_end - range_start + 1} frames)")
    
    # Checkpoint management
    checkpoint_path = get_checkpoint_path(video_path) if checkpoint_every > 0 else None
    results = []
    start_frame = range_start
    
    # Seek to start of frame range
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    if resume and checkpoint_path:
        results, start_frame = load_checkpoint(checkpoint_path)
        if start_frame > 0:
            print(f"Skipping to frame {start_frame}/{total_frames}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_batch = []
    frame_nums = []
    frame_num = start_frame
    frames_to_process = range_end - start_frame + 1
    
    print(f"\n{'='*60}")
    print(f"Processing {frames_to_process} frames (batch_size={batch_size})...")
    print(f"{'='*60}")
    
    with tqdm(total=frames_to_process, desc="Processing frames", unit="frame", 
              initial=0, dynamic_ncols=True) as pbar:
        while frame_num <= range_end:
            ret, frame = cap.read()
            if not ret:
                break
            
            cropped = get_roi(frame, roi_coords)
            
            tensor = preprocess_frame(cropped, transform)
            
            frame_batch.append(tensor)
            frame_nums.append(frame_num)
            
            if len(frame_batch) >= batch_size:
                predictions = predict_weight_batch(model, frame_batch, device)
                
                for fn, (weight, confidence, entropy) in zip(frame_nums, predictions):
                    timestamp = fn / fps if fps > 0 else fn
                    results.append({
                        'frame_num': fn,
                        'timestamp': timestamp,
                        'raw_weight': weight,
                        'confidence': confidence,
                        'entropy': entropy
                    })
                
                pbar.update(len(frame_batch))
                
                frame_batch = []
                frame_nums = []
                
                if checkpoint_path and checkpoint_every > 0 and frame_num % checkpoint_every == 0:
                    save_checkpoint(checkpoint_path, results, frame_num)
            
            frame_num += 1
    
    if len(frame_batch) > 0:
        predictions = predict_weight_batch(model, frame_batch, device)
        for fn, (weight, confidence, entropy) in zip(frame_nums, predictions):
            timestamp = fn / fps if fps > 0 else fn
            results.append({
                'frame_num': fn,
                'timestamp': timestamp,
                'raw_weight': weight,
                'confidence': confidence,
                'entropy': entropy
            })
        pbar.update(len(frame_batch))
    
    cap.release()
    
    if checkpoint_path and checkpoint_every > 0:
        save_checkpoint(checkpoint_path, results, frame_num)
    
    return results, checkpoint_path

# ============================================================
# POST-PROCESSING
# ============================================================
def apply_temporal_smoothing(weights, window_size=5):
    """Apply median filter to smooth weight predictions"""
    float_weights = []
    for w in weights:
        try:
            float_weights.append(float(w))
        except (ValueError, TypeError):
            float_weights.append(0.0)
    
    if len(float_weights) == 0:
        return []
            
    if window_size % 2 == 0:
        window_size += 1
    
    window_size = min(window_size, len(float_weights))
    if window_size < 3:
        return float_weights
        
    smoothed = medfilt(float_weights, kernel_size=window_size)
    
    return smoothed

def detect_outliers(raw_weights, smoothed_weights, threshold=0.1):
    """
    Flag predictions that differ significantly from smoothed values.
    
    Args:
        raw_weights: list of raw weight predictions
        smoothed_weights: list of smoothed weights
        threshold: difference threshold in kg (default: 0.1 = 100 grams)
    
    Returns:
        list of boolean flags
    """
    flags = []
    for raw, smooth in zip(raw_weights, smoothed_weights):
        try:
            r_val = float(raw)
            diff = abs(r_val - smooth)
            flags.append(diff > threshold)
        except (ValueError, TypeError):
            flags.append(True)  # Flag unparseable values
    return flags

def detect_sudden_changes(raw_weights, prev_smoothed, threshold=0.05):
    """Flag frames where the raw prediction suddenly deviates from previous smoothed value.

    Args:
        raw_weights: list of raw weight predictions (strings or float)
        prev_smoothed: list of smoothed weights (floats)
        threshold: kg threshold to call a sudden change

    Returns:
        list of boolean flags (True = sudden change)
    """
    flags = []
    for i, raw in enumerate(raw_weights):
        try:
            r = float(raw)
            prev = prev_smoothed[i-1] if i > 0 else prev_smoothed[i]
            flags.append(abs(r - prev) > threshold)
        except (ValueError, TypeError):
            flags.append(True)
    return flags

import re
def check_format_flags(weights, pattern=r'^\d+\.\d{3}$'):
    """Flag predictions that don't match the expected numeric format 'X.XXX'.

    Returns a list of booleans where True = format is invalid.
    """
    flags = []
    prog = re.compile(pattern)
    for w in weights:
        try:
            s = str(w)
            flags.append(not bool(prog.match(s)))
        except Exception:
            flags.append(True)
    return flags


def load_ground_truth(gt_path):
    """Load ground-truth CSV and return a normalized mapping.

    Accepts CSVs that contain one of the frame columns: 'frame_num', 'frame', 'frameNumber'
    or a timestamp column 'timestamp'/'time'. The weight column may be named 'actual_weight',
    'actual', or 'weight'. Returns a pandas.DataFrame for flexible matching.
    """
    if gt_path is None:
        return None
    if not os.path.exists(gt_path):
        print(f"  Warning: Ground-truth file not found: {gt_path}")
        return None

    try:
        df = pd.read_csv(gt_path)
    except Exception as e:
        print(f"  Warning: Could not read ground-truth CSV: {e}")
        return None

    df_cols = {c.lower(): c for c in df.columns}
    frame_col = None
    for candidate in ('frame_num', 'frame', 'framenumber'):
        if candidate in df_cols:
            frame_col = df_cols[candidate]
            break
    time_col = None
    for candidate in ('timestamp', 'time'):
        if candidate in df_cols:
            time_col = df_cols[candidate]
            break

    weight_col = None
    for candidate in ('actual_weight', 'actual', 'weight'):
        if candidate in df_cols:
            weight_col = df_cols[candidate]
            break

    if weight_col is None:
        print('  Warning: Ground-truth CSV missing weight column (expecting actual_weight/actual/weight)')
        return None

    return {
        'df': df,
        'frame_col': frame_col,
        'time_col': time_col,
        'weight_col': weight_col
    }

def calculate_confidence_flags(confidences, threshold=0.3):
    """
    Flag low-confidence predictions.
    
    Args:
        confidences: list of confidence scores
        threshold: minimum acceptable confidence (default: 0.3)
    
    Returns:
        list of boolean flags (True = needs review)
    """
    return [c < threshold for c in confidences]

def calculate_entropy_flags(entropies, threshold=0.5):
    """Flag high-entropy (uncertain) predictions."""
    flags = []
    for e in entropies:
        try:
            flags.append(e > threshold)
        except Exception:
            flags.append(True)
    return flags

# ============================================================
# OUTPUT
# ============================================================
def save_results_csv(results, output_path, metadata):
    """Save predictions to CSV"""
    df = pd.DataFrame(results)

    # Include actual_weight column if present in any result
    cols = ['frame_num', 'timestamp', 'raw_weight', 'smoothed_weight', 'confidence', 'entropy']
    if any('actual_weight' in r for r in results):
        cols.append('actual_weight')
    cols.append('needs_review')
    # Ensure only existing columns are selected
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total frames: {len(df)}")
    print(f"Video duration: {df['timestamp'].max():.1f} seconds ({df['timestamp'].max()/60:.1f} minutes)")
    print(f"Flagged for review: {df['needs_review'].sum()} ({df['needs_review'].sum()/len(df)*100:.1f}%)")
    print(f"\nWeight Statistics (kg):")
    print(df['smoothed_weight'].describe())
    print(f"\nConfidence Statistics:")
    print(df['confidence'].describe())

def create_annotated_video(video_path, results, output_path, roi_coords=None, frame_range=None):
    """Create video with predicted weights overlaid"""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame range
    start_frame = 0
    end_frame = total_frames - 1
    if frame_range is not None:
        start_frame, end_frame = frame_range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)
        print(f"  Annotating frame range: {start_frame} - {end_frame}")
    
    # Try multiple codecs with fallback
    fourcc_options = [
        ('mp4v', 'MP4V'),
        ('avc1', 'H.264/AVC1'),
        ('X264', 'X264'),
        ('MJPG', 'Motion JPEG')
    ]
    
    out = None
    for codec, name in fourcc_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec) # type: ignore
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"  Using {name} codec")
                break
            else:
                out = None
        except Exception as e:
            print(f"  Warning: {name} codec failed: {e}")
            out = None
    
    if out is None or not out.isOpened():
        print("  Error: Could not initialize video writer with any codec")
        cap.release()
        return
    
    print(f"\nCreating annotated video...")
    
    # Seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    
    # Convert results to dict for faster lookup
    results_dict = {r['frame_num']: r for r in results}
    frames_to_process = end_frame - start_frame + 1
    
    with tqdm(total=frames_to_process, desc="Annotating frames", unit="frame", dynamic_ncols=True) as pbar:
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in results_dict:
                res = results_dict[frame_idx]
                text = f"Weight: {res['smoothed_weight']:.3f} g (Conf: {res['confidence']:.2f})"
                color = (0, 255, 0) if not res['needs_review'] else (0, 0, 255)
                
                # Draw ROI box if applicable
                if roi_coords:
                    try:
                        # Quad ROI -> draw polygon
                        if isinstance(roi_coords, (list, tuple)) and len(roi_coords) == 4 and isinstance(roi_coords[0], (list, tuple)):
                            pts = np.array([[int(p[0]), int(p[1])] for p in roi_coords], dtype=np.int32)
                            cv2.polylines(frame, [pts.reshape((-1,1,2))], isClosed=True, color=(255,0,0), thickness=2)
                        else:
                            print(f"Warning: Expected quad ROI for annotation, got {roi_coords!r}. Skipping ROI drawing.")
                        cv2.putText(frame, text, (30, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    except Exception as e:
                        print(f"Warning: Error drawing ROI: {e}")
                        cv2.putText(frame, text, (30, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                else:
                    cv2.putText(frame, text, (30, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
        
    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")

# ============================================================
# MAIN WORKFLOW
# ============================================================
def main():
    """Main execution pipeline"""
    
    args = parse_args()
    
    # If video not provided via CLI, open file dialog
    if args.video is None:
        try:
            from PyQt5.QtWidgets import QApplication, QFileDialog
            import sys
            
            print("No video argument provided. Opening file dialog...")
            # Create QApplication instance (required for Qt widgets)
            app = QApplication(sys.argv)
            
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Video File",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
            )
            
            if file_path:
                args.video = file_path
                print(f"Selected: {args.video}")
            else:
                print("No file selected. Exiting.")
                return 0
                
        except ImportError:
            print("Error: PyQt5 not installed. Please provide --video argument.")
            return 1
        except Exception as e:
            print(f"Error opening file dialog: {e}")
            return 1

    print("\n" + "="*60)
    print("SCALE OCR VIDEO PROCESSOR")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint interval: {args.checkpoint_every} frames" if args.checkpoint_every > 0 else "Checkpoints: disabled")
    
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        return 1
    
    print(f"\nLoading video: {args.video}")
    try:
        metadata = get_video_metadata(args.video)
        print(f"   Resolution: {metadata['width']}x{metadata['height']}")
        print(f"   FPS: {metadata['fps']:.2f}")
        print(f"   Total frames: {metadata['frame_count']}")
        print(f"   Duration: {metadata['frame_count']/metadata['fps']:.1f} seconds ({metadata['frame_count']/metadata['fps']/60:.1f} minutes)")
    except Exception as e:
        print(f"\nError loading video: {e}")
        return 1

    if not args.conservative and args.ground_truth is None:
        print("\nTIP: For new/unlabeled videos, consider running with --conservative to maximize recall and avoid unflagged errors.")
    
    # Interactive ROI selection if --roi not provided
    roi_coords = args.roi
    frame_range = None
    if roi_coords is None:
        print("\nNo ROI specified. Opening interactive ROI selection...")
        result = select_roi_interactively(args.video)
        if result is None:
            print("ROI selection cancelled. Exiting.")
            return 0
        roi_coords, frame_range = result
        print(f"   Selected ROI: {roi_coords[0]},{roi_coords[1]},{roi_coords[2]},{roi_coords[3]}")
        print(f"   Frame range: {frame_range[0]} - {frame_range[1]}")
    
    transform = get_transforms(image_size=(256, 64), is_train=False)
    
    try:
        results, checkpoint_path = process_video_streaming_batched(
            args.video, model, transform, 
            roi_coords, device,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            frame_range=frame_range
        )
    except KeyboardInterrupt:
        print("\n\nWarning: Processing interrupted by user (Ctrl+C)")
        print("Tip: Use --resume flag to continue from last checkpoint.")
        return 1
    except Exception as e:
        print(f"\nError during processing: {e}")
        print("Tip: Checkpoint saved. Use --resume flag to continue.")
        import traceback
        traceback.print_exc()
        return 1
    
    # Decide smoothing window: explicit --smoothing-window overrides other logic
    if args.smoothing_window is not None:
        smoothing_window = args.smoothing_window
    else:
        # strict mode disables smoothing by default (window=1)
        smoothing_window = 1 if args.strict else 3
    print(f"\nApplying temporal smoothing (window={smoothing_window})...")
    raw_weights = [r['raw_weight'] for r in results]
    smoothed_weights = apply_temporal_smoothing(raw_weights, smoothing_window)
    
    # Update results with smoothed weights
    for i, result in enumerate(results):
        result['smoothed_weight'] = smoothed_weights[i]
    
    print(f"Flagging problematic predictions...")

    # Determine thresholds (strict mode makes flagging aggressive)
    if args.strict:
        outlier_thresh = 0.05  # 50 g
        conf_thresh = max(args.confidence_threshold, 0.85)
        sudden_thresh = 0.05  # 50 g sudden jump
        format_check = True
        print(f"   STRICT mode: outlier={outlier_thresh} kg, conf>={conf_thresh}, sudden={sudden_thresh} kg, format_check=ON")
    else:
        outlier_thresh = 0.1  # 100 g
        conf_thresh = args.confidence_threshold
        sudden_thresh = 0.2
        format_check = False
        print(f"   outlier={outlier_thresh} kg, confidence>={conf_thresh}, sudden={sudden_thresh} kg")

    outlier_flags = detect_outliers(raw_weights, smoothed_weights, threshold=outlier_thresh)
    confidence_flags = calculate_confidence_flags(
        [r['confidence'] for r in results],
        conf_thresh
    )

    sudden_flags = detect_sudden_changes(raw_weights, smoothed_weights, threshold=sudden_thresh)
    format_flags = check_format_flags(raw_weights) if format_check else [False] * len(results)

    entropies = [r.get('entropy', 0.0) for r in results]
    entropy_thresh = 0.3 if args.strict else 0.6
    entropy_flags = calculate_entropy_flags(entropies, threshold=entropy_thresh)

    # Median disagreement flag: compare raw to median smoothed value
    try:
        median_sm = float(np.median(smoothed_weights))
    except Exception:
        median_sm = None
    median_flags = []
    if median_sm is not None:
        med_thresh = outlier_thresh * 0.7
        for rw in raw_weights:
            try:
                median_flags.append(abs(float(rw) - median_sm) > med_thresh)
            except Exception:
                median_flags.append(True)
    else:
        median_flags = [False] * len(results)

    for i, result in enumerate(results):
        # needs_review if any aggressive condition is met
        result['needs_review'] = bool(
            outlier_flags[i] or confidence_flags[i] or sudden_flags[i] or format_flags[i] or entropy_flags[i] or median_flags[i]
        )

    # Conservative mode: invert logic and only mark NOT needs_review when model is extremely certain
    if args.conservative:
        print(f"   CONSERVATIVE mode: only frames meeting high-certainty thresholds will be unflagged (pass_conf={args.pass_confidence}, pass_entropy={args.pass_entropy}, pass_delta={args.pass_delta})")
        for i, result in enumerate(results):
            try:
                conf_ok = float(result.get('confidence', 0.0)) >= args.pass_confidence
            except Exception:
                conf_ok = False
            try:
                ent_ok = float(result.get('entropy', 1.0)) <= args.pass_entropy
            except Exception:
                ent_ok = False
            try:
                raw = float(result.get('raw_weight'))
                smooth = float(result.get('smoothed_weight'))
                delta_ok = abs(raw - smooth) <= args.pass_delta
            except Exception:
                delta_ok = False
            fmt_ok = not format_flags[i]

            certain = conf_ok and ent_ok and delta_ok and fmt_ok
            if not certain:
                result['needs_review'] = True
    gt_info = None
    if args.ground_truth:
        gt_info = load_ground_truth(args.ground_truth)

    if gt_info is not None:
        df_gt = gt_info['df']
        frame_col = gt_info['frame_col']
        time_col = gt_info['time_col']
        weight_col = gt_info['weight_col']

        total_gt_found = 0
        total_mismatches = 0
        mismatches_flagged_before = 0
        mismatches_forced = 0

        frame_lookup = None
        if frame_col is not None:
            try:
                frame_lookup = {int(r[frame_col]): float(r[weight_col]) for _, r in df_gt.iterrows() if pd.notna(r[frame_col])}
            except Exception:
                frame_lookup = None

        for i, res in enumerate(results):
            actual = None
            if args.gt_match_by == 'frame' and frame_lookup is not None:
                actual = frame_lookup.get(int(res['frame_num']), None)
            if actual is None and time_col is not None:
                try:
                    diffs = (df_gt[time_col].astype(float) - float(res['timestamp'])).abs()
                    idx = int(diffs.idxmin())
                    actual = float(df_gt.loc[idx, weight_col])
                except Exception:
                    actual = None

            if actual is not None:
                total_gt_found += 1
                res['actual_weight'] = actual
                try:
                    raw_val = float(res['raw_weight'])
                except Exception:
                    raw_val = None
                try:
                    smooth_val = float(res['smoothed_weight'])
                except Exception:
                    smooth_val = None

                mismatch = False
                if raw_val is None and smooth_val is None:
                    mismatch = True
                else:
                    if raw_val is not None and abs(raw_val - actual) > args.gt_tolerance:
                        mismatch = True
                    if smooth_val is not None and abs(smooth_val - actual) > args.gt_tolerance:
                        mismatch = True

                if mismatch:
                    total_mismatches += 1
                    if res.get('needs_review', False):
                        mismatches_flagged_before += 1
                    else:
                        mismatches_forced += 1
                        res['needs_review'] = True

        print(f"\nGround-truth rows matched: {total_gt_found}")
        print(f"Total mismatches found vs GT: {total_mismatches}")
        print(f"Mismatches already flagged before GT enforcement: {mismatches_flagged_before}")
        print(f"Mismatches forced to flagged by GT enforcement: {mismatches_forced}")
        if total_mismatches > 0 and mismatches_forced == 0:
            print("All mismatches were already flagged by heuristics.")
        elif total_mismatches > 0:
            print("GT enforcement ensured all mismatches are flagged. Set --gt-tolerance smaller to be stricter.")
    
    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(args.video)[0]
        output_path = f"{base_name}_weights.csv"
        
    save_results_csv(results, output_path, metadata)
    
    if args.save_video:
        video_output = os.path.splitext(args.video)[0] + '_annotated.mp4'
        print(f"\nCreating annotated video: {video_output}")
        create_annotated_video(args.video, results, video_output, roi_coords, frame_range)
    
    if checkpoint_path:
        cleanup_checkpoint(checkpoint_path)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    return 0

if __name__ == "__main__":
    exit(main())