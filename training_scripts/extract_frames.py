import os
import sys

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import cv2
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, 
                             QGroupBox, QMessageBox, QAction, QProgressDialog, QFileDialog, QSlider)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThreadPool, QRunnable, pyqtSlot
from functools import partial

# Add project root to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add training_scripts directory for CropLabel import when running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CropLabel import CNNCropLabel, CNN_WIDTH, CNN_HEIGHT

def apply_clahe(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bgr_enhanced


class ExtractionWorker(QRunnable):
    def __init__(self, video_path, frames, crop_rect, output_folder, video_name):
        super().__init__()
        self.video_path = video_path
        self.frames = frames
        self.crop_rect = crop_rect
        self.output_folder = output_folder
        self.video_name = video_name

    def run(self):
        print(f"Starting background extraction for {self.video_name}...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path} in background worker.")
            return

        # Expect a quad [[x,y], ...] (TL, TR, BR, BL).
        if not (isinstance(self.crop_rect, (list, tuple)) and len(self.crop_rect) == 4):
            print(f"Error: Expected quad ROI for worker but got: {self.crop_rect!r}")
            cap.release()
            return
        pts = [(int(p[0]), int(p[1])) for p in self.crop_rect]

        count = 0
        
        for frame_num in self.frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_num} from {self.video_name}")
                continue

            # Warp the quad to CNN size (256x64)
            try:
                src_pts = np.float32(pts)
                dst_pts = np.float32([[0, 0], [CNN_WIDTH, 0], [CNN_WIDTH, CNN_HEIGHT], [0, CNN_HEIGHT]])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (CNN_WIDTH, CNN_HEIGHT), flags=cv2.INTER_LINEAR)
                resized = warped
            except Exception as e:
                print(f"Warning: Could not warp quad ROI for frame {frame_num}: {e}. Skipping frame.")
                continue
            
            out_name = f"{self.video_name}_{frame_num}.jpg"
            out_path = os.path.join(self.output_folder, out_name)
            cv2.imwrite(out_path, resized)
            count += 1
            
        cap.release()
        print(f"Finished background extraction for {self.video_name}: {count} frames.")

class ExtractionWindow(QWidget):
    def __init__(self, video_path, frames_to_extract, output_folder, video_name, start_frame, end_frame, next_callback, skip_callback, default_roi=None, progress_text=""):
        super().__init__()
        self.video_path = video_path
        self.frames_to_extract = sorted(list(set(frames_to_extract)))
        self.output_folder = output_folder
        self.video_name = video_name
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.next_callback = next_callback
        self.skip_callback = skip_callback
        self.default_roi = default_roi
        self.progress_text = progress_text
        
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {video_path}")
            self.skip_callback()
            return

        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        
        self.current_frame_index = self.start_frame
        self.last_loaded_frame = -1
        self.current_raw_frame = None
        self.crop_coords = None
        self.read_timeout_warning_threshold = 5.0
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(100)
        self.pending_slider_action = None
        self.slider_timer.timeout.connect(self._execute_pending_slider_action)
        
        self.setWindowTitle(f"Extracting: {video_name} ({len(self.frames_to_extract)} frames)")
        
        success, first_frame = self.video.read()
        if success:
            self.original_h = first_frame.shape[0]
            self.original_w = first_frame.shape[1]
        else:
            print(f"Warning: Could not read first frame of {video_name}. Using default 1280x720.")
            self.original_h = 720
            self.original_w = 1280
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate optimal window size
        target_video_h = max(400, min(800, int(self.original_h * 0.7)))
        target_video_w = int(target_video_h * (self.original_w / self.original_h))
        controls_width = 220
        ui_overhead = 180
        optimal_width = target_video_w + controls_width
        optimal_height = target_video_h + ui_overhead
        self.resize(optimal_width, optimal_height)
        
        main_layout = self._setup_main_ui()
        if success:
            self.video_label.set_original_size(self.original_w, self.original_h)
            
            # Handle default ROI (carryover). Only accept a quad: [[x,y], ...] (TL, TR, BR, BL)
            if self.default_roi:
                try:
                    if isinstance(self.default_roi, (list, tuple)) and len(self.default_roi) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in self.default_roi):
                        self.video_label.set_points([(int(p[0]), int(p[1])) for p in self.default_roi])
                    else:
                        print(f"Invalid default ROI (expected quad [[x,y],...]): {self.default_roi}. Resetting to standard.")
                        self.video_label.reset_crop_to_standard()
                except Exception:
                    print(f"Invalid default ROI: {self.default_roi}. Resetting to standard.")
                    self.video_label.reset_crop_to_standard()
            else:
                self.video_label.reset_crop_to_standard()

        self.setLayout(main_layout)
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
        self.video_label = CNNCropLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(600, 300)
        self.video_label.crop_selected.connect(self.on_crop_selected)
        video_layout.addWidget(self.video_label)
        
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setToolTip("Go to previous frame (Left arrow)")
        self.prev_action = QAction("Previous frame", self)
        self.prev_action.setShortcut("Left")
        self.prev_action.triggered.connect(self.previous_frame)
        self.addAction(self.prev_action)
        
        self.frame_info_label = QLabel(f"Frame: {self.start_frame} / {self.total_frames}")
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
        
        scrub_layout = QHBoxLayout()
        self.scrub_label = QLabel("Frame:")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(self.start_frame, self.end_frame - 1)
        self.frame_slider.setValue(self.start_frame)
        self.frame_slider.setToolTip("Drag to navigate to a specific frame")
        self.frame_slider.valueChanged.connect(
            partial(self._debounced_slider, self.frame_slider, self.seek_to_frame)
        )
        scrub_layout.addWidget(self.scrub_label)
        scrub_layout.addWidget(self.frame_slider)
        video_layout.addLayout(scrub_layout)
        
        self.jump_back_action = QAction("Jump Back 10 Frames", self)
        self.jump_back_action.setShortcut("Ctrl+Left")
        self.jump_back_action.triggered.connect(
            lambda: self.seek_to_frame(max(self.start_frame, self.current_frame_index - 10))
        )
        self.addAction(self.jump_back_action)
        
        self.jump_forward_action = QAction("Jump Forward 10 Frames", self)
        self.jump_forward_action.setShortcut("Ctrl+Right")
        self.jump_forward_action.triggered.connect(
            lambda: self.seek_to_frame(min(self.end_frame - 1, self.current_frame_index + 10))
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
        
        if self.progress_text:
            progress_label = QLabel(self.progress_text)
            progress_label.setStyleSheet("font-weight: bold; color: #aaa;")
            progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            controls_layout.addWidget(progress_label)

        info_label = QLabel("CNN: 256x64 (4:1) | CLAHE auto-enabled")
        info_label.setStyleSheet(
            "background-color: #2a4a6a; color: white; padding: 4px; border-radius: 3px; font-size: 10px;"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        controls_layout.addWidget(info_label)
        
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
        
        self.copy_roi_button = QPushButton("Copy ROI to Clipboard")
        self.copy_roi_button.setToolTip("Copy current ROI as 'x,y,w,h' for use with --roi argument")
        self.copy_roi_button.clicked.connect(self.copy_roi_to_clipboard)
        self.copy_roi_button.setStyleSheet("font-size: 10px; padding: 2px;")
        crop_layout.addWidget(self.copy_roi_button)
        
        crop_group.setLayout(crop_layout)
        controls_layout.addWidget(crop_group)
        
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
        
        range_group = QGroupBox("Frame Range")
        range_group.setStyleSheet(group_box_style)
        range_layout = QVBoxLayout()
        range_layout.setSpacing(2)
        range_layout.setContentsMargins(6, 6, 6, 6)
        range_info = QLabel(f"{self.start_frame} - {self.end_frame}\n({self.end_frame - self.start_frame} frames)")
        range_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_info.setStyleSheet("font-size: 10px;")
        range_layout.addWidget(range_info)
        range_group.setLayout(range_layout)
        controls_layout.addWidget(range_group)

        self.preview_button = QPushButton("CLAHE: On")
        self.preview_button.setToolTip("Toggle CLAHE preview")
        self.preview_button.setCheckable(True)
        self.preview_button.setChecked(True)
        self.preview_button.setMaximumHeight(28)
        self.preview_button.setStyleSheet("font-size: 10px; padding: 2px;")
        self.preview_button.toggled.connect(self._update_preview_button_text)
        self.preview_button.clicked.connect(self.update_display_frame)
        controls_layout.addWidget(self.preview_button)
        
        self.preview_crop_button = QPushButton("Preview Crop")
        self.preview_crop_button.setToolTip("Show final cropped output")
        self.preview_crop_button.setCheckable(True)
        self.preview_crop_button.setChecked(False)
        self.preview_crop_button.setMaximumHeight(28)
        self.preview_crop_button.setStyleSheet("font-size: 10px; padding: 2px;")
        self.preview_crop_button.clicked.connect(self.update_display_frame)
        controls_layout.addWidget(self.preview_crop_button)

        self.extract_btn = QPushButton("EXTRACT && GO NEXT")
        self.extract_btn.setStyleSheet(
            "background-color: #4a7a4a; color: white; padding: 10px; font-weight: bold;"
        )
        self.extract_btn.clicked.connect(self.extract_frames)
        controls_layout.addWidget(self.extract_btn)
        
        self.skip_btn = QPushButton("SKIP VIDEO")
        self.skip_btn.setStyleSheet(
            "background-color: #7a4a4a; color: white; padding: 5px;"
        )
        self.skip_btn.clicked.connect(self.skip_video)
        controls_layout.addWidget(self.skip_btn)
        
        controls_layout.addStretch()
        return controls_layout

    def _update_preview_button_text(self, checked):
        if checked:
            self.preview_button.setText("CLAHE: On")
        else:
            self.preview_button.setText("CLAHE: Off")

    def on_crop_selected(self, points):
        # `points` is a list of 4 (x,y) points (TL, TR, BR, BL). Compute bounding rect for preview/cropping.
        if not points or len(points) != 4:
            return
        pts = [(int(p[0]), int(p[1])) for p in points]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(1, max_x - min_x)
        h = max(1, max_y - min_y)
        self.crop_coords = (min_x, min_y, w, h)
        # Keep the quad too for copying and saving
        self.crop_coords_quad = pts
        actual_ratio = w / h if h > 0 else 0
        self.crop_dims_label.setText(
            f"Crop: {w}x{h} pixels\nRatio: {actual_ratio:.2f}:1\nPosition: ({min_x}, {min_y})\nQuad: {pts}"
        )
        self.update_display_frame()

    def reset_roi(self):
        self.video_label.reset_crop_to_standard()

    def copy_roi_to_clipboard(self):
        # Only support copying quad ROIs as JSON
        if getattr(self, 'crop_coords_quad', None):
            import json as _json
            roi_str = _json.dumps(self.crop_coords_quad)
            QApplication.clipboard().setText(roi_str)
            original_text = self.copy_roi_button.text()
            self.copy_roi_button.setText("Copied!")
            QTimer.singleShot(1000, lambda: self.copy_roi_button.setText(original_text))
        else:
            QMessageBox.warning(self, "No ROI", "Please define a quad ROI before copying to clipboard.")

    def seek_to_frame(self, frame_num):
        self.current_frame_index = max(self.start_frame, min(frame_num, self.end_frame - 1))
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
            self.prev_button.setEnabled(self.current_frame_index > self.start_frame)
            self.next_button.setEnabled(self.current_frame_index < self.end_frame - 1)
            self.frame_info_label.setText(
                f"Frame: {self.current_frame_index} / {self.total_frames}"
            )
            self.update_display_frame()
        
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)

    def update_display_frame(self):
        if self.current_raw_frame is None:
            self.video_label.setText("No frame loaded")
            return
        preview_frame = self.current_raw_frame.copy()
        if self.preview_button.isChecked():
            preview_frame = apply_clahe(preview_frame)
        self._update_image(preview_frame)

    def _update_image(self, frame):
        if self.preview_crop_button.isChecked() and self.crop_coords:
            x, y, w, h = self.crop_coords
            h_max, w_max = frame.shape[:2]
            crop_x = max(0, min(x, w_max - 1))
            crop_y = max(0, min(y, h_max - 1))
            crop_w = max(1, min(w, w_max - crop_x))
            crop_h = max(1, min(h, h_max - crop_y))
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            current_h, current_w = cropped.shape[:2]
            if current_w > CNN_WIDTH or current_h > CNN_HEIGHT:
                frame = cv2.resize(cropped, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_AREA)
            else:
                frame = cv2.resize(cropped, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
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
        if self.current_frame_index < self.end_frame - 1:
            self.current_frame_index += 1
            self.load_frame_from_video()

    def previous_frame(self):
        if self.current_frame_index > self.start_frame:
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
                
                # Convert mouse position to original video coordinates
                img_x = (zoom_center.x() - off_x) / scale + self.pan_offset_x
                img_y = (zoom_center.y() - off_y) / scale + self.pan_offset_y
                
                # Calculate new visible dimensions after zoom
                new_visible_w = self.original_w / self.zoom_level
                new_visible_h = self.original_h / self.zoom_level
                
                # Calculate where mouse will be in the new visible region (as fraction)
                # We want the mouse to stay at the same relative position in the label
                mouse_rel_x = (zoom_center.x() - off_x) / (visible_w * scale)
                mouse_rel_y = (zoom_center.y() - off_y) / (visible_h * scale)
                
                # Set pan so that img_x, img_y appears at that same relative position
                self.pan_offset_x = img_x - (mouse_rel_x * new_visible_w)
                self.pan_offset_y = img_y - (mouse_rel_y * new_visible_h)
                
                # Clamp to valid range
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
        if self.zoom_level <= 1.0: return
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

    def extract_frames(self):
        # Require a quad ROI from the label
        if not getattr(self.video_label, 'crop_points', None):
            QMessageBox.warning(self, "Warning", "Please define a quad crop region first.")
            return

        # Pass the quad back to controller to handle background extraction
        self.video.release()
        self.close()
        pts = getattr(self.video_label, 'crop_points', None)
        if not pts:
            QMessageBox.warning(self, "Warning", "Please define a quad crop region first.")
            return
        # Normalize to list-of-lists of ints and pass
        quad = [[int(p[0]), int(p[1])] for p in pts]
        self.next_callback(quad)

    def skip_video(self):
        self.video.release()
        self.close()
        self.skip_callback()


class ExtractionController:
    def __init__(self, csv_path, video_dir, output_dir):
        self.df = pd.read_csv(csv_path)
        self.csv_path = Path(csv_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.unique_videos = self.df['filename'].unique()
        self.current_video_idx = 0
        self.last_roi = None
        self.threadpool = QThreadPool()
        # Store normalized ROIs as quads: [[x,y], [x,y], [x,y], [x,y]]
        self.roi_by_video: dict[str, list] = {}
        print(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")
        
    def _write_all_data_json(self) -> None:
        """Write authoritative metadata & per-video ranges/ROIs to data/all_data.json"""
        out_dir = self.csv_path.parent if hasattr(self, "csv_path") else Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "all_data.json"

        summary = {
            "created_at": datetime.now(timezone.utc).isoformat() + "Z",
            "num_samples": int(len(self.df)),
            "num_videos": int(len(self.unique_videos)),
            "weight_range": [
                float(self.df["weight"].min()) if "weight" in self.df.columns else None,
                float(self.df["weight"].max()) if "weight" in self.df.columns else None,
            ],
            "video_ranges": {},
            "rois": {},
        }

        if "filename" in self.df.columns and "frame_number" in self.df.columns:
            for name, group in self.df.groupby("filename"):
                summary["video_ranges"][name] = {
                    "min_frame": int(group["frame_number"].min()),
                    "max_frame": int(group["frame_number"].max())
                }

        for name, roi in self.roi_by_video.items():
            # roi must be a list of 4 [x,y] points (quad)
            try:
                pts = [[int(p[0]), int(p[1])] for p in roi]
                summary["rois"][name] = {"quad": pts}
            except Exception:
                summary["rois"][name] = None

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            
    def start(self):
        self.process_next_video()
        
    def process_next_video(self):
        if self.current_video_idx >= len(self.unique_videos):
            if self.threadpool.activeThreadCount() > 0:
                QMessageBox.information(None, "Finishing Up", "Waiting for background extractions to complete...")
                self.threadpool.waitForDone()
            QMessageBox.information(None, "Done", "All videos processed!")
            sys.exit(0)
            
        video_name = self.unique_videos[self.current_video_idx]
        video_path = self.video_dir / video_name
        
        if not video_path.exists():
            print(f"Video not found: {video_path}. Skipping...")
            self.current_video_idx += 1
            self.process_next_video()
            return
            
        video_data = self.df[self.df['filename'] == video_name]
        frames = video_data['frame_number'].tolist()
        
        # Check for existing outputs
        existing_files = [f for f in frames if (self.output_dir / f"{video_name}_{f}.jpg").exists()]
        if existing_files:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText(f"Found {len(existing_files)} existing extracted frames for {video_name}.")
            msg.setInformativeText("Do you want to overwrite them?")
            msg.setWindowTitle("Overwrite Check")
            overwrite_btn = msg.addButton("Overwrite", QMessageBox.AcceptRole)
            skip_btn = msg.addButton("Skip Video", QMessageBox.RejectRole)
            cancel_btn = msg.addButton("Cancel All", QMessageBox.DestructiveRole)
            msg.exec_()
            
            if msg.clickedButton() == skip_btn:
                print(f"Skipping {video_name}")
                self.current_video_idx += 1
                self.process_next_video()
                return
            elif msg.clickedButton() == cancel_btn:
                sys.exit(0)
            # If overwrite, just proceed
        
        if 'start_frame' in video_data.columns and 'end_frame' in video_data.columns:
            start_frame = int(video_data['start_frame'].iloc[0])
            end_frame = int(video_data['end_frame'].iloc[0])
        else:
            start_frame = 0
            end_frame = max(frames) + 100 if frames else 1000
            
        progress_text = f"Video {self.current_video_idx + 1} of {len(self.unique_videos)}"
        
        self.window = ExtractionWindow(
            str(video_path), 
            frames, 
            str(self.output_dir), 
            video_name,
            start_frame,
            end_frame,
            self.on_video_complete,
            self.on_video_skipped,
            default_roi=self.last_roi,
            progress_text=progress_text
        )
        self.window.show()
        
    def on_video_complete(self, roi):
        self.last_roi = roi

        video_name = self.unique_videos[self.current_video_idx]
        if roi:
            # Expect a quad: list/tuple of four (x,y) pairs
            if not (isinstance(roi, (list, tuple)) and len(roi) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in roi)):
                print(f"Invalid ROI received (expected quad): {roi!r}")
            else:
                pts = [[int(p[0]), int(p[1])] for p in roi]
                self.roi_by_video[video_name] = pts
                self._write_all_data_json()  # write/update JSON as videos finish

        # Start background extraction for the completed video
        video_name = self.unique_videos[self.current_video_idx]
        video_path = str(self.video_dir / video_name)
        video_data = self.df[self.df['filename'] == video_name]
        frames = video_data['frame_number'].tolist()

        # Pass quad points to worker
        worker = ExtractionWorker(video_path, frames, self.roi_by_video.get(video_name, None), str(self.output_dir), video_name)
        self.threadpool.start(worker)

        self.current_video_idx += 1
        self.process_next_video()
        
    def on_video_skipped(self):
        print(f"Skipped video: {self.unique_videos[self.current_video_idx]}")
        self.current_video_idx += 1
        self.process_next_video()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', '-c', default='data/all_data.csv', help='Path to labels CSV (will prompt if not provided)')
    parser.add_argument('--video_dir', '-v', default=None, help='Directory containing video files (will prompt if not provided)')
    parser.add_argument('--output_dir', '-o', default='data/images/', help='Output directory for images (will prompt if not provided)')
    args = parser.parse_args()
    
    app = QApplication.instance() or QApplication(sys.argv)

    csv_path = args.csv
    if not csv_path or not os.path.exists(csv_path):
        if csv_path and not os.path.exists(csv_path):
            QMessageBox.warning(None, "CSV not found", f"CSV file not found: {csv_path}\nPlease select a CSV file.")
        csv_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select labels CSV (with ALL data)",
            os.getcwd(),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not csv_path:
            QMessageBox.critical(None, "Error", "No CSV selected. Exiting.")
            return 1

    video_dir = args.video_dir
    if not video_dir or not os.path.isdir(video_dir):
        if video_dir and not os.path.isdir(video_dir):
            QMessageBox.warning(None, "Video dir not found", f"Video directory not found: {video_dir}\nPlease select a directory.")
        video_dir = QFileDialog.getExistingDirectory(
            None,
            "Select video directory",
            os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not video_dir:
            QMessageBox.critical(None, "Error", "No video directory selected. Exiting.")
            return 1

    output_dir = args.output_dir
    if not output_dir:
         output_dir = QFileDialog.getExistingDirectory(
            None, "Select output directory for images", os.getcwd(),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
    
    if not output_dir:
        QMessageBox.critical(None, "Error", "No output directory selected. Exiting.")
        return 1

    # Final sanity checks
    if not os.path.exists(csv_path):
        QMessageBox.critical(None, "Error", f"CSV file not found: {csv_path}")
        return 1

    controller = ExtractionController(csv_path, video_dir, output_dir)
    controller.start()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
