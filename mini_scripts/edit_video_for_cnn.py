import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import cv2
import numpy as np
import time
import subprocess
import argparse
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, 
                             QGroupBox, QMessageBox, QAction, QApplication, 
                             QFileDialog, QProgressDialog, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal
from qtrangeslider import QRangeSlider
from functools import partial

CNN_WIDTH = 256
CNN_HEIGHT = 64
CNN_ASPECT_RATIO = CNN_WIDTH / CNN_HEIGHT
MIN_HEIGHT = 32  # Minimum crop height to prevent microscopic boxes

def apply_clahe(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bgr_enhanced

class CNNCropLabel(QLabel):
    crop_selected = pyqtSignal(int, int, int, int)
    
    def __init__(self) -> None:
        super().__init__()
        self.crop_rect: tuple[int, int, int, int] | None = None  # (x, y, w, h) in image coordinates
        self.original_w: int = 0
        self.original_h: int = 0
        self.aspect_ratio = CNN_ASPECT_RATIO
        self.setMouseTracking(True)
        
        self.action = None
        self.drag_start_pos = None
        self.initial_crop_rect = None
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset_x = 0  # in original image coordinates
        self.pan_offset_y = 0
        
        # Pan drag state
        self.pan_dragging = False
        self.pan_drag_start = None
        self.pan_drag_start_offset = None

        # Standard ROI defaults
        self.std_x = 625
        self.std_y = 300
        self.std_w = 893
        self.std_h = 223
    
    def set_standard_roi(self, x: int, y: int, w: int, h: int) -> None:
        self.std_x = x
        self.std_y = y
        self.std_w = w
        self.std_h = h
    
    def set_original_size(self, w: int, h: int) -> None:
        self.original_w = w
        self.original_h = h
        if self.crop_rect is None:
            self.reset_crop_to_standard()
    
    def reset_crop_to_standard(self):
        if not self.original_w or not self.original_h:
            return
        
        w, h = self.original_w, self.original_h
        crop_w = self.std_w if w > self.std_w else int(w * 0.5)
        crop_h = self.std_h if h > self.std_h else int(crop_w / self.aspect_ratio)
        crop_x = self.std_x if w > self.std_x else (w - crop_w) // 2
        crop_y = self.std_y if h > self.std_y else (h - crop_h) // 2
        self.crop_rect = (crop_x, crop_y, crop_w, crop_h)
        self.crop_selected.emit(*self.crop_rect)
        self.update()
    
    def set_zoom(self, zoom_level: float, pan_x: float | None = None, pan_y: float | None = None) -> None:
        """Set zoom level and optional pan offset."""
        if not self.original_w or not self.original_h:
            return
        self.zoom_level = max(1.0, min(zoom_level, 10.0))  # Clamp between 1x and 10x
        
        if pan_x is not None:
            # Clamp pan offset to keep visible region within bounds
            max_pan_x = max(0, self.original_w - self.original_w / self.zoom_level)
            self.pan_offset_x = max(0, min(pan_x, max_pan_x))
        
        if pan_y is not None:
            max_pan_y = max(0, self.original_h - self.original_h / self.zoom_level)
            self.pan_offset_y = max(0, min(pan_y, max_pan_y))
        
        self.update()
    
    def _get_display_params(self) -> tuple[float, int, int, float, float] | None:
        if not self.original_w or not self.original_h:
            return None
        label_w = self.width()
        label_h = self.height()
        
        visible_w = self.original_w / self.zoom_level
        visible_h = self.original_h / self.zoom_level
        
        # Scale to fit the visible region in the label
        scale = min(label_w / visible_w, label_h / visible_h)
        display_w = int(visible_w * scale)
        display_h = int(visible_h * scale)
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2
        return scale, offset_x, offset_y, visible_w, visible_h

    def _image_to_label(self, img_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        params = self._get_display_params()
        if not params: return None
        scale, off_x, off_y, visible_w, visible_h = params
        x, y, w, h = img_rect
        
        # Adjust for pan offset (subtract because we're viewing a shifted region)
        x_rel = x - self.pan_offset_x
        y_rel = y - self.pan_offset_y
        
        lx = int(x_rel * scale) + off_x
        ly = int(y_rel * scale) + off_y
        lw = int(w * scale)
        lh = int(h * scale)
        return (lx, ly, lw, lh)

    def _label_to_image_pt(self, pos: QPoint) -> tuple[int, int] | tuple[None, None]:
        params = self._get_display_params()
        if not params: 
            return None, None
        scale, off_x, off_y, visible_w, visible_h = params
        
        # Convert label position to image-relative position
        ix_rel = int((pos.x() - off_x) / scale)
        iy_rel = int((pos.y() - off_y) / scale)
        
        # Add pan offset to get absolute image coordinates
        ix = ix_rel + int(self.pan_offset_x)
        iy = iy_rel + int(self.pan_offset_y)
        return ix, iy

    def mousePressEvent(self, event) -> None:  # event: QMouseEvent
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        if not self.crop_rect:
            return
        
        l_rect = self._image_to_label(self.crop_rect)
        if not l_rect:
            # If zoomed in, allow panning
            if self.zoom_level > 1.0:
                self.pan_dragging = True
                self.pan_drag_start = event.pos()
                self.pan_drag_start_offset = (self.pan_offset_x, self.pan_offset_y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        
        lx, ly, lw, lh = l_rect
        x, y = event.pos().x(), event.pos().y()
        
        handle_r = 10
        # Check handles
        if abs(x - lx) < handle_r and abs(y - ly) < handle_r:
            self.action = 'resize_tl'
        elif abs(x - (lx + lw)) < handle_r and abs(y - ly) < handle_r:
            self.action = 'resize_tr'
        elif abs(x - lx) < handle_r and abs(y - (ly + lh)) < handle_r:
            self.action = 'resize_bl'
        elif abs(x - (lx + lw)) < handle_r and abs(y - (ly + lh)) < handle_r:
            self.action = 'resize_br'
        elif lx < x < lx + lw and ly < y < ly + lh:
            self.action = 'move'
        else:
            # Click outside crop box - allow panning if zoomed in
            if self.zoom_level > 1.0:
                self.pan_dragging = True
                self.pan_drag_start = event.pos()
                self.pan_drag_start_offset = (self.pan_offset_x, self.pan_offset_y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.action = None
            return
            
        self.drag_start_pos = event.pos()
        self.initial_crop_rect = self.crop_rect

    def mouseMoveEvent(self, event) -> None:  # event: QMouseEvent
        if not self.original_w or not self.original_h:
            return
        # Handle pan dragging first
        if self.pan_dragging and self.pan_drag_start and self.pan_drag_start_offset:
            # Calculate drag delta in screen pixels
            dx_screen = event.pos().x() - self.pan_drag_start.x()
            dy_screen = event.pos().y() - self.pan_drag_start.y()
            
            # Convert screen delta to image coordinates
            params = self._get_display_params()
            if params:
                scale, off_x, off_y, visible_w, visible_h = params
                dx_img = -dx_screen / scale  # Negative because dragging right should pan left
                dy_img = -dy_screen / scale
                
                # Apply to original pan offset
                new_pan_x = self.pan_drag_start_offset[0] + dx_img
                new_pan_y = self.pan_drag_start_offset[1] + dy_img
                
                # Clamp to valid range
                max_pan_x = max(0, self.original_w - visible_w)
                max_pan_y = max(0, self.original_h - visible_h)
                self.pan_offset_x = max(0, min(new_pan_x, max_pan_x))
                self.pan_offset_y = max(0, min(new_pan_y, max_pan_y))
                
                # Notify parent window
                parent = self.parent()
                if parent and hasattr(parent, 'on_pan_drag'):
                    parent.on_pan_drag(self.pan_offset_x, self.pan_offset_y)
                
                self.update()
            return
        
        if not self.crop_rect: return
        
        # Cursor update
        l_rect = self._image_to_label(self.crop_rect)
        if l_rect and not self.action:
            lx, ly, lw, lh = l_rect
            x, y = event.pos().x(), event.pos().y()
            handle_r = 10
            if (abs(x - lx) < handle_r and abs(y - ly) < handle_r) or \
               (abs(x - (lx + lw)) < handle_r and abs(y - (ly + lh)) < handle_r):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif (abs(x - (lx + lw)) < handle_r and abs(y - ly) < handle_r) or \
                 (abs(x - lx) < handle_r and abs(y - (ly + lh)) < handle_r):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif lx < x < lx + lw and ly < y < ly + lh:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                # Show open hand cursor when zoomed and over video (but not crop box)
                if self.zoom_level > 1.0:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
        
        if not self.action: return

        curr_ix, curr_iy = self._label_to_image_pt(event.pos())
        if curr_ix is None: return
        
        if not self.initial_crop_rect:
            return
        ix, iy, iw, ih = self.initial_crop_rect
        
        if self.action == 'move':
            # Calculate delta in image coords
            start_ix, start_iy = self._label_to_image_pt(self.drag_start_pos)
            if start_ix is None or start_iy is None:
                return
            dx = curr_ix - start_ix
            dy = curr_iy - start_iy
            
            nx = ix + dx
            ny = iy + dy
            
            nx = max(0, min(nx, self.original_w - iw))
            ny = max(0, min(ny, self.original_h - ih))
            self.crop_rect = (nx, ny, iw, ih)
            
        elif 'resize' in self.action:
            if self.action == 'resize_br':
                fixed_x, fixed_y = ix, iy
                moving_x, moving_y = curr_ix, curr_iy
            elif self.action == 'resize_tl':
                fixed_x, fixed_y = ix + iw, iy + ih
                moving_x, moving_y = curr_ix, curr_iy
            elif self.action == 'resize_tr':
                fixed_x, fixed_y = ix, iy + ih
                moving_x, moving_y = curr_ix, curr_iy
            elif self.action == 'resize_bl':
                fixed_x, fixed_y = ix + iw, iy
                moving_x, moving_y = curr_ix, curr_iy
            
            w_cand = abs(moving_x - fixed_x)
            h_cand = int(w_cand / self.aspect_ratio)
            
            if h_cand < MIN_HEIGHT:
                h_cand = MIN_HEIGHT
                w_cand = int(h_cand * self.aspect_ratio)
                
            if self.action == 'resize_br':
                max_w = self.original_w - ix
                max_h = self.original_h - iy
                if w_cand > max_w: w_cand = max_w; h_cand = int(w_cand / self.aspect_ratio)
                if h_cand > max_h: h_cand = max_h; w_cand = int(h_cand * self.aspect_ratio)
                self.crop_rect = (ix, iy, w_cand, h_cand)
                
            elif self.action == 'resize_tl':
                br_x, br_y = ix + iw, iy + ih
                max_w = br_x
                max_h = br_y
                if w_cand > max_w: w_cand = max_w; h_cand = int(w_cand / self.aspect_ratio)
                if h_cand > max_h: h_cand = max_h; w_cand = int(h_cand * self.aspect_ratio)
                self.crop_rect = (br_x - w_cand, br_y - h_cand, w_cand, h_cand)

            elif self.action == 'resize_tr':
                bl_x, bl_y = ix, iy + ih
                max_w = self.original_w - bl_x
                max_h = bl_y
                if w_cand > max_w: w_cand = max_w; h_cand = int(w_cand / self.aspect_ratio)
                if h_cand > max_h: h_cand = max_h; w_cand = int(h_cand * self.aspect_ratio)
                self.crop_rect = (bl_x, bl_y - h_cand, w_cand, h_cand)

            elif self.action == 'resize_bl':
                tr_x, tr_y = ix + iw, iy
                max_w = tr_x
                max_h = self.original_h - tr_y
                if w_cand > max_w: w_cand = max_w; h_cand = int(w_cand / self.aspect_ratio)
                if h_cand > max_h: h_cand = max_h; w_cand = int(h_cand * self.aspect_ratio)
                self.crop_rect = (tr_x - w_cand, tr_y, w_cand, h_cand)

        self.update()
        self.crop_selected.emit(*self.crop_rect)

    def mouseReleaseEvent(self, event) -> None:  # event: QMouseEvent
        if self.pan_dragging:
            self.pan_dragging = False
            self.pan_drag_start = None
            self.pan_drag_start_offset = None
            # Restore cursor based on zoom state
            if self.zoom_level > 1.0:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        
        self.action = None
        if self.crop_rect:
            self.crop_selected.emit(*self.crop_rect)

    def paintEvent(self, event) -> None:  # event: QPaintEvent
        super().paintEvent(event)
        # Get parent window to check if crop preview is active
        parent = self.parent()
        show_overlay = True
        if parent and hasattr(parent, 'preview_crop_button'):
            show_overlay = not parent.preview_crop_button.isChecked()
        
        if self.crop_rect and show_overlay:
            l_rect = self._image_to_label(self.crop_rect)
            if l_rect:
                lx, ly, lw, lh = l_rect
                painter = QPainter(self)
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
                painter.drawRect(lx, ly, lw, lh)
                
                # guidelines
                grid_color = QColor(255, 255, 255, 80)  # semi-transparent white
                painter.setPen(QPen(grid_color, 1, Qt.PenStyle.DashLine))
                third_w = lw // 3
                painter.drawLine(lx + third_w, ly, lx + third_w, ly + lh)
                painter.drawLine(lx + 2 * third_w, ly, lx + 2 * third_w, ly + lh)
                
                # Handles
                painter.setBrush(QColor(255, 255, 0))
                r = 6
                painter.drawEllipse(QPoint(lx, ly), r, r)
                painter.drawEllipse(QPoint(lx + lw, ly), r, r)
                painter.drawEllipse(QPoint(lx, ly + lh), r, r)
                painter.drawEllipse(QPoint(lx + lw, ly + lh), r, r)
                
                painter.setPen(QPen(QColor(255, 255, 0), 1))
                ratio_text = f"{self.crop_rect[2]}x{self.crop_rect[3]}"
                painter.drawText(lx + 5, ly - 5, ratio_text)
    
    def wheelEvent(self, event) -> None:  # event: QWheelEvent
        """Forward wheel events to parent window for zoom handling."""
        # Emit a signal or call parent method if available
        parent = self.parent()
        if parent and hasattr(parent, 'handle_wheel_event'):
            parent.handle_wheel_event(event)
            event.accept()
        else:
            super().wheelEvent(event)

class CNNEditWindow(QWidget):
    def __init__(self, video_capture, video_path: str, output_folder: str | None = None, default_roi: tuple[int, int, int, int] | None = None) -> None:
        super().__init__()
        self.video = video_capture
        self.video_path = video_path
        self.output_folder = output_folder
        self.default_roi = default_roi
        self.current_frame_index = 0
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.last_loaded_frame = -1
        self.current_raw_frame = None
        self.start_frame = 0
        self.end_frame = self.total_frames
        self.crop_coords = None
        self.read_timeout_warning_threshold = 5.0
        
        # Zoom and pan state
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(100)
        self.pending_slider_action = None
        self.slider_timer.timeout.connect(self._execute_pending_slider_action)
        self.setWindowTitle("CNN Video Preprocessing Tool")
        self.export_completed = False
        
        # Get video dimensions to calculate optimal window size
        success, first_frame = self.video.read()
        if success:
            self.original_h = first_frame.shape[0]
            self.original_w = first_frame.shape[1]
        else:
            self.original_h = 720
            self.original_w = 1280
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Calculate optimal window size
        # Target video display height: 70% of video height, min 400px, max 800px
        target_video_h = max(400, min(800, int(self.original_h * 0.7)))
        target_video_w = int(target_video_h * (self.original_w / self.original_h))
        
        # Add space for controls (220px width) and UI elements (~180px height for sliders/buttons)
        controls_width = 220
        ui_overhead = 180
        optimal_width = target_video_w + controls_width
        optimal_height = target_video_h + ui_overhead
        
        self.resize(optimal_width, optimal_height)
        
        main_layout = self._setup_main_ui()
        if success:
            self.video_label.set_original_size(self.original_w, self.original_h)
        self.setLayout(main_layout)
        self.load_frame_from_video()
    
    def _debounced_slider(self, slider, value_changed_func) -> None:
        self.slider_timer.stop()
        self.pending_slider_action = lambda: value_changed_func(slider.value())
        self.slider_timer.start()
    
    def _execute_pending_slider_action(self) -> None:
        if self.pending_slider_action:
            self.pending_slider_action()
            self.pending_slider_action = None
    
    def _setup_main_ui(self) -> QHBoxLayout:
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        self.video_label = CNNCropLabel()
        if self.default_roi:
            self.video_label.set_standard_roi(*self.default_roi)
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
        self.frame_info_label = QLabel(f"Frame: 1 / {self.total_frames}")
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
        self.frame_slider.setRange(0, self.total_frames - 1)
        self.frame_slider.setValue(0)
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
        
        # Pan shortcuts (when zoom > 1.0)
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
        
        trim_layout = QHBoxLayout()
        self.trim_label = QLabel("Trim:  ")
        self.trim_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.trim_slider.setStyleSheet("QRangeSlider {qproperty-barColor: #447;}")
        self.trim_slider.setRange(0, self.total_frames - 1)
        self.trim_slider.setValue((0, self.total_frames - 1))
        self.trim_slider.setToolTip("Drag handles to set start and end points")
        self.trim_slider.valueChanged.connect(
            partial(self._debounced_slider, self.trim_slider, self.apply_trim)
        )
        trim_layout.addWidget(self.trim_label)
        trim_layout.addWidget(self.trim_slider)
        video_layout.addLayout(trim_layout)
        main_layout.addLayout(video_layout, 3)
        controls_layout = self._setup_controls()
        main_layout.addLayout(controls_layout, 0)
        return main_layout
    
    def _setup_controls(self) -> QVBoxLayout:
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(6)  # Reduce spacing between elements
        controls_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        
        # Unified stylesheet for group boxes
        group_box_style = (
            "QGroupBox { border: 1px solid #555; border-radius: 3px; margin-top: 6px; padding-top: 6px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }"
        )
        
        info_label = QLabel("CNN: 256x64 (4:1) | CLAHE auto-enabled")
        info_label.setStyleSheet(
            "background-color: #2a4a6a; color: white; padding: 4px; border-radius: 3px; font-size: 10px;"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        controls_layout.addWidget(info_label)
        
        # Keyboard shortcuts info
        shortcuts_group = QGroupBox("Shortcuts")
        shortcuts_group.setStyleSheet(group_box_style)
        shortcuts_layout = QVBoxLayout()
        shortcuts_layout.setSpacing(2)
        shortcuts_layout.setContentsMargins(6, 6, 6, 6)
        shortcuts_text = QLabel(
            "Left/Right arrow: Frame nav\n"
            "Ctrl+Left/Right arrow: Jump 10\n"
            "Wheel: Zoom\n"
            "Drag: Pan (when zoomed)\n"
            "Up/Down arrow, Shift+Left/Right arrow: Pan (when zoomed)"
        )
        shortcuts_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        shortcuts_text.setStyleSheet("font-size: 10px;")
        shortcuts_layout.addWidget(shortcuts_text)
        shortcuts_group.setLayout(shortcuts_layout)
        controls_layout.addWidget(shortcuts_group)

        # Crop info
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
        
        # Zoom controls
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
        self.zoom_in_button.clicked.connect(lambda: self.apply_zoom(self.zoom_level * 1.5))
        self.zoom_reset_button = QPushButton("⟲")
        self.zoom_reset_button.setToolTip("Reset zoom to 1x")
        self.zoom_reset_button.setMaximumHeight(18)
        self.zoom_reset_button.setMaximumWidth(35)
        self.zoom_reset_button.clicked.connect(lambda: self.apply_zoom(1.0, reset_pan=True))
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setToolTip("Zoom out (Mouse wheel down)")
        self.zoom_out_button.setMaximumHeight(18)
        self.zoom_out_button.setMaximumWidth(35)
        self.zoom_out_button.clicked.connect(lambda: self.apply_zoom(self.zoom_level / 1.5))
        
        zoom_btn_layout.addWidget(self.zoom_in_button)
        zoom_btn_layout.addWidget(self.zoom_reset_button)
        zoom_btn_layout.addWidget(self.zoom_out_button)
        zoom_main_layout.addLayout(zoom_btn_layout)
        zoom_group.setLayout(zoom_main_layout)
        controls_layout.addWidget(zoom_group)
        
        # Combined processing & output info
        # process_group = QGroupBox("Processing")
        # process_group.setStyleSheet(group_box_style)
        # process_layout = QVBoxLayout()
        # process_layout.setSpacing(2)
        # process_layout.setContentsMargins(6, 6, 6, 6)
        # process_info = QLabel("CLAHE: Clip 2.0, Grid 8x8\nOutput: 256x64 JPEG")
        # process_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # process_info.setStyleSheet("font-size: 10px;")
        # process_layout.addWidget(process_info)
        # process_group.setLayout(process_layout)
        # controls_layout.addWidget(process_group)
        
        # Trim info
        trim_group = QGroupBox("Trim")
        trim_group.setStyleSheet(group_box_style)
        trim_layout = QVBoxLayout()
        trim_layout.setSpacing(2)
        trim_layout.setContentsMargins(6, 6, 6, 6)
        self.trim_info_label = QLabel(f"0 - {self.total_frames - 1}\n({self.total_frames} frames)")
        self.trim_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trim_info_label.setStyleSheet("font-size: 10px;")
        trim_layout.addWidget(self.trim_info_label)
        trim_group.setLayout(trim_layout)
        controls_layout.addWidget(trim_group)

        # Preview buttons
        self.preview_button = QPushButton("CLAHE: On")
        self.preview_button.setToolTip("Toggle CLAHE preview")
        self.preview_button.setCheckable(True)
        self.preview_button.setChecked(True)
        self.preview_button.setMaximumHeight(28)
        self.preview_button.toggled.connect(self._update_preview_button_text)
        self.preview_button.clicked.connect(self.update_display_frame)
        controls_layout.addWidget(self.preview_button)
        
        self.preview_crop_button = QPushButton("Preview Crop")
        self.preview_crop_button.setToolTip("Show final cropped output")
        self.preview_crop_button.setCheckable(True)
        self.preview_crop_button.setChecked(False)
        self.preview_crop_button.setMaximumHeight(28)
        self.preview_crop_button.clicked.connect(self.update_display_frame)
        controls_layout.addWidget(self.preview_crop_button)

        self.save_button = QPushButton("Save Edits and Close")
        self.save_button.setStyleSheet(
            "background-color: #4a7a4a; color: white; padding: 8px; font-weight: bold;"
        )
        self.save_button.clicked.connect(self.save_frames)
        controls_layout.addWidget(self.save_button)
        
        controls_layout.addStretch()
        return controls_layout
    
    def _update_preview_button_text(self, checked: bool) -> None:
        if checked:
            self.preview_button.setText("CLAHE: On")
        else:
            self.preview_button.setText("CLAHE: Off")
    
    def on_crop_selected(self, x: int, y: int, w: int, h: int) -> None:
        self.crop_coords = (x, y, w, h)
        actual_ratio = w / h if h > 0 else 0
        self.crop_dims_label.setText(
            f"Crop: {w}x{h} pixels\nRatio: {actual_ratio:.2f}:1\nPosition: ({x}, {y})"
        )
        self.update_display_frame()
    
    def reset_roi(self) -> None:
        self.video_label.reset_crop_to_standard()
    
    def copy_roi_to_clipboard(self) -> None:
        if self.crop_coords:
            x, y, w, h = self.crop_coords
            roi_str = f"{x},{y},{w},{h}"
            QApplication.clipboard().setText(roi_str)
            
            # Show a temporary tooltip or message
            original_text = self.copy_roi_button.text()
            self.copy_roi_button.setText("Copied!")
            QTimer.singleShot(1000, lambda: self.copy_roi_button.setText(original_text))
    
    def seek_to_frame(self, frame_num: int) -> None:
        self.current_frame_index = max(self.start_frame, min(frame_num, self.end_frame - 1))
        self.load_frame_from_video()
    
    def apply_trim(self, values: tuple[int, int]) -> None:
        new_start, new_end = values
        if new_start >= new_end:
            new_start = new_end - 1
        new_start = max(0, min(new_start, self.total_frames - 1))
        new_end = max(1, min(new_end, self.total_frames))
        if new_start == self.start_frame and new_end == self.end_frame:
            return
        self.start_frame = new_start
        self.end_frame = new_end
        frame_count = self.end_frame - self.start_frame
        duration = frame_count / self.fps if self.fps > 0 else 0
        self.trim_info_label.setText(
            f"{self.start_frame} - {self.end_frame}\n({frame_count} frames, {duration:.1f}s)"
        )
        self.prev_button.setEnabled(self.current_frame_index > self.start_frame)
        self.next_button.setEnabled(self.current_frame_index < self.end_frame - 1)
        if self.current_frame_index < self.start_frame or self.current_frame_index >= self.end_frame:
            self.seek_to_frame(self.start_frame)
        elif self.last_loaded_frame != self.current_frame_index:
            self.load_frame_from_video()
    
    def load_frame_from_video(self) -> None:
        if self.last_loaded_frame != self.current_frame_index:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            time0 = time.time()
            success, frame = self.video.read()
            duration = time.time() - time0
            if (duration >= self.read_timeout_warning_threshold) or (not success):
                QMessageBox.critical(None, 'Error', 'Frame took too long to read. If using a file on a network share, consider making a local copy.')
                return
            else:
                self.current_raw_frame = frame
                self.last_loaded_frame = self.current_frame_index
        if self.current_raw_frame is not None:
            self.prev_button.setEnabled(self.current_frame_index > self.start_frame)
            self.next_button.setEnabled(self.current_frame_index < self.end_frame - 1)
            self.frame_info_label.setText(
                f"Frame: {self.current_frame_index + 1} / {self.total_frames}"
            )
            self.update_display_frame()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_index)
        self.frame_slider.blockSignals(False)
    
    def update_display_frame(self) -> None:
        if self.current_raw_frame is None:
            self.video_label.setText("No frame loaded")
            return
        preview_frame = self.current_raw_frame.copy()
        if self.preview_button.isChecked():
            preview_frame = apply_clahe(preview_frame)
        self._update_image(preview_frame)
    
    def _update_image(self, frame) -> None:  # frame: cv2 ndarray
        # Handle crop preview mode
        if self.preview_crop_button.isChecked() and self.crop_coords:
            x, y, w, h = self.crop_coords
            h_max, w_max = frame.shape[:2]
            
            # Clamp crop coordinates to frame bounds
            crop_x = max(0, min(x, w_max - 1))
            crop_y = max(0, min(y, h_max - 1))
            crop_w = max(1, min(w, w_max - crop_x))
            crop_h = max(1, min(h, h_max - crop_y))
            
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # Resize to final CNN dimensions (same as export)
            current_h, current_w = cropped.shape[:2]
            if current_w > CNN_WIDTH or current_h > CNN_HEIGHT:
                frame = cv2.resize(cropped, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_AREA)
            else:
                frame = cv2.resize(cropped, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
            # Apply zoom by cropping to the visible region
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
    
    def _cv2_to_qimage(self, cv_img) -> QImage:  # cv_img: cv2 ndarray
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        return QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def next_frame(self) -> None:
        if self.current_frame_index < self.end_frame - 1:
            self.current_frame_index += 1
            self.load_frame_from_video()
    
    def previous_frame(self) -> None:
        if self.current_frame_index > self.start_frame:
            self.current_frame_index -= 1
            self.load_frame_from_video()
    
    def handle_wheel_event(self, event) -> None:  # event: QWheelEvent
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        if delta > 0:
            new_zoom = self.zoom_level * 1.1
        else:
            new_zoom = self.zoom_level / 1.1
        
        mouse_pos = event.pos()
        self.apply_zoom(new_zoom, zoom_center=mouse_pos)
    
    def apply_zoom(self, new_zoom: float, reset_pan: bool = False, zoom_center: QPoint | None = None) -> None:
        """Apply zoom level and update display."""
        old_zoom = self.zoom_level
        self.zoom_level = max(1.0, min(new_zoom, 10.0))
        
        if reset_pan or self.zoom_level == 1.0:
            self.pan_offset_x = 0
            self.pan_offset_y = 0
        elif zoom_center and self.original_w and self.original_h:
            # Adjust pan to zoom towards mouse position
            params = self.video_label._get_display_params()
            if params:
                scale, off_x, off_y, visible_w, visible_h = params
                
                # Convert mouse position to image coordinates
                rel_x = (zoom_center.x() - off_x) / scale
                rel_y = (zoom_center.y() - off_y) / scale
                img_x = rel_x + self.pan_offset_x
                img_y = rel_y + self.pan_offset_y
                
                # Calculate new pan offset to keep mouse point stationary
                new_visible_w = self.original_w / self.zoom_level
                new_visible_h = self.original_h / self.zoom_level
                
                # Try to center the zoom on the mouse position
                self.pan_offset_x = img_x - new_visible_w / 2
                self.pan_offset_y = img_y - new_visible_h / 2
                
                # Clamp pan offsets
                max_pan_x = max(0, self.original_w - new_visible_w)
                max_pan_y = max(0, self.original_h - new_visible_h)
                self.pan_offset_x = max(0, min(self.pan_offset_x, max_pan_x))
                self.pan_offset_y = max(0, min(self.pan_offset_y, max_pan_y))
        
        self.video_label.set_zoom(self.zoom_level, self.pan_offset_x, self.pan_offset_y)
        
        self.zoom_info_label.setText(
            f"Zoom: {self.zoom_level:.1f}x\nPan: ({int(self.pan_offset_x)}, {int(self.pan_offset_y)})"
        )
        
        # Enable/disable pan shortcuts based on zoom level
        is_zoomed = self.zoom_level > 1.0
        self.pan_up_action.setEnabled(is_zoomed)
        self.pan_down_action.setEnabled(is_zoomed)
        self.pan_left_action.setEnabled(is_zoomed)
        self.pan_right_action.setEnabled(is_zoomed)
        
        self.update_display_frame()
    
    def pan_view(self, dx: int, dy: int) -> None:
        """Pan the view by dx, dy pixels (only when zoomed in)."""
        if self.zoom_level <= 1.0:
            return
        
        # Apply pan delta
        new_pan_x = self.pan_offset_x + dx
        new_pan_y = self.pan_offset_y + dy
        
        # Clamp to valid range
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
    
    def on_pan_drag(self, pan_x: float, pan_y: float) -> None:
        """Called when user drags to pan the view."""
        self.pan_offset_x = pan_x
        self.pan_offset_y = pan_y
        
        self.zoom_info_label.setText(
            f"Zoom: {self.zoom_level:.1f}x\nPan: ({int(self.pan_offset_x)}, {int(self.pan_offset_y)})"
        )
        self.update_display_frame()
    
    def resizeEvent(self, event) -> None:  # event: QResizeEvent
        super().resizeEvent(event)
        self.update_display_frame()
    
    def save_frames(self) -> None:
        if self.crop_coords is None:
            QMessageBox.warning(self, "No ROI", "Please select a Region of Interest (ROI) before exporting.")
            return
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Frames will be exported as RAW (without CLAHE).\nCLAHE will be applied during training/inference.")
        msg.setWindowTitle("Export Format")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if msg.exec_() == QMessageBox.Cancel:
            return
        
        if self.output_folder:
            output_folder = self.output_folder
        else:
            output_folder = QFileDialog.getExistingDirectory(
                self, "Select output folder for frames", os.getcwd(),
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
        
        if not output_folder:
            return
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        frame_count = self.end_frame - self.start_frame
        progress = QProgressDialog("Exporting frames...", "Cancel", 0, frame_count, self)
        progress.setWindowTitle("Exporting Frames for CNN")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        x, y, w, h = self.crop_coords
        exported_count = 0
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        for i, frame_num in enumerate(range(self.start_frame, self.end_frame)):
            if progress.wasCanceled():
                break
            success, frame = self.video.read()
            if not success:
                continue
            h_max, w_max = frame.shape[:2]
            crop_x = max(0, min(x, w_max - 1))
            crop_y = max(0, min(y, h_max - 1))
            crop_w = max(1, min(w, w_max - crop_x))
            crop_h = max(1, min(h, h_max - crop_y))
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            # Save RAW frames (no CLAHE) - CLAHE is applied at runtime during training/inference
            current_h, current_w = cropped.shape[:2]
            if current_w > CNN_WIDTH or current_h > CNN_HEIGHT:
                resized = cv2.resize(cropped, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(cropped, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_LINEAR)
            output_path = os.path.join(output_folder, f"{video_name}_{frame_num}.jpg")
            cv2.imwrite(output_path, resized)
            exported_count += 1
            progress.setValue(i + 1)
            progress.setLabelText(f"Exporting frame {i + 1} / {frame_count}")
            QApplication.processEvents()
        progress.close()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
        QMessageBox.information(self, "Export Complete", 
            f"Exported {exported_count} frames to:\n{output_folder}\n\nFrame size: {CNN_WIDTH}x{CNN_HEIGHT} pixels\nFormat: RAW JPEG (CLAHE applied during training)")
        
        self.export_completed = True
        self.close()
    
    def closeEvent(self, event) -> None:  # event: QCloseEvent
        if hasattr(self, 'export_completed') and self.export_completed:
            event.accept()
            return
        
        reply = QMessageBox.question(self, 'Confirm', 'Close without exporting?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    parser = argparse.ArgumentParser(description='CNN Video Preprocessing Tool')
    parser.add_argument('--video', '-v', type=str, help='Path to video file to open directly')
    parser.add_argument('--output', '-o', type=str, help='Path to output folder for exported frames')
    parser.add_argument('--roi', '-r', type=str, help='Default ROI in format x,y,w,h (e.g. "625,300,893,223")')
    args = parser.parse_args()
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    default_roi = None
    if args.roi:
        try:
            default_roi = tuple(map(int, args.roi.split(',')))
            if len(default_roi) != 4:
                raise ValueError("ROI must have 4 components")
        except Exception as e:
            print(f"Invalid ROI format: {e}. Using defaults.")
            default_roi = None

    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            QMessageBox.critical(None, "Error", f"Video file not found: {video_path}")
            sys.exit(1)
    else:
        video_filter = "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV);;All Files (*)"
        video_path, _ = QFileDialog.getOpenFileName(None, "Select video file for CNN preprocessing", os.getcwd(), video_filter)
        if not video_path:
            QMessageBox.information(None, "No file", "No video selected. Exiting...")
            sys.exit(0)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        QMessageBox.critical(None, "Error", "Cannot open video file.")
        sys.exit(1)
    window = CNNEditWindow(video, video_path, args.output, default_roi)
    window.show()
    app.exec_()
    video.release()

if __name__ == "__main__":
    main()