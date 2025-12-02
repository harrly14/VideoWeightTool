import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import cv2
import numpy as np
import time
import subprocess
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
    
    def __init__(self):
        super().__init__()
        self.crop_rect = None  # (x, y, w, h) in image coordinates
        self.original_w = None
        self.original_h = None
        self.aspect_ratio = CNN_ASPECT_RATIO
        self.setMouseTracking(True)
        
        self.action = None
        self.drag_start_pos = None
        self.initial_crop_rect = None
    
    def set_original_size(self, w, h):
        self.original_w = w
        self.original_h = h
        if self.crop_rect is None:
            # Default crop based on standard video location
            STANDARD_WIDTH = 893
            STANDARD_HEIGHT = 223
            STANDARD_X = 625
            STANDARD_Y = 300
            crop_w = STANDARD_WIDTH if w > STANDARD_WIDTH else int(w * 0.5)
            crop_h = STANDARD_HEIGHT if h > STANDARD_HEIGHT else int(crop_w / self.aspect_ratio)
            crop_x = STANDARD_X if w > STANDARD_X else (w - crop_w) // 2
            crop_y = STANDARD_Y if h > STANDARD_Y else (h - crop_h) // 2
            self.crop_rect = (crop_x, crop_y, crop_w, crop_h)
            self.crop_selected.emit(*self.crop_rect)
            self.update()
    
    def _get_display_params(self):
        if not self.original_w or not self.original_h:
            return None
        label_w = self.width()
        label_h = self.height()
        scale = min(label_w / self.original_w, label_h / self.original_h)
        display_w = int(self.original_w * scale)
        display_h = int(self.original_h * scale)
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2
        return scale, offset_x, offset_y

    def _image_to_label(self, img_rect):
        params = self._get_display_params()
        if not params: return None
        scale, off_x, off_y = params
        x, y, w, h = img_rect
        lx = int(x * scale) + off_x
        ly = int(y * scale) + off_y
        lw = int(w * scale)
        lh = int(h * scale)
        return (lx, ly, lw, lh)

    def _label_to_image_pt(self, pos):
        params = self._get_display_params()
        if not params: return None, None
        scale, off_x, off_y = params
        ix = int((pos.x() - off_x) / scale)
        iy = int((pos.y() - off_y) / scale)
        return ix, iy

    def mousePressEvent(self, event):
        if not self.crop_rect or event.button() != Qt.MouseButton.LeftButton:
            return
        
        l_rect = self._image_to_label(self.crop_rect)
        if not l_rect: return
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
            self.action = None
            return
            
        self.drag_start_pos = event.pos()
        self.initial_crop_rect = self.crop_rect

    def mouseMoveEvent(self, event):
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
                self.setCursor(Qt.CursorShape.ArrowCursor)
        
        if not self.action: return

        curr_ix, curr_iy = self._label_to_image_pt(event.pos())
        if curr_ix is None: return
        
        ix, iy, iw, ih = self.initial_crop_rect
        
        if self.action == 'move':
            # Calculate delta in image coords
            start_ix, start_iy = self._label_to_image_pt(self.drag_start_pos)
            dx = curr_ix - start_ix
            dy = curr_iy - start_iy
            
            nx = ix + dx
            ny = iy + dy
            
            nx = max(0, min(nx, self.original_w - iw))
            ny = max(0, min(ny, self.original_h - ih))
            self.crop_rect = (nx, ny, iw, ih)
            
        elif 'resize' in self.action:
            # Fixed corners:
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

    def mouseReleaseEvent(self, event):
        self.action = None
        if self.crop_rect:
            self.crop_selected.emit(*self.crop_rect)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.crop_rect:
            l_rect = self._image_to_label(self.crop_rect)
            if l_rect:
                lx, ly, lw, lh = l_rect
                painter = QPainter(self)
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
                painter.drawRect(lx, ly, lw, lh)
                
                # Rule of thirds grid (3x1 for horizontal thirds)
                painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
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

class CNNEditWindow(QWidget):
    def __init__(self, video_capture, video_path):
        super().__init__()
        self.video = video_capture
        self.video_path = video_path
        self.current_frame_index = 0
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.last_loaded_frame = -1
        self.current_raw_frame = None
        self.start_frame = 0
        self.end_frame = self.total_frames
        self.crop_coords = None
        self.output_folder = None
        self.read_timeout_warning_threshold = 5.0
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(100)
        self.pending_slider_action = None
        self.slider_timer.timeout.connect(self._execute_pending_slider_action)
        self.setWindowTitle("CNN Video Preprocessing Tool")
        self.resize(1000, 600)
        main_layout = self._setup_main_ui()
        success, first_frame = self.video.read()
        if success:
            self.original_h = first_frame.shape[0]
            self.original_w = first_frame.shape[1]
            self.video_label.set_original_size(self.original_w, self.original_h)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        info_label = QLabel(
            "CNN Input: 256x64 pixels (4:1 aspect ratio) | CLAHE auto-enhancement enabled | Crop will auto-enforce aspect ratio"
        )
        info_label.setStyleSheet(
            "background-color: #2a4a6a; color: white; padding: 8px; border-radius: 4px;"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        video_layout.addWidget(info_label)
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
        trim_layout = QHBoxLayout()
        self.trim_label = QLabel("Trim:")
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
        main_layout.addLayout(video_layout, 2)
        controls_layout = self._setup_controls()
        main_layout.addLayout(controls_layout, 1)
        return main_layout
    
    def _setup_controls(self):
        controls_layout = QVBoxLayout()
        crop_group = QGroupBox("ROI Selection (4:1 Aspect Ratio)")
        crop_layout = QVBoxLayout()
        self.crop_info_label = QLabel(
            "Drag box to move. Drag corners to resize.\nAspect ratio is locked."
        )
        self.crop_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop_info_label.setWordWrap(True)
        self.crop_dims_label = QLabel("Crop: Default")
        self.crop_dims_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_layout.addWidget(self.crop_info_label)
        crop_layout.addWidget(self.crop_dims_label)
        crop_group.setLayout(crop_layout)
        controls_layout.addWidget(crop_group)
        clahe_group = QGroupBox("Preprocessing (Automatic)")
        clahe_layout = QVBoxLayout()
        clahe_info = QLabel(
            "CLAHE Enhancement\nClip Limit: 2.0\nTile Grid: 8x8\nConverts to grayscale internally\nOutput: 3-channel BGR"
        )
        clahe_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clahe_layout.addWidget(clahe_info)
        clahe_group.setLayout(clahe_layout)
        controls_layout.addWidget(clahe_group)
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        output_info = QLabel(
            f"Final Size: {CNN_WIDTH}x{CNN_HEIGHT} pixels\nFormat: Individual JPEG frames\nInterpolation: INTER_AREA (downscale)\n                  INTER_LINEAR (upscale)"
        )
        output_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_layout.addWidget(output_info)
        output_group.setLayout(output_layout)
        controls_layout.addWidget(output_group)
        trim_group = QGroupBox("Trim Info")
        trim_layout = QVBoxLayout()
        self.trim_info_label = QLabel(
            f"Start: 0\nEnd: {self.total_frames - 1}\nFrames: {self.total_frames}"
        )
        self.trim_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trim_layout.addWidget(self.trim_info_label)
        trim_group.setLayout(trim_layout)
        controls_layout.addWidget(trim_group)
        self.save_button = QPushButton("Export Frames for CNN")
        self.save_button.setStyleSheet(
            "background-color: #4a7a4a; color: white; padding: 10px; font-weight: bold;"
        )
        self.save_button.clicked.connect(self.save_frames)
        controls_layout.addWidget(self.save_button)
        self.preview_button = QPushButton("Preview CLAHE Output")
        self.preview_button.setToolTip("Toggle CLAHE preview on current frame")
        self.preview_button.setCheckable(True)
        self.preview_button.setChecked(True)
        self.preview_button.setText("Preview: CLAHE On")
        self.preview_button.toggled.connect(self._update_preview_button_text)
        self.preview_button.clicked.connect(self.update_display_frame)
        controls_layout.addWidget(self.preview_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        controls_layout.addWidget(self.close_button)
        controls_layout.addStretch()
        return controls_layout
    
    def _update_preview_button_text(self, checked):
        if checked:
            self.preview_button.setText("Preview: CLAHE On")
        else:
            self.preview_button.setText("Preview: CLAHE Off")
    
    def on_crop_selected(self, x, y, w, h):
        self.crop_coords = (x, y, w, h)
        actual_ratio = w / h if h > 0 else 0
        self.crop_dims_label.setText(
            f"Crop: {w}x{h} pixels\nRatio: {actual_ratio:.2f}:1\nPosition: ({x}, {y})"
        )
        self.update_display_frame()
    
    def seek_to_frame(self, frame_num):
        self.current_frame_index = max(self.start_frame, min(frame_num, self.end_frame - 1))
        self.load_frame_from_video()
    
    def apply_trim(self, values):
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
            f"Start: {self.start_frame}\nEnd: {self.end_frame}\nFrames: {frame_count}\nDuration: {duration:.1f}s"
        )
        self.prev_button.setEnabled(self.current_frame_index > self.start_frame)
        self.next_button.setEnabled(self.current_frame_index < self.end_frame - 1)
        if self.current_frame_index < self.start_frame or self.current_frame_index >= self.end_frame:
            self.seek_to_frame(self.start_frame)
        elif self.last_loaded_frame != self.current_frame_index:
            self.load_frame_from_video()
    
    def load_frame_from_video(self):
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
    
    def update_display_frame(self):
        if self.current_raw_frame is None:
            self.video_label.setText("No frame loaded")
            return
        preview_frame = self.current_raw_frame.copy()
        if self.preview_button.isChecked():
            preview_frame = apply_clahe(preview_frame)
        self._update_image(preview_frame)
    
    def _update_image(self, frame):
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
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_frame()
    
    def save_frames(self):
        if self.crop_coords is None:
            QMessageBox.warning(self, "No ROI", "Please select a Region of Interest (ROI) before exporting.")
            return
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
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm', 'Close without exporting?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    video_filter = "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV);;All Files (*)"
    video_path, _ = QFileDialog.getOpenFileName(None, "Select video file for CNN preprocessing", os.getcwd(), video_filter)
    if not video_path:
        QMessageBox.information(None, "No file", "No video selected. Exiting...")
        sys.exit(0)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        QMessageBox.critical(None, "Error", "Cannot open video file.")
        sys.exit(1)
    window = CNNEditWindow(video, video_path)
    window.show()
    app.exec_()
    video.release()

if __name__ == "__main__":
    main()