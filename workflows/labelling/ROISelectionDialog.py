"""
ROI Section Dialog - Combined ROI and frame range selection for a single section.
Allows user to define both a quad ROI and its applicable frame range.
"""

import cv2
import json
import numpy as np
from typing import Optional, List, Tuple, Dict
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QMessageBox, QSlider, QSpinBox, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QEvent, QObject
from qtrangeslider import QRangeSlider

from core.ui.ROICropLabel import ROICropLabel, CNN_WIDTH, CNN_HEIGHT


class ROISelectionDialog(QDialog):
    """
    Dialog for defining a single ROI section (ROI + frame range).
    User can adjust the quad ROI and set start/end frames for this section.
    """
    
    def __init__(self, video_capture, video_filename: str, 
                 available_start: int, available_end: int,
                 default_roi: Optional[List[Tuple[int, int]]] = None,
                 section_number: int = 1,
                 existing_sections: Optional[List[Dict]] = None,
                 parent=None):
        """
        Initialize ROI section dialog.
        
        Args:
            video_capture: cv2.VideoCapture object (already opened)
            video_filename: Name of video file for display
            available_start: First available frame (not covered by other sections)
            available_end: Last available frame
            default_roi: Previous ROI to use as default (carryover)
            section_number: Which section number this is (for display)
            existing_sections: List of already-defined sections (for display)
            parent: Parent widget
        """
        super().__init__(parent)
        self.video = video_capture
        self.video_filename = video_filename
        self.available_start = available_start
        self.available_end = available_end
        self.default_roi = default_roi
        self.section_number = section_number
        self.existing_sections = existing_sections or []
        
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.section_start = available_start
        self.section_end = available_end
        
        self.preview_frame = (available_start + available_end) // 2
        self.roi_points = None
        
        self.current_frame_index = available_start
        self.current_raw_frame = None
        self.last_loaded_frame = -1
        self.last_moved_handle = 'preview'
        
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(15)
        self.slider_timer.timeout.connect(self._execute_pending_update)
        self.pending_update = None
        
        # Get video dimensions
        self.video.set(cv2.CAP_PROP_POS_FRAMES, available_start)
        success, frame = self.video.read()
        if success:
            self.video_height, self.video_width = frame.shape[:2]
            self.current_raw_frame = frame
            self.last_loaded_frame = available_start
        else:
            self.video_height, self.video_width = 720, 1280
        
        self.setWindowTitle(f"Define ROI Section {section_number} - {video_filename}")
        self.setMinimumSize(720, 405)
        self.resize(1513, 760)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
        
        self.setup_ui()
        # Install event filter on dialog and all children to intercept shortcuts globally
        self.installEventFilter(self)
        for child in self.findChildren(QObject):
            if hasattr(child, 'installEventFilter'):
                child.installEventFilter(self)
        
        self.video_label.setFocus()

        if self.default_roi:
            self.video_label.set_points(self.default_roi)
            self.roi_points = self.default_roi
        else:
            self.reset_roi()

        self.update_display_frame()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        header_label = QLabel(
            f"<b>Section {self.section_number}</b> — "
            f"Available frames: {self.available_start} - {self.available_end} "
            f"({self.available_end - self.available_start + 1} frames)"
        )
        header_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header_label)
        
        if self.existing_sections:
            existing_text = "Already defined: " + ", ".join([
                f"Section {i+1}: frames {s['start_frame']}-{s['end_frame']}"
                for i, s in enumerate(self.existing_sections)
            ])
            existing_label = QLabel(existing_text)
            existing_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
            layout.addWidget(existing_label)
        
        content_layout = QHBoxLayout()
        
        video_layout = QVBoxLayout()
        
        self.video_label = ROICropLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(500, 282)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.set_original_size(self.video_width, self.video_height)
        self.video_label.crop_selected.connect(self._on_roi_changed)
        self.video_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        video_layout.addWidget(self.video_label)
        
        range_group = QGroupBox()
        range_layout = QVBoxLayout()
        
        self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.range_slider.setRange(self.available_start, self.available_end)
        self.range_slider.setValue((self.section_start, self.preview_frame, self.section_end))
        self.range_slider.setStyleSheet("QRangeSlider {qproperty-barColor: #447;}")
        self.range_slider.valueChanged.connect(self._on_range_changed)
        range_layout.addWidget(self.range_slider)
        
        spinbox_layout = QHBoxLayout()
        
        start_layout = QHBoxLayout()
        start_label = QLabel("Start frame:")
        self.start_spinbox = QSpinBox()
        self.start_spinbox.setRange(self.available_start, self.available_end)
        self.start_spinbox.setValue(self.available_start)
        self.start_spinbox.valueChanged.connect(self._on_start_spinbox_changed)
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_spinbox)
        spinbox_layout.addLayout(start_layout)
        
        spinbox_layout.addStretch()

        current_frame_layout = QHBoxLayout()
        current_frame_label = QLabel("Current frame:")
        self.current_frame_display = QLabel(str(self.current_frame_index))
        self.current_frame_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_frame_display.setMinimumWidth(60) 
        current_frame_layout.addWidget(current_frame_label)
        current_frame_layout.addWidget(self.current_frame_display)
        spinbox_layout.addLayout(current_frame_layout)
        
        spinbox_layout.addStretch()
        
        end_layout = QHBoxLayout()
        end_label = QLabel("End frame:")
        self.end_spinbox = QSpinBox()
        self.end_spinbox.setRange(self.available_start, self.available_end)
        self.end_spinbox.setValue(self.available_end)
        self.end_spinbox.valueChanged.connect(self._on_end_spinbox_changed)
        end_layout.addWidget(end_label)
        end_layout.addWidget(self.end_spinbox)
        spinbox_layout.addLayout(end_layout)
        
        range_layout.addLayout(spinbox_layout)
        range_group.setMaximumHeight(150)
        range_group.setLayout(range_layout)
        self.range_group = range_group
        video_layout.addWidget(range_group)
        
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(6)
        controls_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)
        
        group_box_style = (
            "QGroupBox { border: 1px solid #555; border-radius: 3px; margin-top: 6px; padding-top: 6px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }"
        )
        
        shortcuts_group = QGroupBox("Shortcuts")
        shortcuts_group.setStyleSheet(group_box_style)
        shortcuts_layout = QVBoxLayout()
        shortcuts_layout.setSpacing(2)
        shortcuts_layout.setContentsMargins(6, 6, 6, 6)
        shortcuts_text = QLabel(
            "← → Frame nav\n"
            "Ctrl+ ← → Jump 10\n"
            "Wheel: Zoom\n"
            "Drag: Pan (when zoomed)\n"
            "↑↓ Shift+ ← → Pan (when zoomed)"
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

        self.preview_button = QPushButton("CLAHE: Off")
        self.preview_button.setToolTip("Toggle CLAHE preview")
        self.preview_button.setCheckable(True)
        self.preview_button.setChecked(False)
        self.preview_button.setMaximumHeight(28)
        self.preview_button.setStyleSheet("font-size: 10px; padding: 2px;")
        self.preview_button.toggled.connect(self._update_preview_button_text)
        self.preview_button.clicked.connect(self.update_display_frame)
        controls_layout.addWidget(self.preview_button)
        
        preview_group = QGroupBox("ROI Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(CNN_WIDTH, CNN_HEIGHT)
        self.preview_label.setStyleSheet("background-color: #333; border: 1px solid #666;")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)
        preview_group.setLayout(preview_layout)
        controls_layout.addWidget(preview_group)

        button_layout = QVBoxLayout()
        
        self.confirm_button = QPushButton("Confirm Section")
        self.confirm_button.setStyleSheet("background-color: #4a7a4a; color: white; padding: 8px 16px;")
        self.confirm_button.clicked.connect(self._on_confirm)
        self.confirm_button.setDefault(True)
        button_layout.addWidget(self.confirm_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("background-color: #7a4a4a; color: white; padding: 8px 16px;")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        controls_layout.addLayout(button_layout)     

        controls_layout.addStretch()
        content_layout.addLayout(video_layout, stretch=3)
        content_layout.addLayout(controls_layout, stretch=1)
        layout.addLayout(content_layout)
        self.setLayout(layout)

    def reset_roi(self):
        self.video_label.reset_crop_to_standard()

    def copy_roi_to_clipboard(self):
        if self.roi_points:
            roi_str = json.dumps(self.roi_points)
            cb = QApplication.clipboard()
            if cb:
                cb.setText(roi_str)
            original_text = self.copy_roi_button.text()
            self.copy_roi_button.setText("Copied!")
            QTimer.singleShot(1000, lambda: self.copy_roi_button.setText(original_text))
        else:
            QMessageBox.warning(self, "No ROI", "Please define a quad ROI before copying to clipboard.")
    
    def _update_preview_button_text(self, checked):
        if checked:
            self.preview_button.setText("CLAHE: On")
        else:
            self.preview_button.setText("CLAHE: Off")

    def apply_clahe(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        bgr_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return bgr_enhanced

    def update_display_frame(self):
        if self.current_raw_frame is None:
            self.video_label.setText("No frame loaded")
            return
        self._update_display()
    
    def _on_range_changed(self, values):
        new_start, new_preview, new_end = values

        # Determine which handle moved by comparing to previous values. 
        # The preview handle could change as a result of the other handles changing, 
        # but the one that moved most is likely the one being dragged
        start_delta = abs(new_start - self.section_start)
        preview_delta = abs(new_preview - self.preview_frame)
        end_delta = abs(new_end - self.section_end)
        
        if start_delta >= preview_delta and start_delta >= end_delta and start_delta > 0:
            self.last_moved_handle = 'start'
        elif end_delta >= preview_delta and end_delta >= start_delta and end_delta > 0:
            self.last_moved_handle = 'end'
        elif preview_delta > 0:
            self.last_moved_handle = 'preview'
        
        self.section_start = new_start
        self.preview_frame = new_preview
        self.section_end = new_end

        # Enforce ordering: start < preview < end (with minimum separation of 2 for preview to fit)
        needs_correction = False
        
        if self.section_start >= self.section_end - 1:
            if self.last_moved_handle == 'start':
                self.section_start = self.section_end - 2
            else:
                self.section_end = self.section_start + 2
            needs_correction = True

        # If start/end handles are dragged close to preview, relocate preview to midpoint
        # to prevent crowding and overlapping handles
        if self.last_moved_handle == 'start':
            if self.preview_frame - self.section_start < 5: 
                self.preview_frame = (self.section_start + self.section_end) // 2
                needs_correction = True
        elif self.last_moved_handle == 'end':
            if self.section_end - self.preview_frame < 5:
                self.preview_frame = (self.section_start + self.section_end) // 2
                needs_correction = True
        
        if self.preview_frame <= self.section_start:
            self.preview_frame = self.section_start + 1
            needs_correction = True
        if self.preview_frame >= self.section_end:
            self.preview_frame = self.section_end - 1
            needs_correction = True
        
        # Final sanity check: ensure start < preview < end
        if not (self.section_start < self.preview_frame < self.section_end):
            self.preview_frame = (self.section_start + self.section_end) // 2
            needs_correction = True
        
        if needs_correction:
            self.range_slider.blockSignals(True)
            self.range_slider.setValue((self.section_start, self.preview_frame, self.section_end))
            self.range_slider.blockSignals(False)

        self.start_spinbox.blockSignals(True)
        self.end_spinbox.blockSignals(True)
        self.start_spinbox.setValue(self.section_start)
        self.end_spinbox.setValue(self.section_end)
        self.start_spinbox.blockSignals(False)
        self.end_spinbox.blockSignals(False)

        if self.last_moved_handle == 'start':
            self.pending_update = self.section_start
        elif self.last_moved_handle == 'end':
            self.pending_update = self.section_end
        else:
            self.pending_update = self.preview_frame
        self.slider_timer.start()
    
    def _execute_pending_update(self):
        if self.pending_update is not None:
            if self.last_moved_handle == 'preview':
                self.load_frame(self.pending_update)
            else:
                self._show_frame(self.pending_update)
            self.pending_update = None
    
    def _on_start_spinbox_changed(self, value):
        if value >= self.section_end - 1:
            value = self.section_end - 2
            self.start_spinbox.blockSignals(True)
            self.start_spinbox.setValue(value)
            self.start_spinbox.blockSignals(False)

        self.section_start = value
        self.last_moved_handle = 'start'
        
        if self.preview_frame <= self.section_start:
            self.preview_frame = self.section_start + 1
        
        self.range_slider.blockSignals(True)
        self.range_slider.setValue((self.section_start, self.preview_frame, self.section_end))
        self.range_slider.blockSignals(False)

        self._show_frame(value)
    
    def _on_end_spinbox_changed(self, value):
        if value <= self.section_start + 1:
            value = self.section_start + 2
            self.end_spinbox.blockSignals(True)
            self.end_spinbox.setValue(value)
            self.end_spinbox.blockSignals(False)

        self.section_end = value
        self.last_moved_handle = 'end'
        
        if self.preview_frame >= self.section_end:
            self.preview_frame = self.section_end - 1
        
        self.range_slider.blockSignals(True)
        self.range_slider.setValue((self.section_start, self.preview_frame, self.section_end))
        self.range_slider.blockSignals(False)

        self._show_frame(value)
    
    def _use_all_available(self):
        """Set range to all available frames."""
        self.section_start = self.available_start
        self.section_end = self.available_end
        self.preview_frame = (self.available_start + self.available_end) // 2
        
        self.range_slider.blockSignals(True)
        self.start_spinbox.blockSignals(True)
        self.end_spinbox.blockSignals(True)
        
        self.range_slider.setValue((self.section_start, self.preview_frame, self.section_end))
        self.start_spinbox.setValue(self.section_start)
        self.end_spinbox.setValue(self.section_end)
        
        self.range_slider.blockSignals(False)
        self.start_spinbox.blockSignals(False)
        self.end_spinbox.blockSignals(False)
        
        self.load_frame(self.preview_frame)
    
    def _on_roi_changed(self, points):
        if points and len(points) == 4:
            self.roi_points = [[int(p[0]), int(p[1])] for p in points]
            self._update_preview()
    
    def _on_reset_roi(self):
        self.video_label.reset_crop_to_standard()
    
    def _show_frame(self, frame_num: int):
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        
        if self.last_loaded_frame != frame_num:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = self.video.read()
            
            if success:
                self.current_raw_frame = frame
                self.last_loaded_frame = frame_num
                self.current_frame_index = frame_num
            else:
                return
        
        if self.current_raw_frame is not None:
            self._update_display()
            self.current_frame_display.setText(str(frame_num))
    
    def load_frame(self, frame_num: int):
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        
        if self.last_loaded_frame != frame_num:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, frame = self.video.read()
            
            if success:
                self.current_raw_frame = frame
                self.last_loaded_frame = frame_num
                self.current_frame_index = frame_num
            else:
                return
        
        if self.current_raw_frame is not None:
            self._update_display()
            self.preview_frame = frame_num
            self.range_slider.blockSignals(True)
            self.range_slider.setValue((self.section_start, self.preview_frame, self.section_end))
            self.range_slider.blockSignals(False)


            self.current_frame_display.setText(str(frame_num))
    
    def _update_display(self, frame=None):
        if frame is None:
            frame = self.current_raw_frame
        if frame is None:
            return

        if getattr(self, 'preview_button', None) and self.preview_button.isChecked():
            frame = self.apply_clahe(frame)
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.setFullPixmap(pixmap)
        self._update_preview()
    
    def _update_preview(self):
        if self.current_raw_frame is None or not self.roi_points:
            return
        
        try:
            frame_to_warp = self.current_raw_frame
            if getattr(self, 'preview_button', None) and self.preview_button.isChecked():
                frame_to_warp = self.apply_clahe(frame_to_warp)

            pts = np.array(self.roi_points, dtype=np.float32)
            dst_pts = np.array([
                [0, 0], 
                [CNN_WIDTH - 1, 0], 
                [CNN_WIDTH - 1, CNN_HEIGHT - 1], 
                [0, CNN_HEIGHT - 1]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts, dst_pts)
            warped = cv2.warpPerspective(frame_to_warp, M, (CNN_WIDTH, CNN_HEIGHT))
            
            rgb_preview = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_preview.shape
            qimage = QImage(rgb_preview.data, w, h, c * w, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(qimage))
        except Exception as e:
            print(f"Preview error: {e}")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
    
    def wheelEvent(self, event):
        if self.video_label.underMouse():
            self.video_label.wheelEvent(event)
        else:
            super().wheelEvent(event)
    
    def eventFilter(self, obj, event):
        """Intercept arrow key events before child widgets consume them."""
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                self.keyPressEvent(event)
                if event.isAccepted():
                    return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        try:
            under_mouse = self.video_label.underMouse()
            zoomed = getattr(self.video_label, 'zoom_level', 1.0) > 1.0
        except Exception:
            under_mouse = False
            zoomed = False

        if under_mouse and zoomed:
            step_frac = 0.05
            try:
                visible_w = self.video_label.original_w / self.video_label.zoom_level
                visible_h = self.video_label.original_h / self.video_label.zoom_level
                dx = int(visible_w * step_frac)
                dy = int(visible_h * step_frac)
            except Exception:
                dx = 20
                dy = 20

            if key == Qt.Key.Key_Left and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                new_pan_x = getattr(self.video_label, 'pan_offset_x', 0) - dx
                self.video_label.set_zoom(self.video_label.zoom_level, pan_x=new_pan_x)
                event.accept()
                return
            if key == Qt.Key.Key_Right and (modifiers & Qt.KeyboardModifier.ShiftModifier):
                new_pan_x = getattr(self.video_label, 'pan_offset_x', 0) + dx
                self.video_label.set_zoom(self.video_label.zoom_level, pan_x=new_pan_x)
                event.accept()
                return
            if key == Qt.Key.Key_Up:
                new_pan_y = getattr(self.video_label, 'pan_offset_y', 0) - dy
                self.video_label.set_zoom(self.video_label.zoom_level, pan_y=new_pan_y)
                event.accept()
                return
            if key == Qt.Key.Key_Down:
                new_pan_y = getattr(self.video_label, 'pan_offset_y', 0) + dy
                self.video_label.set_zoom(self.video_label.zoom_level, pan_y=new_pan_y)
                event.accept()
                return

        # Arrow keys for frame navigation
        if key == Qt.Key.Key_Left:
            step = 10 if modifiers & Qt.KeyboardModifier.ControlModifier else 1
            new_frame = max(self.section_start + 1, self.preview_frame - step)
            self.load_frame(new_frame)
            event.accept()
            return
        elif key == Qt.Key.Key_Right:
            step = 10 if modifiers & Qt.KeyboardModifier.ControlModifier else 1
            new_frame = min(self.section_end - 1, self.preview_frame + step)
            self.load_frame(new_frame)
            event.accept()
            return

        super().keyPressEvent(event)
    
    def _on_confirm(self):
        if not self.roi_points or len(self.roi_points) != 4:
            QMessageBox.warning(
                self,
                "No ROI Selected",
                "Please select a region of interest before confirming."
            )
            return
        
        if self.section_start >= self.section_end:
            QMessageBox.warning(
                self,
                "Invalid Range",
                "Start frame must be less than end frame."
            )
            return
        
        self.accept()
    
    def get_section(self) -> Optional[Dict]:
        if self.roi_points and len(self.roi_points) == 4:
            return {
                'quad': self.roi_points,
                'start_frame': self.section_start,
                'end_frame': self.section_end
            }
        return None
