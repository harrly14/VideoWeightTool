"""
Range Selection Dialog for selecting valid frame range using QRangeSlider.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QGroupBox, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from qtrangeslider import QRangeSlider


class RangeSelectionDialog(QDialog):
    """Dialog for selecting valid frame range before labelling."""
    
    def __init__(self, video_capture, video_filename: str, parent=None):
        """
        Initialize range selection dialog.
        
        Args:
            video_capture: cv2.VideoCapture object (already opened)
            video_filename: Name of video file for display
            parent: Parent widget
        """
        super().__init__(parent)
        self.video = video_capture
        self.video_filename = video_filename
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        
        self.current_frame_index = 0
        self.current_raw_frame = None
        self.last_loaded_frame = -1
        
        self.last_moved_handle = 'start'  # 'start' or 'end'
        
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(50)  # 50ms debounce
        self.slider_timer.timeout.connect(self._execute_pending_update)
        self.pending_update = None
        
        self.setWindowTitle(f"Select Valid Frame Range - {video_filename}")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        self.setup_ui()
        self.load_frame(0)
    
    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        self.frame_info_label = QLabel(f"Showing frame: 0 / {self.total_frames - 1}")
        self.frame_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_info_label)
        
        range_group = QGroupBox("Valid Frame Range")
        range_layout = QVBoxLayout()
        
        self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.range_slider.setRange(0, self.total_frames - 1)
        self.range_slider.setValue((0, self.total_frames - 1))
        self.range_slider.setStyleSheet("QRangeSlider {qproperty-barColor: #447;}")
        self.range_slider.valueChanged.connect(self._on_range_slider_changed)
        range_layout.addWidget(self.range_slider)
        
        spinbox_layout = QHBoxLayout()
        
        start_layout = QHBoxLayout()
        start_label = QLabel("Start frame:")
        self.start_spinbox = QSpinBox()
        self.start_spinbox.setRange(0, self.total_frames - 1)
        self.start_spinbox.setValue(0)
        self.start_spinbox.valueChanged.connect(self._on_start_spinbox_changed)
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_spinbox)
        spinbox_layout.addLayout(start_layout)
        
        spinbox_layout.addStretch()
        
        end_layout = QHBoxLayout()
        end_label = QLabel("End frame:")
        self.end_spinbox = QSpinBox()
        self.end_spinbox.setRange(0, self.total_frames - 1)
        self.end_spinbox.setValue(self.total_frames - 1)
        self.end_spinbox.valueChanged.connect(self._on_end_spinbox_changed)
        end_layout.addWidget(end_label)
        end_layout.addWidget(self.end_spinbox)
        spinbox_layout.addLayout(end_layout)
        
        range_layout.addLayout(spinbox_layout)
        
        self.range_info_label = QLabel(f"Selected range: 0 - {self.total_frames - 1} ({self.total_frames} frames)")
        self.range_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_layout.addWidget(self.range_info_label)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
        
        instructions = QLabel(
            "Drag the slider handles to set start and end frames. "
            "The video preview will update to show the frame under the handle you're moving. "
            "Use the spinboxes for precise frame selection."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(instructions)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.confirm_button = QPushButton("Confirm Range")
        self.confirm_button.clicked.connect(self._on_confirm)
        self.confirm_button.setDefault(True)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _on_range_slider_changed(self, values):
        """Handle range slider value change (debounced)."""
        new_start, new_end = values
        
        # Determine which handle moved by comparing to previous values
        if new_start != self.start_frame:
            self.last_moved_handle = 'start'
            self.start_frame = new_start
        if new_end != self.end_frame:
            self.last_moved_handle = 'end'
            self.end_frame = new_end
        
        if self.start_frame >= self.end_frame:
            if self.last_moved_handle == 'start':
                self.start_frame = self.end_frame - 1
            else:
                self.end_frame = self.start_frame + 1
            self.range_slider.blockSignals(True)
            self.range_slider.setValue((self.start_frame, self.end_frame))
            self.range_slider.blockSignals(False)
        
        self.start_spinbox.blockSignals(True)
        self.end_spinbox.blockSignals(True)
        self.start_spinbox.setValue(self.start_frame)
        self.end_spinbox.setValue(self.end_frame)
        self.start_spinbox.blockSignals(False)
        self.end_spinbox.blockSignals(False)
        
        frame_to_show = self.start_frame if self.last_moved_handle == 'start' else self.end_frame
        self.pending_update = frame_to_show
        self.slider_timer.start()
        
        self._update_range_info()
    
    def _on_start_spinbox_changed(self, value):
        """Handle start spinbox change."""
        if value >= self.end_frame:
            value = self.end_frame - 1
            self.start_spinbox.blockSignals(True)
            self.start_spinbox.setValue(value)
            self.start_spinbox.blockSignals(False)
        
        self.start_frame = value
        self.last_moved_handle = 'start'
        
        self.range_slider.blockSignals(True)
        self.range_slider.setValue((self.start_frame, self.end_frame))
        self.range_slider.blockSignals(False)
        
        self.load_frame(value)
        self._update_range_info()
    
    def _on_end_spinbox_changed(self, value):
        """Handle end spinbox change."""
        if value <= self.start_frame:
            value = self.start_frame + 1
            self.end_spinbox.blockSignals(True)
            self.end_spinbox.setValue(value)
            self.end_spinbox.blockSignals(False)
        
        self.end_frame = value
        self.last_moved_handle = 'end'
        
        self.range_slider.blockSignals(True)
        self.range_slider.setValue((self.start_frame, self.end_frame))
        self.range_slider.blockSignals(False)
        
        self.load_frame(value)
        self._update_range_info()
    
    def _execute_pending_update(self):
        """Execute pending frame update after debounce."""
        if self.pending_update is not None:
            self.load_frame(self.pending_update)
            self.pending_update = None
    
    def _update_range_info(self):
        """Update range info label."""
        frame_count = self.end_frame - self.start_frame + 1
        self.range_info_label.setText(
            f"Selected range: {self.start_frame} - {self.end_frame} ({frame_count} frames)"
        )
    
    def load_frame(self, frame_num: int):
        """Load and display a specific frame."""
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
            self.frame_info_label.setText(f"Showing frame: {frame_num} / {self.total_frames - 1}")
    
    def _update_display(self):
        """Update the video display."""
        if self.current_raw_frame is None:
            return
        
        rgb_image = cv2.cvtColor(self.current_raw_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """Handle resize to update display."""
        super().resizeEvent(event)
        self._update_display()
    
    def _on_confirm(self):
        """Confirm range selection."""
        if self.start_frame >= self.end_frame:
            QMessageBox.warning(
                self,
                "Invalid Range",
                "Start frame must be less than end frame."
            )
            return
        
        self.accept()
    
    def get_range(self) -> tuple[int, int]:
        """
        Get the selected frame range.
        
        Returns:
            tuple: (start_frame, end_frame)
        """
        return self.start_frame, self.end_frame
