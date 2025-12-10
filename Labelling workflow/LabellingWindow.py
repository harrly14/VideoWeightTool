"""
Labelling Window for labelling frames within a pre-selected range.
Includes auto-stop after n labels and review/exit/next dialog.
"""

import csv
import cv2
import os
import sys
import numpy as np
import time
import re
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout,
    QSlider, QGroupBox, QMessageBox, QAction, QSpinBox, QDialog,
    QDialogButtonBox
)
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator
from PyQt5.QtCore import Qt, QRegExp, QEvent, QTimer
from functools import partial
from VideoParams import VideoParams
from ZoomPanLabel import ZoomPanLabel
from FrameSampler import FrameSampler


class VideoCompleteDialog(QDialog):
    """Dialog shown when all labels are complete for a video."""
    
    EXIT_BATCH = 0
    REVIEW_LABELS = 1
    NEXT_VIDEO = 2
    
    def __init__(self, filename: str, labelled_count: int, total_target: int, 
                 video_index: int, total_videos: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Complete")
        self.result_action = self.NEXT_VIDEO
        
        layout = QVBoxLayout()
        
        msg = QLabel(
            f"<b>Video {video_index}/{total_videos}: {filename}</b><br><br>"
            f"Labelled {labelled_count} / {total_target} target frames.<br><br>"
            "What would you like to do?"
        )
        msg.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(msg)
        
        button_layout = QHBoxLayout()
        
        exit_btn = QPushButton("Exit Batch")
        exit_btn.clicked.connect(lambda: self._set_result(self.EXIT_BATCH))
        button_layout.addWidget(exit_btn)
        
        review_btn = QPushButton("Review Labels")
        review_btn.clicked.connect(lambda: self._set_result(self.REVIEW_LABELS))
        button_layout.addWidget(review_btn)
        
        next_btn = QPushButton("Next Video")
        next_btn.setDefault(True)
        next_btn.clicked.connect(lambda: self._set_result(self.NEXT_VIDEO))
        button_layout.addWidget(next_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def _set_result(self, action: int):
        self.result_action = action
        self.accept()
    
    def get_action(self) -> int:
        return self.result_action


class LabellingWindow(QWidget):
    """Window for labelling frames within a pre-selected valid range."""
    
    # Weight validation: 0-20 with up to 3 decimal places
    WEIGHT_PATTERN = r'^(20(\.0{0,3})?|1?\d(\.\d{0,3})?|\.\d{0,3})$'
    
    def __init__(self, video_capture, video_filename: str, start_frame: int, 
                 end_frame: int, target_labels: int, video_index: int = 1, 
                 total_videos: int = 1):
        """
        Initialize labelling window.
        
        Args:
            video_capture: cv2.VideoCapture object
            video_filename: Name of the video file
            start_frame: Valid range start (from RangeSelectionDialog)
            end_frame: Valid range end (from RangeSelectionDialog)
            target_labels: Target number of frames to label
            video_index: Current video index (for display)
            total_videos: Total videos in batch (for display)
        """
        super().__init__()
        
        self.video_params = VideoParams()
        self.video = video_capture
        self.video_filename = video_filename
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_labels = target_labels
        self.video_index = video_index
        self.total_videos = total_videos
        
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_index = start_frame
        self.last_loaded_frame = -1
        self.current_raw_frame = None
        
        sampler = FrameSampler(self.total_frames, start_frame, end_frame, target_labels)
        self.target_frames, self.interval, self.warning = sampler.get_target_frames()
        self.target_frame_index = 0
        
        self.frame_data = {}  # {frame_num: weight_string}
        self.labels_entered = 0
        
        # Result action (for batch manager)
        self.result_action = VideoCompleteDialog.NEXT_VIDEO
        self.user_cancelled = False
        
        self.read_timeout_warning_threshold = 5.0
        
        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(100)
        self.pending_slider_action = None
        self.slider_timer.timeout.connect(self._execute_pending_slider_action)
        
        self.autosave_timer = QTimer()
        self.autosave_timer.setInterval(60000)  # 1 minute
        self.autosave_timer.timeout.connect(self.auto_save_backup)
        self.autosave_timer.start()
        
        self.setWindowTitle(f"Labelling: {video_filename} ({video_index}/{total_videos})")
        self.resize(1000, 700)
        
        main_layout = self.setup_main_ui()
        
        success, first_frame = self.video.read()
        if success:
            self.original_h = first_frame.shape[0]
            self.original_w = first_frame.shape[1]
            self.video_label.set_original_size(self.original_w, self.original_h)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        self.setLayout(main_layout)
        
        if self.warning:
            QMessageBox.warning(
                self,
                "Fewer Labels Available",
                f"Valid range has fewer frames than target labels.\n"
                f"Will label {len(self.target_frames)} frames instead of {target_labels}."
            )
        
        self.current_frame_index = self.target_frames[0]
        self.load_frame_from_video()
        self.update_progress_display()
    
    def _debounced_slider(self, slider, value_changed_func):
        """Debounce slider events."""
        self.slider_timer.stop()
        self.pending_slider_action = lambda: value_changed_func(slider.value())
        self.slider_timer.start()
    
    def _execute_pending_slider_action(self):
        """Execute pending slider action."""
        if self.pending_slider_action:
            self.pending_slider_action()
            self.pending_slider_action = None
    
    def auto_save_backup(self):
        """Auto-save backup of frame data."""
        backup_file = f".backup_{self.video_filename}.csv"
        try:
            with open(backup_file, 'w', newline='') as file:
                writer = csv.writer(file)
                for frame_num, weight in sorted(self.frame_data.items()):
                    writer.writerow([frame_num, weight])
        except Exception as e:
            print(f"Auto-save failed: {e}")
    
    def setup_main_ui(self) -> QVBoxLayout:
        """Setup main UI layout."""
        main_layout = QVBoxLayout()
        
        header_layout = QHBoxLayout()
        header_label = QLabel(
            f"<b>Video:</b> {self.video_filename} | "
            f"<b>Range:</b> {self.start_frame} - {self.end_frame} | "
            f"<b>Target frames:</b> {len(self.target_frames)}"
        )
        header_label.setTextFormat(Qt.TextFormat.RichText)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        self.video_label = ZoomPanLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label, stretch=2)
        
        labelling_group = QGroupBox("Labelling")
        labelling_layout = QVBoxLayout()
        
        weight_entry_layout = QHBoxLayout()
        weight_label = QLabel("Weight (0-20):")
        self.weight_entry = QLineEdit()
        self.weight_entry.setPlaceholderText("Enter weight (0-20, up to 3 decimals)")
        self.weight_entry.setToolTip("Enter weight for current frame (Enter/Space to submit)")
        
        pattern = QRegExp(r'^[0-9.]*$')
        validator = QRegExpValidator(pattern)
        self.weight_entry.setValidator(validator)
        self.weight_entry.installEventFilter(self)
        self.weight_entry.returnPressed.connect(self.write_entry)
        
        self.weight_button = QPushButton("Submit")
        self.weight_button.clicked.connect(self.write_entry)
        
        weight_entry_layout.addWidget(weight_label)
        weight_entry_layout.addWidget(self.weight_entry, stretch=1)
        weight_entry_layout.addWidget(self.weight_button)
        labelling_layout.addLayout(weight_entry_layout)
        
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← Previous Target")
        self.prev_button.clicked.connect(self.previous_target_frame)
        self.prev_button.setToolTip("Go to previous target frame (Left arrow)")
        
        self.prev_action = QAction("Previous target frame", self)
        self.prev_action.setShortcut("Left")
        self.prev_action.triggered.connect(self.previous_target_frame)
        self.addAction(self.prev_action)
        
        self.next_button = QPushButton("Next Target →")
        self.next_button.clicked.connect(self.next_target_frame)
        self.next_button.setToolTip("Go to next target frame (Right arrow)")
        
        self.next_action = QAction("Next target frame", self)
        self.next_action.setShortcut("Right")
        self.next_action.triggered.connect(self.next_target_frame)
        self.addAction(self.next_action)
        
        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)
        labelling_layout.addLayout(nav_layout)
        
        scrub_layout = QHBoxLayout()
        scrub_label = QLabel("Frame:")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(self.start_frame, self.end_frame)
        self.frame_slider.setValue(self.start_frame)
        self.frame_slider.setToolTip("Drag to navigate to any frame in valid range")
        self.frame_slider.valueChanged.connect(
            partial(self._debounced_slider, self.frame_slider, self.seek_to_frame)
        )
        
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setRange(self.start_frame, self.end_frame)
        self.frame_spinbox.setValue(self.start_frame)
        self.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        
        self.frame_display = QLabel(f"{self.start_frame} / {self.end_frame}")
        
        scrub_layout.addWidget(scrub_label)
        scrub_layout.addWidget(self.frame_slider, stretch=1)
        scrub_layout.addWidget(self.frame_spinbox)
        scrub_layout.addWidget(self.frame_display)
        labelling_layout.addLayout(scrub_layout)
        
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progress:")
        self.progress_display = QLabel(f"0 / {len(self.target_frames)} labels entered")
        self.target_display = QLabel(f"Target frame: 1 / {len(self.target_frames)}")
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_display)
        progress_layout.addStretch()
        progress_layout.addWidget(self.target_display)
        labelling_layout.addLayout(progress_layout)
        
        labelling_group.setLayout(labelling_layout)
        main_layout.addWidget(labelling_group)
        
        controls_layout = self.setup_display_controls()
        main_layout.addLayout(controls_layout)
        
        button_layout = QHBoxLayout()
        
        exit_batch_button = QPushButton("Exit Batch")
        exit_batch_button.clicked.connect(self._exit_batch)
        button_layout.addWidget(exit_batch_button)
        
        button_layout.addStretch()
        
        save_close_button = QPushButton("Save and Continue")
        save_close_button.clicked.connect(self.save_and_close)
        button_layout.addWidget(save_close_button)
        
        main_layout.addLayout(button_layout)
        
        self.focus_action = QAction("Focus weight entry", self)
        self.focus_action.setShortcut("W")
        self.focus_action.triggered.connect(lambda: self.weight_entry.setFocus())
        self.addAction(self.focus_action)
        
        return main_layout
    
    def setup_display_controls(self) -> QHBoxLayout:
        """Setup display adjustment sliders (brightness, saturation, contrast)."""
        controls_layout = QHBoxLayout()
        
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout()
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-255, 255)
        self.brightness_slider.setValue(self.video_params.brightness)
        self.brightness_slider.setToolTip("Adjust brightness (-255 to 255)")
        self.brightness_slider.valueChanged.connect(
            partial(self._on_param_change, "brightness")
        )
        self.brightness_display = QLabel("0")
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(self.brightness_slider)
        brightness_row.addWidget(self.brightness_display)
        brightness_layout.addLayout(brightness_row)
        reset_brightness = QPushButton("Reset")
        reset_brightness.clicked.connect(partial(self.reset_slider, "brightness"))
        brightness_layout.addWidget(reset_brightness)
        brightness_group.setLayout(brightness_layout)
        controls_layout.addWidget(brightness_group)
        
        saturation_group = QGroupBox("Saturation")
        saturation_layout = QVBoxLayout()
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(0, 300)
        self.saturation_slider.setValue(self.video_params.saturation)
        self.saturation_slider.setToolTip("Adjust saturation (0-300, 100 is normal)")
        self.saturation_slider.valueChanged.connect(
            partial(self._on_param_change, "saturation")
        )
        self.saturation_display = QLabel("100")
        saturation_row = QHBoxLayout()
        saturation_row.addWidget(self.saturation_slider)
        saturation_row.addWidget(self.saturation_display)
        saturation_layout.addLayout(saturation_row)
        reset_saturation = QPushButton("Reset")
        reset_saturation.clicked.connect(partial(self.reset_slider, "saturation"))
        saturation_layout.addWidget(reset_saturation)
        saturation_group.setLayout(saturation_layout)
        controls_layout.addWidget(saturation_group)
        
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout()
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(self.video_params.contrast)
        self.contrast_slider.setToolTip("Adjust contrast (0-200, 100 is normal)")
        self.contrast_slider.valueChanged.connect(
            partial(self._on_param_change, "contrast")
        )
        self.contrast_display = QLabel("100")
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self.contrast_slider)
        contrast_row.addWidget(self.contrast_display)
        contrast_layout.addLayout(contrast_row)
        reset_contrast = QPushButton("Reset")
        reset_contrast.clicked.connect(partial(self.reset_slider, "contrast"))
        contrast_layout.addWidget(reset_contrast)
        contrast_group.setLayout(contrast_layout)
        controls_layout.addWidget(contrast_group)
        
        return controls_layout
    
    def eventFilter(self, obj, event):
        """Handle space key in weight entry."""
        if obj is self.weight_entry and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Space:
                self.write_entry()
                return True
        return super().eventFilter(obj, event)
    
    def _on_frame_spinbox_changed(self, value):
        """Handle frame spinbox change."""
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        self.seek_to_frame(value)
    
    def _on_param_change(self, param_name: str, value):
        """Handle display parameter slider change."""
        old_val = getattr(self.video_params, param_name)
        setattr(self.video_params, param_name, value)
        
        if param_name == "brightness":
            self.brightness_display.setText(str(value))
        elif param_name == "saturation":
            self.saturation_display.setText(str(value))
        elif param_name == "contrast":
            self.contrast_display.setText(str(value))
        
        try:
            self.video_params.validate()
            self.update_display_frame()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid parameter", str(e))
            setattr(self.video_params, param_name, old_val)
    
    def reset_slider(self, param_name: str):
        """Reset display slider to default."""
        default_val = self.video_params.get_default_value(param_name)
        if default_val is None:
            return
        
        if param_name == "brightness":
            self.brightness_slider.setValue(default_val)
            self.brightness_display.setText(str(default_val))
        elif param_name == "saturation":
            self.saturation_slider.setValue(default_val)
            self.saturation_display.setText(str(default_val))
        elif param_name == "contrast":
            self.contrast_slider.setValue(default_val)
            self.contrast_display.setText(str(default_val))
        
        self.video_params.validate()
        self.update_display_frame()
    
    def seek_to_frame(self, frame_num: int):
        """Seek to specific frame."""
        frame_num = max(self.start_frame, min(frame_num, self.end_frame))
        self.current_frame_index = frame_num
        
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame_num)
        self.frame_spinbox.blockSignals(False)
        
        if frame_num in self.target_frames:
            self.target_frame_index = self.target_frames.index(frame_num)
            self.update_progress_display()
        
        self.load_frame_from_video()
    
    def load_frame_from_video(self):
        """Load frame from video file."""
        if self.last_loaded_frame != self.current_frame_index:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            
            time0 = time.time()
            success, frame = self.video.read()
            duration = time.time() - time0
            
            if duration >= self.read_timeout_warning_threshold or not success:
                QMessageBox.critical(
                    None, "Error",
                    "Frame took too long to read. If using a network file, consider making a local copy."
                )
                sys.exit(-1)
            
            self.current_raw_frame = frame
            self.last_loaded_frame = self.current_frame_index
        
        if self.current_raw_frame is not None:
            if self.current_frame_index in self.target_frames:
                value = self.frame_data.get(self.current_frame_index, "")
                self.weight_entry.setText(value)
            else:
                self.weight_entry.setText("")
            
            self.prev_button.setEnabled(self.target_frame_index > 0)
            self.next_button.setEnabled(self.target_frame_index < len(self.target_frames) - 1)
            
            self.update_display_frame()
            
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
            self.frame_display.setText(f"{self.current_frame_index} / {self.end_frame}")
    
    def update_display_frame(self):
        """Update displayed frame with effects applied."""
        if self.current_raw_frame is None:
            self.video_label.setText("No frame loaded")
            return
        
        preview_frame = self.apply_preview_effects(self.current_raw_frame)
        self.update_image(preview_frame)
    
    def apply_preview_effects(self, frame):
        """Apply brightness, saturation, and contrast adjustments."""
        if (self.video_params.contrast == 100
                and self.video_params.brightness == 0
                and self.video_params.saturation == 100):
            return frame
        
        processed = frame.astype(np.float32)
        
        # Brightness and contrast (in YCrCb)
        if self.video_params.contrast != 100 or self.video_params.brightness != 0:
            ycrcb = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_BGR2YCrCb).astype(np.float32)
            contrast_factor = self.video_params.contrast / 100.0
            ycrcb[:, :, 0] = (ycrcb[:, :, 0] - 128.0) * contrast_factor + 128.0 + self.video_params.brightness
            processed = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        else:
            processed = processed.astype(np.uint8)
        
        # Saturation (in HSV)
        if self.video_params.saturation != 100:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (self.video_params.saturation / 100.0), 0, 255)
            processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return processed
    
    def update_image(self, frame):
        """Update displayed frame image."""
        qimage = self.cv2_to_qimage(frame)
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.setFullPixmap(pixmap)
    
    def cv2_to_qimage(self, cv_img):
        """Convert OpenCV image to QImage."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        return QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        self.update_display_frame()
    
    def validate_weight(self, value: str) -> tuple[bool, str]:
        """
        Validate weight value (0-20, up to 3 decimal places).
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not value:
            return False, "Weight cannot be empty"
        
        try:
            num = float(value)
            if num < 0 or num > 20:
                return False, "Weight must be between 0 and 20"
            
            if '.' in value:
                decimal_part = value.split('.')[1]
                if len(decimal_part) > 3:
                    return False, "Maximum 3 decimal places allowed"
            
            return True, ""
        except ValueError:
            return False, "Invalid number format"

    def _save_current_weight_if_valid(self, show_warning: bool = False) -> bool:
        """Save current weight entry if valid; return True if saved."""
        current_value = self.weight_entry.text().strip()
        if not current_value:
            return False

        if self.current_frame_index not in self.target_frames:
            if show_warning:
                QMessageBox.warning(
                    self, "Not a Target Frame",
                    "You can only label target frames. Navigate to a target frame using Previous/Next buttons."
                )
            return False

        is_valid, error_msg = self.validate_weight(current_value)
        if not is_valid:
            if show_warning:
                QMessageBox.warning(self, "Invalid Weight", error_msg)
            return False

        is_new_label = self.current_frame_index not in self.frame_data
        self.frame_data[self.current_frame_index] = current_value
        if is_new_label:
            self.labels_entered += 1
            self.update_progress_display()

        return True
    
    def write_entry(self):
        """Save weight entry for current frame."""
        value = self.weight_entry.text().strip()
        if not value:
            return

        saved = self._save_current_weight_if_valid(show_warning=True)
        if saved:
            if self.labels_entered >= len(self.target_frames):
                self._show_complete_dialog()
                return

            if self.target_frame_index < len(self.target_frames) - 1:
                self.next_target_frame()
    
    def update_progress_display(self):
        """Update progress display."""
        self.progress_display.setText(
            f"{self.labels_entered} / {len(self.target_frames)} labels entered"
        )
        self.target_display.setText(
            f"Target frame: {self.target_frame_index + 1} / {len(self.target_frames)}"
        )
    
    def next_target_frame(self):
        """Navigate to next target frame."""
        if self.target_frame_index >= len(self.target_frames) - 1:
            return
        self._save_current_weight_if_valid(show_warning=False)

        self.target_frame_index += 1
        self.current_frame_index = self.target_frames[self.target_frame_index]
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(self.current_frame_index)
        self.frame_spinbox.blockSignals(False)
        self.load_frame_from_video()
    
    def previous_target_frame(self):
        """Navigate to previous target frame."""
        if self.target_frame_index <= 0:
            return
        self._save_current_weight_if_valid(show_warning=False)

        self.target_frame_index -= 1
        self.current_frame_index = self.target_frames[self.target_frame_index]
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(self.current_frame_index)
        self.frame_spinbox.blockSignals(False)
        self.load_frame_from_video()
    
    def _show_complete_dialog(self):
        """Show video complete dialog with options."""
        dialog = VideoCompleteDialog(
            self.video_filename,
            self.labels_entered,
            len(self.target_frames),
            self.video_index,
            self.total_videos,
            self
        )
        dialog.exec_()
        
        self.result_action = dialog.get_action()
        
        if self.result_action == VideoCompleteDialog.EXIT_BATCH:
            self.user_cancelled = False  # Not cancelled, just exiting
            self.close()
        elif self.result_action == VideoCompleteDialog.REVIEW_LABELS:
            # Stay open for review - user can edit labels
            pass
        else:  # NEXT_VIDEO
            self.close()
    
    def _exit_batch(self):
        """Exit entire batch."""
        reply = QMessageBox.question(
            self, "Exit Batch",
            "Are you sure you want to exit the batch?\n\n"
            "Current video labels will be saved.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.result_action = VideoCompleteDialog.EXIT_BATCH
            self.close()
    
    def save_and_close(self):
        """Save and move to next video."""
        self._save_current_weight_if_valid(show_warning=False)
        self._show_complete_dialog()
    
    def closeEvent(self, event):
        """Handle window close."""
        backup_file = f".backup_{self.video_filename}.csv"
        if os.path.exists(backup_file):
            try:
                os.remove(backup_file)
            except:
                pass
        
        self.autosave_timer.stop()
        self.slider_timer.stop()
        
        event.accept()
    
    def get_frame_data(self) -> dict:
        """Get labelled frame data."""
        return self.frame_data
    
    def get_result_action(self) -> int:
        """Get result action (EXIT_BATCH, REVIEW_LABELS, NEXT_VIDEO)."""
        return self.result_action
