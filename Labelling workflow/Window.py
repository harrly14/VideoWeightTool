import csv
import cv2
import os
import sys
import numpy as np
import time
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QGroupBox, QMessageBox, QAction, QSpinBox
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator
from PyQt5.QtCore import Qt, QRegExp, QEvent, QTimer
from functools import partial
from VideoParams import VideoParams
from ZoomPanLabel import ZoomPanLabel
from FrameSampler import FrameSampler


class EditWindow(QWidget):
    def __init__(self, video_capture, video_filename: str, target_labels: int):
        """
        Initialize labelling window.
        
        Args:
            video_capture: cv2.VideoCapture object
            video_filename: Name of the video file (for output)
            target_labels: Target number of frames to label (n)
        """
        super().__init__()

        self.video_params = VideoParams()
        self.video = video_capture
        self.video_filename = video_filename
        self.target_labels = target_labels
        
        self.current_frame_index = 0
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_loaded_frame = -1
        self.current_raw_frame = None
        
        # Valid range for labelling (user-selectable)
        self.valid_range_start = 0
        self.valid_range_end = self.total_frames - 1
        
        # Target frames to label (calculated after valid range is set)
        self.target_frames = []
        self.target_frame_index = 0
        self.labelling_started = False
        
        self.frame_data = {}  # {frame_num: weight_string}
        self.processed_frames_count = 0

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

        self.setWindowTitle("Video Labelling Tool")
        self.resize(950, 600)

        main_layout = self.setup_main_ui()

        # Get original video dimensions
        success, first_frame = self.video.read()
        if success:
            self.original_h = first_frame.shape[0]
            self.original_w = first_frame.shape[1]
            self.video_label.set_original_size(self.original_w, self.original_h)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.setLayout(main_layout)
        self.load_frame_from_video()

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
            data_copy = dict(self.frame_data)
            with open(backup_file, 'w', newline='') as file:
                writer = csv.writer(file)
                for frame_num, weight in sorted(data_copy.items()):
                    writer.writerow([frame_num, weight])
        except Exception as e:
            print(f"Auto-save failed: {e}")

    def setup_main_ui(self) -> QVBoxLayout:
        """Setup main UI layout."""
        main_layout = QVBoxLayout()

        # Video display
        self.video_label = ZoomPanLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 300)
        main_layout.addWidget(self.video_label)

        # Valid range selection (before labelling starts)
        valid_range_group = QGroupBox("Valid Frame Range")
        valid_range_layout = QVBoxLayout()

        # Start frame slider/input
        start_layout = QHBoxLayout()
        start_label = QLabel("Start frame:")
        self.valid_start_slider = QSlider(Qt.Orientation.Horizontal)
        self.valid_start_slider.setRange(0, self.total_frames - 1)
        self.valid_start_slider.setValue(0)
        self.valid_start_slider.setToolTip("Select valid range start frame")
        self.valid_start_slider.valueChanged.connect(
            partial(self._debounced_slider, self.valid_start_slider, self.on_valid_start_changed)
        )
        self.valid_start_input = QSpinBox()
        self.valid_start_input.setRange(0, self.total_frames - 1)
        self.valid_start_input.setValue(0)
        self.valid_start_input.valueChanged.connect(self.on_valid_start_input_changed)
        self.valid_start_display = QLabel("0")
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.valid_start_slider)
        start_layout.addWidget(self.valid_start_input)
        start_layout.addWidget(self.valid_start_display)
        valid_range_layout.addLayout(start_layout)

        # End frame slider/input
        end_layout = QHBoxLayout()
        end_label = QLabel("End frame:")
        self.valid_end_slider = QSlider(Qt.Orientation.Horizontal)
        self.valid_end_slider.setRange(0, self.total_frames - 1)
        self.valid_end_slider.setValue(self.total_frames - 1)
        self.valid_end_slider.setToolTip("Select valid range end frame")
        self.valid_end_slider.valueChanged.connect(
            partial(self._debounced_slider, self.valid_end_slider, self.on_valid_end_changed)
        )
        self.valid_end_input = QSpinBox()
        self.valid_end_input.setRange(0, self.total_frames - 1)
        self.valid_end_input.setValue(self.total_frames - 1)
        self.valid_end_input.valueChanged.connect(self.on_valid_end_input_changed)
        self.valid_end_display = QLabel(str(self.total_frames - 1))
        end_layout.addWidget(end_label)
        end_layout.addWidget(self.valid_end_slider)
        end_layout.addWidget(self.valid_end_input)
        end_layout.addWidget(self.valid_end_display)
        valid_range_layout.addLayout(end_layout)

        valid_range_group.setLayout(valid_range_layout)
        main_layout.addWidget(valid_range_group)

        # Labelling controls (initially disabled)
        labelling_group = QGroupBox("Labelling")
        labelling_layout = QVBoxLayout()

        # Start labelling button
        self.start_labelling_button = QPushButton("Start Labelling")
        self.start_labelling_button.clicked.connect(self.start_labelling)
        labelling_layout.addWidget(self.start_labelling_button)

        # Weight entry
        weight_entry_layout = QHBoxLayout()
        weight_label = QLabel("Weight:")
        self.weight_entry = QLineEdit()
        self.weight_entry.setPlaceholderText("Enter weight (0-1 decimal or integer)")
        self.weight_entry.setToolTip("Enter weight for current frame (Enter/Space to submit)")
        self.weight_entry.setEnabled(False)

        pattern = QRegExp(r'^[0-9.,]*$')
        validator = QRegExpValidator(pattern)
        self.weight_entry.setValidator(validator)
        self.weight_entry.installEventFilter(self)
        self.weight_entry.returnPressed.connect(self.write_entry)

        weight_button = QPushButton("Submit")
        weight_button.clicked.connect(self.write_entry)
        weight_button.setEnabled(False)
        self.weight_button = weight_button

        weight_entry_layout.addWidget(weight_label)
        weight_entry_layout.addWidget(self.weight_entry)
        weight_entry_layout.addWidget(weight_button)
        labelling_layout.addLayout(weight_entry_layout)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setToolTip("Go to previous target frame (Left arrow)")
        self.prev_button.setEnabled(False)

        self.prev_action = QAction("Previous frame", self)
        self.prev_action.setShortcut("Left")
        self.prev_action.triggered.connect(self.previous_frame)
        self.addAction(self.prev_action)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setToolTip("Go to next target frame (Right arrow)")
        self.next_button.setEnabled(False)

        self.next_action = QAction("Next frame", self)
        self.next_action.setShortcut("Right")
        self.next_action.triggered.connect(self.next_frame)
        self.addAction(self.next_action)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        labelling_layout.addLayout(nav_layout)

        # Frame scrub (for freely navigating, not target frames)
        scrub_layout = QHBoxLayout()
        scrub_label = QLabel("Frame Navigation:")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, self.total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setToolTip("Drag to navigate to any frame")
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(
            partial(self._debounced_slider, self.frame_slider, self.seek_to_frame)
        )
        self.frame_input = QSpinBox()
        self.frame_input.setRange(0, self.total_frames - 1)
        self.frame_input.setValue(0)
        self.frame_input.setEnabled(False)
        self.frame_input.valueChanged.connect(self.on_frame_input_changed)
        self.frame_display = QLabel(f"0 / {self.total_frames - 1}")
        scrub_layout.addWidget(scrub_label)
        scrub_layout.addWidget(self.frame_slider)
        scrub_layout.addWidget(self.frame_input)
        scrub_layout.addWidget(self.frame_display)
        labelling_layout.addLayout(scrub_layout)

        # Progress display
        progress_layout = QHBoxLayout()
        progress_label = QLabel("Progress:")
        self.progress_display = QLabel("0 / 0 target frames")
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_display)
        progress_layout.addStretch()
        labelling_layout.addLayout(progress_layout)

        labelling_group.setLayout(labelling_layout)
        main_layout.addWidget(labelling_group)

        # Display adjustments
        controls_layout = self.setup_controls()
        main_layout.addLayout(controls_layout)

        # Save and close button
        save_button = QPushButton("Save and Close")
        save_button.clicked.connect(self.save_and_close)
        main_layout.addWidget(save_button)

        return main_layout

    def setup_controls(self) -> QHBoxLayout:
        """Setup display adjustment sliders."""
        controls_layout = QHBoxLayout()

        # Brightness
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout()
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-255, 255)
        self.brightness_slider.setValue(self.video_params.brightness)
        self.brightness_slider.setToolTip("Adjust brightness (-255 to 255, 0 is normal)")
        self.brightness_slider.valueChanged.connect(
            partial(self._on_param_change, "brightness")
        )
        self.brightness_display = QLabel("0")
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(self.brightness_slider)
        brightness_row.addWidget(self.brightness_display)
        brightness_layout.addLayout(brightness_row)

        reset_brightness_button = QPushButton("Reset")
        reset_brightness_button.clicked.connect(partial(self.reset_slider, "brightness"))
        brightness_layout.addWidget(reset_brightness_button)
        brightness_group.setLayout(brightness_layout)
        controls_layout.addWidget(brightness_group)

        # Saturation
        saturation_group = QGroupBox("Saturation")
        saturation_layout = QVBoxLayout()
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(0, 300)
        self.saturation_slider.setValue(self.video_params.saturation)
        self.saturation_slider.setToolTip("Adjust saturation (0 to 300, 100 is normal)")
        self.saturation_slider.valueChanged.connect(
            partial(self._on_param_change, "saturation")
        )
        self.saturation_display = QLabel("100")
        saturation_row = QHBoxLayout()
        saturation_row.addWidget(self.saturation_slider)
        saturation_row.addWidget(self.saturation_display)
        saturation_layout.addLayout(saturation_row)

        reset_saturation_button = QPushButton("Reset")
        reset_saturation_button.clicked.connect(partial(self.reset_slider, "saturation"))
        saturation_layout.addWidget(reset_saturation_button)
        saturation_group.setLayout(saturation_layout)
        controls_layout.addWidget(saturation_group)

        # Contrast
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout()
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(self.video_params.contrast)
        self.contrast_slider.setToolTip("Adjust contrast (0 to 200, 100 is normal)")
        self.contrast_slider.valueChanged.connect(
            partial(self._on_param_change, "contrast")
        )
        self.contrast_display = QLabel("100")
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self.contrast_slider)
        contrast_row.addWidget(self.contrast_display)
        contrast_layout.addLayout(contrast_row)

        reset_contrast_button = QPushButton("Reset")
        reset_contrast_button.clicked.connect(partial(self.reset_slider, "contrast"))
        contrast_layout.addWidget(reset_contrast_button)
        contrast_group.setLayout(contrast_layout)
        controls_layout.addWidget(contrast_group)

        return controls_layout

    def eventFilter(self, obj, event):
        """Handle space key in weight entry."""
        if obj is self.weight_entry and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space:
                self.write_entry()
                return True
        return super().eventFilter(obj, event)

    def on_valid_start_changed(self, value):
        """Handle valid start frame slider change."""
        if value > self.valid_range_end:
            value = self.valid_range_end
            self.valid_start_slider.blockSignals(True)
            self.valid_start_slider.setValue(value)
            self.valid_start_slider.blockSignals(False)
        self.valid_range_start = value
        self.valid_start_input.blockSignals(True)
        self.valid_start_input.setValue(value)
        self.valid_start_input.blockSignals(False)
        self.valid_start_display.setText(str(value))
        self.seek_to_frame(value)

    def on_valid_start_input_changed(self, value):
        """Handle valid start frame input change."""
        self.valid_start_slider.blockSignals(True)
        self.valid_start_slider.setValue(value)
        self.valid_start_slider.blockSignals(False)
        self.on_valid_start_changed(value)

    def on_valid_end_changed(self, value):
        """Handle valid end frame slider change."""
        if value < self.valid_range_start:
            value = self.valid_range_start
            self.valid_end_slider.blockSignals(True)
            self.valid_end_slider.setValue(value)
            self.valid_end_slider.blockSignals(False)
        self.valid_range_end = value
        self.valid_end_input.blockSignals(True)
        self.valid_end_input.setValue(value)
        self.valid_end_input.blockSignals(False)
        self.valid_end_display.setText(str(value))

    def on_valid_end_input_changed(self, value):
        """Handle valid end frame input change."""
        self.valid_end_slider.blockSignals(True)
        self.valid_end_slider.setValue(value)
        self.valid_end_slider.blockSignals(False)
        self.on_valid_end_changed(value)

    def on_frame_input_changed(self, value):
        """Handle frame input spinbox change."""
        if not self.labelling_started:
            return
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        self.seek_to_frame(value)

    def start_labelling(self):
        """Calculate target frames and start labelling process."""
        # Warn user that valid range will be locked
        reply = QMessageBox.warning(
            self,
            "Start Labelling",
            f"Valid range will be locked to frames {self.valid_range_start}-{self.valid_range_end}.\n\n"
            f"You will label {self.target_labels} frames distributed throughout this range.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # Calculate target frames
        try:
            sampler = FrameSampler(
                self.total_frames,
                self.valid_range_start,
                self.valid_range_end,
                self.target_labels
            )
            self.target_frames, m, warning = sampler.get_target_frames()

            if warning:
                QMessageBox.warning(
                    self,
                    "Fewer Labels Available",
                    f"Valid range has fewer frames than target labels.\n"
                    f"Will label {len(self.target_frames)} frames instead of {self.target_labels}."
                )

            # Lock valid range sliders
            self.valid_start_slider.setEnabled(False)
            self.valid_end_slider.setEnabled(False)
            self.valid_start_input.setEnabled(False)
            self.valid_end_input.setEnabled(False)

            # Enable labelling controls
            self.weight_entry.setEnabled(True)
            self.weight_button.setEnabled(True)
            self.prev_button.setEnabled(True)
            self.next_button.setEnabled(True)
            self.frame_slider.setEnabled(True)
            self.frame_input.setEnabled(True)

            self.start_labelling_button.setEnabled(False)
            self.labelling_started = True

            # Navigate to first target frame
            self.target_frame_index = 0
            self.current_frame_index = self.target_frames[0]
            self.load_frame_from_video()
            self.update_progress_display()

        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Failed to start labelling: {e}")

    def _on_param_change(self, param_name: str, value):
        """Handle parameter slider change."""
        old_val = getattr(self.video_params, param_name)
        setattr(self.video_params, param_name, value)
        
        # Update display value
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
        """Reset slider to default value."""
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

    def seek_to_frame(self, frame_num):
        """Seek to specific frame."""
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.current_frame_index = frame_num
        self.frame_input.blockSignals(True)
        self.frame_input.setValue(frame_num)
        self.frame_input.blockSignals(False)
        self.load_frame_from_video()

    def update_image(self, frame):
        """Update displayed frame image."""
        qimage = self.cv2_to_qimage(frame)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def load_frame_from_video(self):
        """Load frame from video file."""
        if self.last_loaded_frame != self.current_frame_index:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

            time0 = time.time()
            success, frame = self.video.read()
            duration = time.time() - time0

            if duration >= self.read_timeout_warning_threshold or not success:
                QMessageBox.critical(
                    None,
                    "Error",
                    "Frame took too long to read. If using a network file, consider making a local copy."
                )
                sys.exit(-1)
            else:
                self.current_raw_frame = frame
                self.last_loaded_frame = self.current_frame_index

        if self.current_raw_frame is not None:
            # Load weight for current frame (if labelling started)
            if self.labelling_started and self.current_frame_index in self.target_frames:
                value = self.frame_data.get(self.current_frame_index, "0")
                self.weight_entry.setText("" if value == "0" else value)

            # Update navigation buttons
            if self.labelling_started:
                self.prev_button.setEnabled(self.target_frame_index > 0)
                self.next_button.setEnabled(self.target_frame_index < len(self.target_frames) - 1)

            self.update_display_frame()
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_index)
            self.frame_slider.blockSignals(False)
            self.frame_display.setText(f"{self.current_frame_index} / {self.total_frames - 1}")

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

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        self.update_display_frame()

    def write_entry(self):
        """Save weight entry for current frame."""
        if not self.labelling_started:
            return

        value = self.weight_entry.text().strip()

        if not value:
            return

        try:
            self.frame_data[self.current_frame_index] = value
            self.processed_frames_count += 1
            self.update_progress_display()

            # Auto-advance to next target frame
            if self.target_frame_index < len(self.target_frames) - 1:
                self.next_frame()

        except ValueError:
            QMessageBox.warning(None, "Error", "Invalid number entered")

    def update_progress_display(self):
        """Update progress display."""
        self.progress_display.setText(
            f"{self.processed_frames_count} / {len(self.target_frames)} target frames"
        )

    def next_frame(self):
        """Navigate to next target frame."""
        if not self.labelling_started or self.target_frame_index >= len(self.target_frames) - 1:
            return

        # Save current weight if entered
        if self.weight_entry.text().strip():
            self.frame_data[self.current_frame_index] = self.weight_entry.text().strip()

        self.target_frame_index += 1
        self.current_frame_index = self.target_frames[self.target_frame_index]
        self.load_frame_from_video()

    def previous_frame(self):
        """Navigate to previous target frame."""
        if not self.labelling_started or self.target_frame_index <= 0:
            return

        # Save current weight if entered
        if self.weight_entry.text().strip():
            self.frame_data[self.current_frame_index] = self.weight_entry.text().strip()

        self.target_frame_index -= 1
        self.current_frame_index = self.target_frames[self.target_frame_index]
        self.load_frame_from_video()

    def confirm_close(self) -> bool:
        """Confirm window close."""
        reply = QMessageBox.question(
            self,
            "Close",
            "Are you sure you want to close? Any unsaved labels will be lost.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        """Handle window close event."""
        if self.confirm_close():
            # Clean up backup
            backup_file = f".backup_{self.video_filename}.csv"
            if os.path.exists(backup_file):
                try:
                    os.remove(backup_file)
                except:
                    pass
            event.accept()
        else:
            event.ignore()

    def save_and_close(self):
        """Save frame data and close window."""
        # Save current weight entry
        if self.weight_entry.text().strip():
            self.frame_data[self.current_frame_index] = self.weight_entry.text().strip()

        self.close()

    def cv2_to_qimage(self, cv_img):
        """Convert OpenCV image to QImage."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        convert_to_qt_format = QImage(
            rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        return convert_to_qt_format
