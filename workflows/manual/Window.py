import csv
import cv2
import os
import sys
import numpy as np
import time
from collections import OrderedDict
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QGroupBox, QMessageBox, QAction, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator
from PyQt5.QtCore import Qt, QRegExp, QEvent, QTimer
from qtrangeslider import QRangeSlider
from functools import partial
from core.config import VideoParams
from core.ui.RectCropLabel import RectCropLabel


class EditWindow(QWidget): 
    def __init__(self, video_capture, csv_path):
        super().__init__()

        self.video_params = VideoParams()
        self.video = video_capture
        self.current_frame_index = 0
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_loaded_frame = -1
        self.current_raw_frame = None
        self.start_frame = 0
        self.end_frame = self.total_frames
        self.csv_path = csv_path
        self.processed_frames_count = 0
        self.frame_data = self.load_csv_data()
        self.temporal_frame_cache = OrderedDict()
        self.temporal_cache_max_size = 20

        self.read_timeout_warning_threshold = 5.0

        self.slider_timer = QTimer()
        self.slider_timer.setSingleShot(True)
        self.slider_timer.setInterval(100)
        self.pending_slider_action = None
        self.slider_timer.timeout.connect(self._execute_pending_slider_action)

        self.autosave_timer = QTimer()
        self.autosave_timer.setInterval(60000) # 1 minute
        self.autosave_timer.timeout.connect(self.auto_save_backup)
        self.autosave_timer.start()

        self.setWindowTitle("Video editing tool")
        self.resize(950, 465)

        main_layout = self.setup_main_ui()

        # Get original video dimensions
        success, first_frame = self.video.read()
        if success:
            self.original_h = first_frame.shape[0]
            self.original_w = first_frame.shape[1]
            self.video_label.set_original_size(self.original_w, self.original_h)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.setLayout(main_layout)
        self.current_frame_index = self.processed_frames_count
        self.load_frame_from_video()
    
    def _debounced_slider(self, slider, value_changed_func):
        self.slider_timer.stop()
        self.pending_slider_action = lambda: value_changed_func(slider.value())
        self.slider_timer.start()
    
    def _execute_pending_slider_action(self):
        if self.pending_slider_action:
            self.pending_slider_action()
            self.pending_slider_action = None
    
    def auto_save_backup(self):
        backup_file = self.csv_path + ".backup"
        try: 
            data_copy = dict(self.frame_data)
            with open(backup_file, 'w', newline='') as backup_file:
                writer = csv.writer(backup_file)
                writer.writerow(["frame_num", "weight"])

                for i in range(self.total_frames):
                    value = data_copy.get(i, "0")
                    writer.writerow([i,value])

            backup_files = sorted([file for file in os.listdir(os.path.dirname(self.csv_path)) 
                              if file.startswith(os.path.basename(self.csv_path)) and file.endswith('.backup')])
            for old_file in backup_files[:-3]:
                try:
                    os.remove(os.path.join(os.path.dirname(self.csv_path), old_file))
                except:
                    pass
        except Exception as e:
            print(f"Auto-save failed: {e}")

    def setup_main_ui(self):
        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()

        self.video_label = RectCropLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 200)
        self.video_label.warp_changed.connect(self.on_warp_changed)
        video_layout.addWidget(self.video_label)

        nav_layout = QHBoxLayout()
        
        weight_label = QLabel("Enter weight:")
        self.weight_entry = QLineEdit()
        self.weight_entry.setPlaceholderText("Weight")
        self.weight_entry.setToolTip("Enter weight for the current frame (Enter/Space to submit)")

        pattern = QRegExp(r'^[0-9.,]*$')
        validator = QRegExpValidator(pattern)
        self.weight_entry.setValidator(validator)
        self.weight_entry.installEventFilter(self)

        self.weight_entry.returnPressed.connect(self.write_entry)

        self.focus_weight_action = QAction("Focus Weight Entry", self)
        self.focus_weight_action.setShortcut("W")
        self.focus_weight_action.triggered.connect(lambda: self.weight_entry.setFocus())
        self.addAction(self.focus_weight_action)

        weight_enter_button = QPushButton("Enter")
        weight_enter_button.clicked.connect(self.write_entry)

        weight_row = QHBoxLayout()
        weight_row.addWidget(weight_label)
        weight_row.addWidget(self.weight_entry)
        weight_row.addWidget(weight_enter_button)
        video_layout.addLayout(weight_row)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_frame)
        self.prev_button.setToolTip("Go to previous frame (Left arrow)")

        self.prev_action = QAction("Previous frame", self)
        self.prev_action.setShortcut("Left")
        self.prev_action.triggered.connect(self.previous_frame)
        self.addAction(self.prev_action)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setToolTip("Go to next frame (Right arrow)")

        self.next_action = QAction("Next frame", self)
        self.next_action.setShortcut("Right")
        self.next_action.triggered.connect(self.next_frame)
        self.addAction(self.next_action)    

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        video_layout.addLayout(nav_layout)

        scrub_layout = QHBoxLayout()
        self.scrub_label = QLabel("Scrub:")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, self.total_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setToolTip("Drag to navigate to a specific frame")
        self.frame_slider.valueChanged.connect(partial(self._debounced_slider, self.frame_slider, self.seek_to_frame))
        scrub_layout.addWidget(self.scrub_label)
        scrub_layout.addWidget(self.frame_slider)
        video_layout.addLayout(scrub_layout)

        self.jump_back_action = QAction("Jump Back 10 Frames", self)
        self.jump_back_action.setShortcut("Ctrl+Left")
        self.jump_back_action.triggered.connect(lambda: self.seek_to_frame(max(self.start_frame, self.current_frame_index - 10)))
        self.addAction(self.jump_back_action)

        self.jump_forward_action = QAction("Jump Forward 10 Frames", self)
        self.jump_forward_action.setShortcut("Ctrl+Right")
        self.jump_forward_action.triggered.connect(lambda: self.seek_to_frame(min(self.end_frame - 1, self.current_frame_index + 10)))
        self.addAction(self.jump_forward_action)

        trim_layout = QHBoxLayout()
        self.trim_label = QLabel("Trim:  ")
        self.trim_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.trim_slider.setStyleSheet("QRangeSlider {qproperty-barColor: #447;}")
        self.trim_slider.setRange(0, self.total_frames - 1)
        self.trim_slider.setValue((0, self.total_frames - 1))
        self.trim_slider.setToolTip("Drag handles to set start and end points of the video")
        self.trim_slider.valueChanged.connect(partial(self._debounced_slider, self.trim_slider, self.apply_trim))
        trim_layout.addWidget(self.trim_label)
        trim_layout.addWidget(self.trim_slider)
        video_layout.addLayout(trim_layout)

        main_layout.addLayout(video_layout, 2)


        controls_layout = self.setup_controls()
        main_layout.addLayout(controls_layout, 1)
        return main_layout

    def setup_controls(self):
        controls_layout = QVBoxLayout()

        crop_group = QGroupBox("Crop")
        crop_layout = QVBoxLayout()
        self.crop_info_label = QLabel("Enable Warp crop, adjust corners, then confirm")
        self.crop_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.warp_button = QPushButton("Edit warp crop")
        self.warp_button.setToolTip("Edit perspective crop corners")
        self.warp_button.clicked.connect(self.start_warp_crop_edit)

        self.confirm_warp_crop_button = QPushButton("Confirm warp crop")
        self.confirm_warp_crop_button.clicked.connect(self.confirm_warp_crop)
        self.clear_crop_button = QPushButton("Reset warped crop")
        self.clear_crop_button.clicked.connect(self.clear_perspective_warp)
        crop_layout.addWidget(self.crop_info_label)
        crop_layout.addWidget(self.warp_button)
        crop_layout.addWidget(self.confirm_warp_crop_button)
        crop_layout.addWidget(self.clear_crop_button)
        crop_group.setLayout(crop_layout)
        controls_layout.addWidget(crop_group)
      
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout()
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-255, 255) 
        self.brightness_slider.setValue(self.video_params.brightness)
        self.brightness_slider.setToolTip("Adjust brightness (-255 to 255, 0 is normal)")
        self.brightness_slider.valueChanged.connect(partial(self._on_param_change, "brightness"))
        brightness_layout.addWidget(self.brightness_slider)
        self.reset_brightness_button = QPushButton("Reset brightness")
        self.reset_brightness_button.clicked.connect(partial(self.reset_slider, "brightness"))
        brightness_layout.addWidget(self.reset_brightness_button)
        brightness_group.setLayout(brightness_layout)
        controls_layout.addWidget(brightness_group)

        saturation_group = QGroupBox("Saturation")
        saturation_layout = QVBoxLayout()
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(0, 300) # 0 to 300, where 100 = 1.0
        self.saturation_slider.setValue(self.video_params.saturation)
        self.saturation_slider.setToolTip("Adjust saturation (0 to 300, 100 is normal)")
        self.saturation_slider.valueChanged.connect(partial(self._on_param_change, "saturation")) 
        saturation_layout.addWidget(self.saturation_slider)
        self.reset_saturation_button = QPushButton("Reset saturation")
        self.reset_saturation_button.clicked.connect(partial(self.reset_slider, "saturation"))
        saturation_layout.addWidget(self.reset_saturation_button)
        saturation_group.setLayout(saturation_layout)
        controls_layout.addWidget(saturation_group)

        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout()
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200) # 0 to 200, where 100 = 1.0
        self.contrast_slider.setValue(self.video_params.contrast)
        self.contrast_slider.setToolTip("Adjust contrast (0 to 200, 100 is normal)")
        self.contrast_slider.valueChanged.connect(partial(self._on_param_change, "contrast"))
        contrast_layout.addWidget(self.contrast_slider)
        self.reset_contrast_button = QPushButton("Reset contrast")
        self.reset_contrast_button.clicked.connect(partial(self.reset_slider, "contrast"))
        contrast_layout.addWidget(self.reset_contrast_button)
        contrast_group.setLayout(contrast_layout)
        controls_layout.addWidget(contrast_group)

        self.clahe_button = QPushButton("CLAHE: Off")
        self.clahe_button.setToolTip("Toggle CLAHE (Contrast Limited Adaptive Histogram Equalization) preview")
        self.clahe_button.setCheckable(True)
        self.clahe_button.setChecked(False)
        self.clahe_button.toggled.connect(self._update_clahe_button_text)
        self.clahe_button.clicked.connect(self.update_display_frame)
        controls_layout.addWidget(self.clahe_button)

        temporal_group = QGroupBox("Temporal averaging")
        temporal_layout = QVBoxLayout()

        self.temporal_button = QPushButton("Temporal Avg: Off")
        self.temporal_button.setCheckable(True)
        self.temporal_button.setChecked(self.video_params.temporal_avg_enabled)
        self.temporal_button.setToolTip("Temporal frame averaging preview (pixel-level only)")
        self.temporal_button.toggled.connect(self._on_temporal_toggle)
        self._update_temporal_button_text(self.temporal_button.isChecked())
        temporal_layout.addWidget(self.temporal_button)

        temporal_window_row = QHBoxLayout()
        temporal_window_label = QLabel("Window:")
        self.temporal_window_combo = QComboBox()
        self.temporal_window_combo.addItems(["1", "3", "5", "7", "9"])
        current_window = str(self.video_params.temporal_avg_window)
        if current_window not in ["1", "3", "5", "7", "9"]:
            self.temporal_window_combo.addItem(current_window)
        self.temporal_window_combo.setCurrentText(current_window)
        self.temporal_window_combo.currentTextChanged.connect(self._on_temporal_window_changed)
        temporal_window_row.addWidget(temporal_window_label)
        temporal_window_row.addWidget(self.temporal_window_combo)
        temporal_layout.addLayout(temporal_window_row)

        temporal_group.setLayout(temporal_layout)
        controls_layout.addWidget(temporal_group)

        frame_info_group = QGroupBox("Frame info")
        frame_info_layout = QVBoxLayout()

        scrub_info_layout = QHBoxLayout()
        scrub_info_label = QLabel("Current frame:")
        self.scrub_frame_label = QLabel(f"{self.current_frame_index + 1}/{self.end_frame}")
        scrub_info_layout.addWidget(scrub_info_label)
        scrub_info_layout.addWidget(self.scrub_frame_label)
        frame_info_layout.addLayout(scrub_info_layout)

        processed_frames_layout = QHBoxLayout()
        processed_info_label = QLabel("Processed frames:")
        self.processed_frame_label = QLabel(f"{self.processed_frames_count}/{self.end_frame}")
        processed_frames_layout.addWidget(processed_info_label)
        processed_frames_layout.addWidget(self.processed_frame_label)
        frame_info_layout.addLayout(processed_frames_layout)

        scrub_to_layout = QHBoxLayout()
        scrub_to_label = QLabel("Scrub to:")
        self.scrub_to_entry = QLineEdit()
        self.scrub_to_entry.setFixedWidth(90)
        self.scrub_to_entry.setPlaceholderText(f"1-{self.total_frames}")
        self.scrub_to_entry.setToolTip("Jump to frame number (1-based), then press Enter")
        self.scrub_to_entry.setValidator(QRegExpValidator(QRegExp(r"^[0-9]*$")))
        self.scrub_to_entry.returnPressed.connect(self.scrub_to_frame)
        scrub_to_layout.addWidget(scrub_to_label)
        scrub_to_layout.addWidget(self.scrub_to_entry)
        frame_info_layout.addLayout(scrub_to_layout)

        frame_info_group.setLayout(frame_info_layout)
        controls_layout.addWidget(frame_info_group)

        apply_button = QPushButton("Save and close")
        apply_button.clicked.connect(self.save_and_close)
        controls_layout.addWidget(apply_button)

        return controls_layout

    def eventFilter(self, obj, event):
        if obj is self.weight_entry and event.type() == QEvent.KeyPress: #type: ignore[attr-defined]
            if event.key() == Qt.Key_Space: #type: ignore[attr-defined]           
                self.write_entry()
                return True #consume space key
        return super().eventFilter(obj, event)
    
    def _on_param_change(self, param_name, value):
        old_val  = getattr(self.video_params, param_name)
        setattr(self.video_params, param_name, value)
        try:
            self.video_params.validate()
            self.update_display_frame()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid parameter", str(e))
            setattr(self.video_params, param_name, old_val)

    def _update_clahe_button_text(self, checked):
        if checked:
            self.clahe_button.setText("CLAHE: On")
        else:
            self.clahe_button.setText("CLAHE: Off")

    def _update_temporal_button_text(self, checked):
        self.temporal_button.setText("Temporal Avg: On" if checked else "Temporal Avg: Off")

    def _on_temporal_toggle(self, checked):
        self.video_params.temporal_avg_enabled = checked
        self._update_temporal_button_text(checked)
        if checked:
            self._prefetch_temporal_neighbors()
        self.update_display_frame()

    def _on_temporal_window_changed(self, value):
        try:
            window = int(value)
        except ValueError:
            return
        self.video_params.temporal_avg_window = window
        self.video_params.validate()
        self._prefetch_temporal_neighbors()
        self.update_display_frame()

    def _reset_temporal_cache(self, keep_current=True):
        self.temporal_frame_cache.clear()
        if keep_current and self.current_raw_frame is not None:
            self._cache_frame(self.current_frame_index, self.current_raw_frame)

    def _cache_frame(self, frame_index, frame):
        self.temporal_frame_cache[frame_index] = frame
        self.temporal_frame_cache.move_to_end(frame_index)
        while len(self.temporal_frame_cache) > self.temporal_cache_max_size:
            self.temporal_frame_cache.popitem(last=False)

    def _get_temporal_indices(self, center_index):
        half_window = self.video_params.temporal_avg_window // 2
        start_index = max(self.start_frame, center_index - half_window)
        end_index = min(self.end_frame - 1, center_index + half_window)
        return range(start_index, end_index + 1)

    def _get_frame_from_cache(self, frame_index):
        if frame_index in self.temporal_frame_cache:
            self.temporal_frame_cache.move_to_end(frame_index)
            return self.temporal_frame_cache[frame_index]

        if frame_index == self.current_frame_index and self.current_raw_frame is not None:
            self._cache_frame(frame_index, self.current_raw_frame)
            return self.current_raw_frame

        restore_pos = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self.video.read()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, restore_pos)

        if not success:
            return None

        self._cache_frame(frame_index, frame)
        return frame

    def _prefetch_temporal_neighbors(self):
        if not self.video_params.temporal_avg_enabled or self.current_raw_frame is None:
            return
        for frame_index in self._get_temporal_indices(self.current_frame_index):
            self._get_frame_from_cache(frame_index)

    def _get_temporal_averaged_frame(self):
        if not self.video_params.temporal_avg_enabled or self.current_raw_frame is None:
            return self.current_raw_frame
        if self.video_params.temporal_avg_window <= 1:
            return self.current_raw_frame

        frames_to_average = []
        for frame_index in self._get_temporal_indices(self.current_frame_index):
            frame = self._get_frame_from_cache(frame_index)
            if frame is not None:
                frames_to_average.append(frame.astype(np.float32))

        if not frames_to_average:
            return self.current_raw_frame

        averaged = np.mean(frames_to_average, axis=0)
        return np.clip(averaged, 0, 255).astype(np.uint8)

    def apply_clahe(self, frame):
        from core.roi_utils import apply_clahe
        # apply_clahe returns single-channel (H, W, 1), convert back to 3-channel BGR for display
        single_channel = apply_clahe(frame)
        grayscale_2d = single_channel[:, :, 0]  # Extract (H, W) from (H, W, 1)
        return cv2.cvtColor(grayscale_2d, cv2.COLOR_GRAY2BGR)

    def reset_slider(self, param_name):
        default_val = self.video_params.get_default_value(param_name)
        if default_val is None:
            return
        if param_name == "brightness":
            self.brightness_slider.setValue(default_val)
            self.video_params.brightness = default_val
        elif param_name == "saturation":
            self.saturation_slider.setValue(default_val)
            self.video_params.saturation = default_val
        elif param_name == "contrast": 
            self.contrast_slider.setValue(default_val)
            self.video_params.contrast = default_val

        self.update_display_frame()

    def _default_warp_quad(self):
        if not hasattr(self, "original_w") or not hasattr(self, "original_h"):
            return None
        return [
            (0, 0),
            (self.original_w - 1, 0),
            (self.original_w - 1, self.original_h - 1),
            (0, self.original_h - 1),
        ]

    def start_warp_crop_edit(self):
        # Enter editing mode on the original preview so the user can reshape the quad.
        self.video_label.set_warp_mode(True)

        self.video_params.warp_enabled = False
        self.crop_info_label.setText("Adjust corners, then click Confirm warp crop")

        if self.video_params.warp_quad is None:
            default_quad = self._default_warp_quad()
            if default_quad is not None:
                self.video_params.warp_quad = default_quad
                self.video_label.set_warp_quad(default_quad)
        else:
            self.video_label.set_warp_quad(self.video_params.warp_quad)

        self.video_label.update()
        self.update_display_frame()

    def on_warp_changed(self, points):
        if not points:
            self.video_params.warp_quad = None
            return
        normalized = []
        for point in points:
            try:
                normalized.append((int(point[0]), int(point[1])))
            except (TypeError, ValueError, IndexError):
                return
        if len(normalized) != 4:
            return
        self.video_params.warp_quad = normalized

    def confirm_warp_crop(self):
        if not self.video_params.warp_quad or len(self.video_params.warp_quad) != 4:
            QMessageBox.warning(self, "No warp crop", "Enable Warp crop and position all four corners first.")
            return

        self.video_params.crop_coords = None
        self.video_params.warp_enabled = True
        self.video_label.set_warp_mode(False)
        self.crop_info_label.setText("Warped crop confirmed")
        self.update_display_frame()

    def clear_perspective_warp(self):
        self.video_params.warp_quad = None
        self.video_params.warp_enabled = False
        self.video_params.crop_coords = None
        self.video_label.clear_warp_quad()
        self.video_label.set_warp_mode(False)
        self.crop_info_label.setText("Enable Warp crop, adjust corners, then confirm")
        self.update_display_frame()

    def apply_perspective_warp(self, frame, quad):
        if frame is None or quad is None or len(quad) != 4:
            return frame
        height, width = frame.shape[:2]
        try:
            src = np.array(quad, dtype=np.float32)
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_LINEAR)
        except Exception:
            return frame

    def _update_scrub_to_bounds(self):
        # user-facing frame stuff is 1-indexed instead of 0
        min_user_frame = self.start_frame + 1
        max_user_frame = self.end_frame
        self.scrub_to_entry.setPlaceholderText(f"{min_user_frame}-{max_user_frame}")
        self.scrub_to_entry.setToolTip(
            f"Jump to frame number ({min_user_frame}-{max_user_frame}), then press Enter"
        )

    def scrub_to_frame(self):
        value = self.scrub_to_entry.text().strip()
        if not value:
            return

        try:
            user_frame = int(value)
        except ValueError:
            QMessageBox.warning(self, "Invalid frame", "Please enter a valid frame number")
            return

        min_user_frame = self.start_frame + 1
        max_user_frame = self.end_frame

        if not (min_user_frame <= user_frame <= max_user_frame):
            QMessageBox.warning(
                self,
                "Frame out of range",
                f"Please enter a frame between {min_user_frame} and {max_user_frame}."
            )
            return

        self.seek_to_frame(user_frame - 1)

    def seek_to_frame(self, frame_num):
        if abs(frame_num - self.current_frame_index) > self.temporal_cache_max_size:
            self._reset_temporal_cache(keep_current=False)
        self.current_frame_index = frame_num
        self.load_frame_from_video()

    def apply_trim(self, values):
        new_start, new_end = values
        if new_start >= new_end:
            new_start = new_end - 1
        new_start = max(0, min(new_start, self.total_frames - 1))
        new_end = max(1, min(new_end, self.total_frames))

        if new_start == self.start_frame and new_end == self.end_frame:
            return
        
        old_start = self.video_params.trim_start
        old_end = self.video_params.trim_end

        try:
            self.start_frame = new_start
            self.end_frame = new_end
            self.video_params.trim_start = self.start_frame
            self.video_params.trim_end = self.end_frame
            self.video_params.validate()

            self.prev_button.setEnabled(False if self.current_frame_index <= self.start_frame else True)
            self.next_button.setEnabled(False if self.current_frame_index >= self.end_frame - 1 else True)

            self.processed_frame_label.setText(f"{self.processed_frames_count}/{self.end_frame}")
            self.scrub_frame_label.setText(f"{self.current_frame_index + 1}/{self.end_frame}")
            self._update_scrub_to_bounds()
            self._reset_temporal_cache(keep_current=True)
            
            if self.current_frame_index < self.start_frame or self.current_frame_index >= self.end_frame:
                self.seek_to_frame(self.start_frame)
            elif self.last_loaded_frame != self.current_frame_index:
                self.load_frame_from_video()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid parameter", str(e))
            self.video_params.trim_start = old_start
            self.video_params.trim_end = old_end
 

    def update_image(self, frame):
        qimage = self.cv2_to_qimage(frame)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def load_frame_from_video(self):
        if self.last_loaded_frame != self.current_frame_index:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)

            time0 = time.time()
            success, frame = self.video.read()
            duration = time.time() - time0
            
            if (duration >= self.read_timeout_warning_threshold) or (not success):
                QMessageBox.critical(None, 'Error', 'Frame took too long to read. ' \
                'If using a file on a network share, consider making a local copy. You can find a backup of your CSV in the specified outputs folder.')
                try:
                    if hasattr(self.video, "release"):
                        self.video.release()
                except Exception:
                    pass
                sys.exit(-1)
            else:
                self.current_raw_frame = frame
                self.last_loaded_frame = self.current_frame_index
                self.scrub_frame_label.setText(f"{self.current_frame_index + 1}/{self.end_frame}")
                self._cache_frame(self.current_frame_index, self.current_raw_frame)
                self._prefetch_temporal_neighbors()

            
        if self.current_raw_frame is not None:
            value = self.frame_data.get(self.current_frame_index, "0")
            self.weight_entry.setText("" if value == '0' else value)
            self.scrub_to_entry.setText(str(self.current_frame_index + 1))
            self._update_scrub_to_bounds()

            self.prev_button.setEnabled(False if self.current_frame_index <= self.start_frame else True)
            self.next_button.setEnabled(False if self.current_frame_index >= self.end_frame - 1 else True)

            self.update_display_frame()
        
        self.frame_slider.setValue(self.current_frame_index)

    def update_display_frame(self):
        if self.current_raw_frame is None:
            self.video_label.setText("No frame loaded")
            return
        
        frame = self._get_temporal_averaged_frame()
        if self.clahe_button.isChecked():
            frame = self.apply_clahe(frame)
        
        preview_frame = self.apply_preview_effects(frame)

        if self.video_params.warp_enabled and self.video_params.warp_quad:
            preview_frame = self.apply_perspective_warp(preview_frame, self.video_params.warp_quad)

        if self.video_params.crop_coords:
            x, y, w, h = self.video_params.crop_coords
            # Ensure crop coordinates are within the frame dimensions
            h_max, w_max, _ = preview_frame.shape
            x, y, w, h = max(0, x), max(0, y), min(w, w_max - x), min(h, h_max - y)
            preview_frame = preview_frame[y:y+h, x:x+w]

        self.update_image(preview_frame)

    def apply_preview_effects(self, frame):
        if (self.video_params.contrast == 100 and 
            self.video_params.brightness == 0 and 
            self.video_params.saturation == 100):
            return frame
        
        processed = frame.astype(np.float32)

        # Contrast + Brightness (in YCrCb to preserve color like ffmpeg)
        if self.video_params.contrast != 100 or self.video_params.brightness != 0:
            ycrcb = cv2.cvtColor(processed.astype(np.uint8), cv2.COLOR_BGR2YCrCb).astype(np.float32)
            contrast_factor = self.video_params.contrast / 100.0
            ycrcb[:, :, 0] = (ycrcb[:, :, 0] - 128.0) * contrast_factor + 128.0 + self.video_params.brightness
            processed = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        else:
            processed = processed.astype(np.uint8)

        # Saturation
        if self.video_params.saturation != 100:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (self.video_params.saturation / 100.0), 0, 255)
            processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return processed
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_frame()

    def write_entry(self):
        value = self.weight_entry.text().strip()
        
        if not value: return
        
        try:
            self.frame_data[self.current_frame_index] = value
            self.save_to_csv()
            self.processed_frames_count += 1
            self.processed_frame_label.setText(f"{self.processed_frames_count}/{self.end_frame}")
            
            if self.current_frame_index < self.end_frame - 1:
                self.next_frame()
                
        except ValueError:
            QMessageBox.warning(None, "Error", "Invalid number entered")
    
    def save_to_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_num", "weight"])
            
            for i in range(self.total_frames):
                value = self.frame_data.get(i, "0")
                writer.writerow([i, value])

    def load_csv_data(self):
        data = {i: "0" for i in range (self.total_frames)}
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_num = int(row.get('frame_num', -1))
                    weight = row.get('weight', "0")
                    if 0 <= frame_num < self.total_frames:
                        data[frame_num] = weight
                        if weight != "0":
                            self.processed_frames_count += 1
        except FileNotFoundError:
            QMessageBox.critical(self, 'Error', 'CSV file not found')
        return data

    def cv2_to_qimage(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        convert_to_Qt_format = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format
    
    def next_frame(self):
        if self.current_frame_index < self.end_frame - 1:
            if self.weight_entry.text().strip():
                self.frame_data[self.current_frame_index] = self.weight_entry.text().strip()
                self.save_to_csv()
            self.current_frame_index += 1
            self.load_frame_from_video()

    def previous_frame(self):
        if self.current_frame_index > self.start_frame:
            if self.weight_entry.text().strip():
                self.frame_data[self.current_frame_index] = self.weight_entry.text().strip()
                self.save_to_csv()
            self.current_frame_index -= 1
            self.load_frame_from_video()

    def confirm_close(self):
        try:
            self.video_params.validate()
        except ValueError as e:
            QMessageBox.critical(self, "Validation Error", f"Cannot close: {e}\nPlease fix the parameters.")
            return 
        
        reply = QMessageBox.question(self, 'Message', 'Are you sure you want to quit?',
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        if self.confirm_close():
            backup_csv = self.csv_path + ".backup"
            if os.path.exists(backup_csv):
                os.remove(backup_csv)
            event.accept()
        else:
            event.ignore()

    def save_and_close(self):
        value = self.weight_entry.text().strip()
        if value:
            self.frame_data[self.current_frame_index] = value
        self.save_to_csv()
        self.close()