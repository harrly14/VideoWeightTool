import csv
import cv2
import os
import sys
import numpy as np
import time
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QGroupBox, QMessageBox, QAction
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator
from PyQt5.QtCore import Qt, QRegExp, QEvent, QTimer
from qtrangeslider import QRangeSlider
from functools import partial
from VideoParams import VideoParams
from CropLabel import CropImageLabel


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
        self.frame_data = self.load_csv_data()

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

        # get original vieo dimensions
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
    
    def auto_save_backup(self):
        backup_file = self.csv_path + ".backup"
        try: 
            data_copy = dict(self.frame_data)
            with open(backup_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["frame_num", "weight"])

                for i in range(self.total_frames):
                    value = data_copy.get(i, "0")
                    writer.writerow([i,value])

            backup_files = sorted([f for f in os.listdir(os.path.dirname(self.csv_path)) 
                              if f.startswith(os.path.basename(self.csv_path)) and f.endswith('.backup')])
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

        self.video_label = CropImageLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(400, 200)
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

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_frame)
        self.prev_button.setToolTip("Go to next frame (Right arrow)")

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

        controls_layout = self.setup_controls()
        main_layout.addLayout(controls_layout, 1)
        return main_layout

    def setup_controls(self):
        controls_layout = QVBoxLayout()

        crop_group = QGroupBox("Crop")
        crop_layout = QVBoxLayout()
        self.crop_info_label = QLabel("Click and drag on video to select crop area")
        self.crop_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clear_crop_button = QPushButton("Clear crop")
        self.clear_crop_button.clicked.connect(self.clear_crop)
        crop_layout.addWidget(self.crop_info_label)
        crop_layout.addWidget(self.clear_crop_button)
        crop_group.setLayout(crop_layout)
        controls_layout.addWidget(crop_group)
      
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout()
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-255, 255) 
        self.brightness_slider.setValue(self.video_params.brightness)
        self.brightness_slider.setToolTip("Adjust brightness (-255 to 255, 0 is norrmal)")
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
        self.saturation_slider.setToolTip("Adjust saturation (0 to 300, 100 is norrmal)")
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
        self.contrast_slider.setToolTip("Adjust contrast (0 to 200, 100 is norrmal)")
        self.contrast_slider.valueChanged.connect(partial(self._on_param_change, "contrast"))
        contrast_layout.addWidget(self.contrast_slider)
        self.reset_contrast_button = QPushButton("Reset contrast")
        self.reset_contrast_button.clicked.connect(partial(self.reset_slider, "contrast"))
        contrast_layout.addWidget(self.reset_contrast_button)
        contrast_group.setLayout(contrast_layout)
        controls_layout.addWidget(contrast_group)

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

    def on_crop_selected(self, x, y, w, h):
        old_crop_coords = self.video_params.crop_coords
        try:
            self.video_params.crop_coords = (x, y, w, h)
            self.video_params.validate()

            self.crop_info_label.setText("Cropped.")
            self.video_label.cropping_enabled = False
            self.update_display_frame()

            self.video_label.crop_start = None
            self.video_label.crop_end = None
            self.video_label.update()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid parameter", str(e))
            self.video_params.crop_coords = old_crop_coords
            
    def clear_crop(self):
        self.video_params.crop_coords = None
        self.crop_info_label.setText("Click and drag on the video to select crop area")
        self.video_label.crop_start = None
        self.video_label.crop_end = None
        self.video_label.cropping_enabled = True
        self.video_label.update()
        self.update_display_frame()

    def seek_to_frame(self, frame_num):
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

            
        if self.current_raw_frame is not None:
                value = self.frame_data.get(self.current_frame_index, "0")
                self.weight_entry.setText("" if value == '0' else value)

                self.prev_button.setEnabled(False if self.current_frame_index <= self.start_frame else True)
                self.next_button.setEnabled(False if self.current_frame_index >= self.end_frame - 1 else True)

                self.update_display_frame()
        
        self.frame_slider.setValue(self.current_frame_index)

    def update_display_frame(self):
        if self.current_raw_frame is None:
            self.video_label.setText("No frame loaded")
            return
        
        preview_frame = self.apply_preview_effects(self.current_raw_frame)

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
        # i honestly have not clue how this works, Chat generated it for me
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
            
            if self.current_frame_index < self.end_frame - 1:
                self.next_frame()
                
        except ValueError:
            QMessageBox.warning(None, "Error", "Invalid number entered")
    
    def save_to_csv(self):
        with open(self.csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frame_num", "weight"])
            
            for i in range(self.total_frames):
                value = self.frame_data.get(i, "0")
                writer.writerow([i, value])

    def load_csv_data(self):
        data = {i: "0" for i in range (self.total_frames)}
        try:
            with open(self.csv_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    frame_num = int(row.get('frame_num', -1))
                    weight = row.get('weight', "0")
                    if 0 <= frame_num < self.total_frames:
                        data[frame_num] = weight
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
            self.current_frame_index += 1
            self.load_frame_from_video()

    def previous_frame(self):
        if self.current_frame_index > self.start_frame:
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