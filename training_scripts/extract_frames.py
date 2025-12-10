import os
import sys
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, 
                             QGroupBox, QMessageBox, QAction, QProgressDialog)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

# Add project root to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CNN_WIDTH = 256
CNN_HEIGHT = 64
CNN_ASPECT_RATIO = CNN_WIDTH / CNN_HEIGHT
MIN_HEIGHT = 32

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
        self.crop_rect = None
        self.original_w = 0
        self.original_h = 0
        self.aspect_ratio = CNN_ASPECT_RATIO
        self.setMouseTracking(True)
        
        self.action = None
        self.drag_start_pos = None
        self.initial_crop_rect = None
        
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.pan_dragging = False
        self.pan_drag_start = None
        self.pan_drag_start_offset = None

        self.std_x = 625
        self.std_y = 300
        self.std_w = 893
        self.std_h = 223
    
    def set_standard_roi(self, x, y, w, h):
        self.std_x = x
        self.std_y = y
        self.std_w = w
        self.std_h = h
    
    def set_original_size(self, w, h):
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
    
    def set_zoom(self, zoom_level, pan_x=None, pan_y=None):
        if not self.original_w or not self.original_h:
            return
        self.zoom_level = max(1.0, min(zoom_level, 10.0))
        
        if pan_x is not None:
            max_pan_x = max(0, self.original_w - self.original_w / self.zoom_level)
            self.pan_offset_x = max(0, min(pan_x, max_pan_x))
        
        if pan_y is not None:
            max_pan_y = max(0, self.original_h - self.original_h / self.zoom_level)
            self.pan_offset_y = max(0, min(pan_y, max_pan_y))

    def _get_display_params(self):
        if not self.original_w or not self.original_h:
            return None
        label_w = self.width()
        label_h = self.height()
        
        visible_w = self.original_w / self.zoom_level
        visible_h = self.original_h / self.zoom_level
        
        scale = min(label_w / visible_w, label_h / visible_h)
        display_w = int(visible_w * scale)
        display_h = int(visible_h * scale)
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2
        return scale, offset_x, offset_y, visible_w, visible_h

    def _image_to_label(self, img_rect):
        params = self._get_display_params()
        if not params: return None
        scale, off_x, off_y, _, _ = params
        x, y, w, h = img_rect
        
        x_rel = x - self.pan_offset_x
        y_rel = y - self.pan_offset_y
        
        lx = int(x_rel * scale) + off_x
        ly = int(y_rel * scale) + off_y
        lw = int(w * scale)
        lh = int(h * scale)
        return (lx, ly, lw, lh)

    def _label_to_image_pt(self, pos):
        params = self._get_display_params()
        if not params: return None, None
        scale, off_x, off_y, _, _ = params
        
        ix_rel = int((pos.x() - off_x) / scale)
        iy_rel = int((pos.y() - off_y) / scale)
        
        ix = ix_rel + int(self.pan_offset_x)
        iy = iy_rel + int(self.pan_offset_y)
        return ix, iy

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        if not self.crop_rect: return
        
        l_rect = self._image_to_label(self.crop_rect)
        if not l_rect:
            if self.zoom_level > 1.0:
                self.pan_dragging = True
                self.pan_drag_start = event.pos()
                self.pan_drag_start_offset = (self.pan_offset_x, self.pan_offset_y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        
        lx, ly, lw, lh = l_rect
        x, y = event.pos().x(), event.pos().y()
        handle_r = 10
        
        if abs(x - lx) < handle_r and abs(y - ly) < handle_r: self.action = 'resize_tl'
        elif abs(x - (lx + lw)) < handle_r and abs(y - ly) < handle_r: self.action = 'resize_tr'
        elif abs(x - lx) < handle_r and abs(y - (ly + lh)) < handle_r: self.action = 'resize_bl'
        elif abs(x - (lx + lw)) < handle_r and abs(y - (ly + lh)) < handle_r: self.action = 'resize_br'
        elif lx < x < lx + lw and ly < y < ly + lh: self.action = 'move'
        else:
            if self.zoom_level > 1.0:
                self.pan_dragging = True
                self.pan_drag_start = event.pos()
                self.pan_drag_start_offset = (self.pan_offset_x, self.pan_offset_y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.action = None
            return
            
        self.drag_start_pos = event.pos()
        self.initial_crop_rect = self.crop_rect

    def mouseMoveEvent(self, event):
        if not self.original_w: return
        
        if self.pan_dragging and self.pan_drag_start:
            dx_screen = event.pos().x() - self.pan_drag_start.x()
            dy_screen = event.pos().y() - self.pan_drag_start.y()
            params = self._get_display_params()
            if params:
                scale = params[0]
                dx_img = -dx_screen / scale
                dy_img = -dy_screen / scale
                
                new_pan_x = self.pan_drag_start_offset[0] + dx_img
                new_pan_y = self.pan_drag_start_offset[1] + dy_img
                
                max_pan_x = max(0, self.original_w - self.original_w / self.zoom_level)
                max_pan_y = max(0, self.original_h - self.original_h / self.zoom_level)
                self.pan_offset_x = max(0, min(new_pan_x, max_pan_x))
                self.pan_offset_y = max(0, min(new_pan_y, max_pan_y))
                
                parent = self.parent()
                if parent and hasattr(parent, 'on_pan_drag'):
                    parent.on_pan_drag(self.pan_offset_x, self.pan_offset_y)
                self.update()
            return

        if not self.crop_rect: return
        
        if not self.action: return
        curr_ix, curr_iy = self._label_to_image_pt(event.pos())
        if curr_ix is None: return
        
        ix, iy, iw, ih = self.initial_crop_rect
        
        if self.action == 'move':
            start_ix, start_iy = self._label_to_image_pt(self.drag_start_pos)
            if start_ix is None: return
            dx = curr_ix - start_ix
            dy = curr_iy - start_iy
            nx = max(0, min(ix + dx, self.original_w - iw))
            ny = max(0, min(iy + dy, self.original_h - ih))
            self.crop_rect = (nx, ny, iw, ih)
            
        elif 'resize' in self.action:
            if self.action == 'resize_br':
                w_cand = abs(curr_ix - ix)
                h_cand = int(w_cand / self.aspect_ratio)
                if h_cand < MIN_HEIGHT: h_cand = MIN_HEIGHT; w_cand = int(h_cand * self.aspect_ratio)
                self.crop_rect = (ix, iy, w_cand, h_cand)
            
        self.update()
        self.crop_selected.emit(*self.crop_rect)

    def mouseReleaseEvent(self, event):
        self.pan_dragging = False
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
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawRect(lx, ly, lw, lh)

class ExtractionWindow(QWidget):
    def __init__(self, video_path, frames_to_extract, output_folder, video_name, next_callback, skip_callback):
        super().__init__()
        self.video_path = video_path
        self.frames_to_extract = sorted(list(set(frames_to_extract)))
        self.output_folder = output_folder
        self.video_name = video_name
        self.next_callback = next_callback
        self.skip_callback = skip_callback
        
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {video_path}")
            self.skip_callback()
            return

        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        
        self.setWindowTitle(f"Extracting: {video_name} ({len(self.frames_to_extract)} frames)")
        self.resize(1200, 800)
        
        layout = QHBoxLayout()
        
        video_layout = QVBoxLayout()
        self.video_label = CNNCropLabel()
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)
        
        controls_layout = QVBoxLayout()
        
        info_label = QLabel(f"Video: {video_name}\nFrames to extract: {len(self.frames_to_extract)}")
        controls_layout.addWidget(info_label)
        
        self.preview_btn = QPushButton("Preview Random Frame")
        self.preview_btn.clicked.connect(self.load_random_frame)
        controls_layout.addWidget(self.preview_btn)
        
        self.reset_crop_btn = QPushButton("Reset Crop")
        self.reset_crop_btn.clicked.connect(self.video_label.reset_crop_to_standard)
        controls_layout.addWidget(self.reset_crop_btn)
        
        controls_layout.addStretch()
        
        self.extract_btn = QPushButton("EXTRACT & NEXT VIDEO")
        self.extract_btn.setStyleSheet("background-color: #4a7a4a; color: white; padding: 10px; font-weight: bold;")
        self.extract_btn.clicked.connect(self.extract_frames)
        controls_layout.addWidget(self.extract_btn)
        
        self.skip_btn = QPushButton("SKIP VIDEO")
        self.skip_btn.setStyleSheet("background-color: #7a4a4a; color: white; padding: 5px;")
        self.skip_btn.clicked.connect(self.skip_video)
        controls_layout.addWidget(self.skip_btn)
        
        layout.addLayout(video_layout, 3)
        layout.addLayout(controls_layout, 1)
        self.setLayout(layout)
        
        ret, frame = self.video.read()
        if ret:
            self.video_label.set_original_size(frame.shape[1], frame.shape[0])
            self.load_random_frame()
        
    def load_random_frame(self):
        if not self.frames_to_extract: return
        import random
        frame_num = random.choice(self.frames_to_extract)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.video.read()
        if ret:
            self.current_raw_frame = frame
            self.update_display()
            
    def update_display(self):
        if hasattr(self, 'current_raw_frame'):
            preview = apply_clahe(self.current_raw_frame)
            h, w, ch = preview.shape
            bytes_per_line = ch * w
            qt_img = QImage(preview.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img))
            
    def extract_frames(self):
        if not self.video_label.crop_rect:
            QMessageBox.warning(self, "Warning", "Please define a crop region first.")
            return
            
        crop_x, crop_y, crop_w, crop_h = self.video_label.crop_rect
        
        progress = QProgressDialog("Extracting frames...", "Cancel", 0, len(self.frames_to_extract), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        
        count = 0
        for i, frame_num in enumerate(self.frames_to_extract):
            if progress.wasCanceled():
                break
                
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.video.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_num}")
                continue
                
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            resized = cv2.resize(cropped, (CNN_WIDTH, CNN_HEIGHT), interpolation=cv2.INTER_AREA)
            
            out_name = f"{self.video_name}_{frame_num}.jpg"
            out_path = os.path.join(self.output_folder, out_name)
            cv2.imwrite(out_path, resized)
            
            count += 1
            progress.setValue(i + 1)
            
        progress.close()
        print(f"Extracted {count} frames for {self.video_name}")
        self.video.release()
        self.close()
        self.next_callback()

    def skip_video(self):
        self.video.release()
        self.close()
        self.skip_callback()
        
    def on_pan_drag(self, x, y):
        self.video_label.set_zoom(self.video_label.zoom_level, x, y)


class ExtractionController:
    def __init__(self, csv_path, video_dir, output_dir):
        self.df = pd.read_csv(csv_path)
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.unique_videos = self.df['filename'].unique()
        self.current_video_idx = 0
        
    def start(self):
        self.process_next_video()
        
    def process_next_video(self):
        if self.current_video_idx >= len(self.unique_videos):
            QMessageBox.information(None, "Done", "All videos processed!")
            sys.exit(0)
            
        video_name = self.unique_videos[self.current_video_idx]
        video_path = self.video_dir / video_name
        
        if not video_path.exists():
            print(f"Video not found: {video_path}. Skipping...")
            self.current_video_idx += 1
            self.process_next_video()
            return
            
        frames = self.df[self.df['filename'] == video_name]['frame_number'].tolist()
        
        self.window = ExtractionWindow(
            str(video_path), 
            frames, 
            str(self.output_dir), 
            video_name,
            self.on_video_complete,
            self.on_video_skipped
        )
        self.window.show()
        
    def on_video_complete(self):
        self.current_video_idx += 1
        self.process_next_video()
        
    def on_video_skipped(self):
        print(f"Skipped video: {self.unique_videos[self.current_video_idx]}")
        self.current_video_idx += 1
        self.process_next_video()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/all_data.csv', help='Path to labels CSV')
    parser.add_argument('--video_dir', required=True, help='Directory containing video files')
    parser.add_argument('--output_dir', default='data/images', help='Output directory for images')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    if not os.path.exists(args.csv):
        QMessageBox.critical(None, "Error", f"CSV file not found: {args.csv}")
        return
        
    controller = ExtractionController(args.csv, args.video_dir, args.output_dir)
    controller.start()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
