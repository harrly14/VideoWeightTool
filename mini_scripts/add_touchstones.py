"""
Touchstone Labeling Tool

Creates ground-truth labels for every Nth frame in a video.
These touchstones can be used with process_video.py --ground-truth
to validate and correct CNN predictions.

Usage:
    Basic (selevt video via dialog): python add_touchstones.py
    With options:python add_touchstones.py --video video.mp4 --interval 50 --roi 100,50,400,150
    Output with CNN processing: python process_video.py --video video.mp4 --ground-truth video_touchstones.csv
"""

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)
import cv2
import csv
import sys
import argparse
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, 
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QRegExpValidator
from PyQt5.QtCore import Qt, QRegExp


class TouchstoneLabeler(QWidget):
    def __init__(self, video_path, interval=100, output_path=None, roi=None):
        super().__init__()
        self.video_path = video_path
        self.interval = interval
        self.roi = self._parse_roi(roi) if roi else None
        
        # Setup video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate touchstone frames
        self.touchstone_frames = list(range(0, self.total_frames, interval))
        self.current_idx = 0
        
        # Labels storage: {frame_num: weight}
        self.labels = {}
        
        # Output path
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}_touchstones.csv"
        self.output_path = output_path
        
        # Load existing labels if file exists
        self._load_existing_labels()
        
        self.setup_ui()
        self.load_current_frame()
    
    def _parse_roi(self, roi_str):
        """Parse ROI string 'x,y,w,h' to tuple"""
        try:
            return tuple(map(int, roi_str.split(',')))
        except:
            return None
    
    def _load_existing_labels(self):
        """Load existing touchstone labels if file exists"""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        frame_num = int(row['frame_num'])
                        weight = row['weight']
                        if weight and weight != '0':
                            self.labels[frame_num] = weight
                print(f"Loaded {len(self.labels)} existing labels from {self.output_path}")
                
                # Skip to first unlabeled frame
                for i, frame_num in enumerate(self.touchstone_frames):
                    if frame_num not in self.labels:
                        self.current_idx = i
                        break
                else:
                    self.current_idx = len(self.touchstone_frames) - 1
            except Exception as e:
                print(f"Warning: Could not load existing labels: {e}")
    
    def setup_ui(self):
        self.setWindowTitle("Touchstone Labeler")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Info label
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        layout.addWidget(self.video_label)
        
        # ROI display (if ROI specified)
        if self.roi:
            self.roi_label = QLabel()
            self.roi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.roi_label)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, len(self.touchstone_frames))
        layout.addWidget(self.progress)
        
        # Weight entry
        entry_layout = QHBoxLayout()
        entry_layout.addWidget(QLabel("Weight (kg):"))
        
        self.weight_entry = QLineEdit()
        self.weight_entry.setPlaceholderText("e.g., 7.535")
        pattern = QRegExp(r'^[0-9.]*$')
        self.weight_entry.setValidator(QRegExpValidator(pattern))
        self.weight_entry.returnPressed.connect(self.save_and_next)
        entry_layout.addWidget(self.weight_entry)
        
        layout.addLayout(entry_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("← Previous (A)")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setShortcut("A")
        nav_layout.addWidget(self.prev_btn)
        
        self.skip_btn = QPushButton("Skip (S)")
        self.skip_btn.clicked.connect(self.skip_frame)
        self.skip_btn.setShortcut("S")
        nav_layout.addWidget(self.skip_btn)
        
        self.save_btn = QPushButton("Save & Next (Enter/D)")
        self.save_btn.clicked.connect(self.save_and_next)
        self.save_btn.setShortcut("D")
        nav_layout.addWidget(self.save_btn)
        
        layout.addLayout(nav_layout)
        
        # Save and quit button
        bottom_layout = QHBoxLayout()
        
        self.save_quit_btn = QPushButton("Save All & Quit (Q)")
        self.save_quit_btn.clicked.connect(self.save_and_quit)
        self.save_quit_btn.setShortcut("Q")
        bottom_layout.addWidget(self.save_quit_btn)
        
        self.jump_btn = QPushButton("Jump to Frame...")
        self.jump_btn.clicked.connect(self.jump_to_frame)
        bottom_layout.addWidget(self.jump_btn)
        
        layout.addLayout(bottom_layout)
        
        self.setLayout(layout)
        
        # Focus on weight entry
        self.weight_entry.setFocus()
    
    def load_current_frame(self):
        """Load and display the current touchstone frame"""
        if self.current_idx >= len(self.touchstone_frames):
            self.show_complete_dialog()
            return
        
        frame_num = self.touchstone_frames[self.current_idx]
        
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            QMessageBox.warning(self, "Error", f"Could not read frame {frame_num}")
            return
        
        # Update info
        timestamp = frame_num / self.fps if self.fps > 0 else 0
        labeled_count = len(self.labels)
        self.info_label.setText(
            f"Frame {frame_num}/{self.total_frames} | "
            f"Time: {timestamp:.1f}s | "
            f"Touchstone {self.current_idx + 1}/{len(self.touchstone_frames)} | "
            f"Labeled: {labeled_count}"
        )
        
        # Update progress
        self.progress.setValue(self.current_idx + 1)
        
        # Display full frame
        self.display_frame(frame, self.video_label)
        
        # Display ROI crop if specified
        if self.roi:
            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w]
            self.display_frame(roi_frame, self.roi_label, max_width=400)
        
        # Pre-fill weight if already labeled
        if frame_num in self.labels:
            self.weight_entry.setText(self.labels[frame_num])
        else:
            self.weight_entry.clear()
        
        self.weight_entry.setFocus()
        self.weight_entry.selectAll()
    
    def display_frame(self, frame, label, max_width=None):
        """Convert CV2 frame to QPixmap and display"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit label
        if max_width:
            pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
        else:
            pixmap = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        
        label.setPixmap(pixmap)
    
    def save_current_label(self):
        """Save the current weight entry"""
        weight = self.weight_entry.text().strip()
        if weight:
            frame_num = self.touchstone_frames[self.current_idx]
            self.labels[frame_num] = weight
            return True
        return False
    
    def save_and_next(self):
        """Save current label and move to next frame"""
        if self.save_current_label():
            self.current_idx += 1
            self.load_current_frame()
            self.save_to_csv()  # Auto-save after each label
        else:
            QMessageBox.warning(self, "No Weight", "Please enter a weight value")
    
    def skip_frame(self):
        """Skip current frame without labeling"""
        self.current_idx += 1
        self.load_current_frame()
    
    def prev_frame(self):
        """Go to previous touchstone frame"""
        if self.current_idx > 0:
            self.save_current_label()  # Save if there's a value
            self.current_idx -= 1
            self.load_current_frame()
    
    def jump_to_frame(self):
        """Jump to a specific touchstone index"""
        from PyQt5.QtWidgets import QInputDialog
        idx, ok = QInputDialog.getInt(
            self, "Jump to Touchstone", 
            f"Enter touchstone number (1-{len(self.touchstone_frames)}):",
            self.current_idx + 1, 1, len(self.touchstone_frames)
        )
        if ok:
            self.save_current_label()
            self.current_idx = idx - 1
            self.load_current_frame()
    
    def save_to_csv(self):
        """Save all labels to CSV"""
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_num', 'timestamp', 'weight'])
            
            for frame_num in sorted(self.labels.keys()):
                timestamp = frame_num / self.fps if self.fps > 0 else 0
                writer.writerow([frame_num, f"{timestamp:.3f}", self.labels[frame_num]])
        
        # print(f"Saved {len(self.labels)} labels to {self.output_path}")
    
    def save_and_quit(self):
        """Save all labels and close"""
        self.save_current_label()
        self.save_to_csv()
        
        QMessageBox.information(
            self, "Saved", 
            f"Saved {len(self.labels)} touchstone labels to:\n{self.output_path}\n\n"
            f"Use with process_video.py:\n"
            f"  --ground-truth {self.output_path}"
        )
        self.close()
    
    def show_complete_dialog(self):
        """Show completion dialog"""
        self.save_to_csv()
        QMessageBox.information(
            self, "Complete",
            f"All {len(self.touchstone_frames)} touchstone frames reviewed!\n"
            f"Labeled: {len(self.labels)} frames\n\n"
            f"Saved to: {self.output_path}"
        )
        self.close()
    
    def closeEvent(self, event):
        """Handle window close"""
        self.cap.release()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Label touchstone frames in a video")
    parser.add_argument('--video', help='Path to video file (or select via dialog)')
    parser.add_argument('--interval', type=int, default=100, 
                       help='Frame interval for touchstones (default: 100)')
    parser.add_argument('--output', help='Output CSV path (default: video_touchstones.csv)')
    parser.add_argument('--roi', help='ROI as x,y,w,h to show cropped region')
    args = parser.parse_args()
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Get video path
    video_path = args.video
    if not video_path:
        video_filter = "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV);;All Files (*)"
        video_path, _ = QFileDialog.getOpenFileName(
            None, "Select Video File", os.getcwd(), video_filter
        )
    
    if not video_path:
        print("No video selected. Exiting.")
        sys.exit(0)
    
    try:
        labeler = TouchstoneLabeler(
            video_path, 
            interval=args.interval,
            output_path=args.output,
            roi=args.roi
        )
        labeler.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()