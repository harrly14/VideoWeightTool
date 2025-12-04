import os
import cv2
import csv
import sys
from pathlib import Path
from typing import Optional
from Window import EditWindow
from FrameSampler import FrameSampler
from PyQt5.QtWidgets import QApplication, QMessageBox


class BatchManager:
    """Manage batch processing of multiple videos for labelling."""
    
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV'}
    
    def __init__(self, video_dir: str, csv_path: str, target_labels: int):
        """
        Initialize batch manager.
        
        Args:
            video_dir: Directory containing video files
            csv_path: Path to output CSV file
            target_labels: Target number of labels per video (n)
        """
        self.video_dir = video_dir
        self.csv_path = csv_path
        self.target_labels = target_labels
        self.csv_exists = os.path.exists(csv_path)
        self.processed_videos = set()
        
        if self.csv_exists:
            self._load_processed_videos()
    
    def _load_processed_videos(self):
        """Load list of already-processed videos from CSV."""
        try:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                # Skip header if present (check first row)
                first_row = next(reader, None)
                if first_row and first_row[0].lower() in ['filename', 'frame_number']:
                    # This is a header row, skip it
                    for row in reader:
                        if row:
                            self.processed_videos.add(row[0])
                else:
                    # First row is data
                    if first_row and first_row:
                        self.processed_videos.add(first_row[0])
                    for row in reader:
                        if row:
                            self.processed_videos.add(row[0])
        except Exception as e:
            print(f"Warning: Could not load processed videos from CSV: {e}")
    
    def discover_videos(self) -> list[str]:
        """
        Auto-discover video files in directory.
        
        Returns:
            List of video file paths, sorted
        """
        video_files = []
        
        if not os.path.isdir(self.video_dir):
            raise ValueError(f"Video directory does not exist: {self.video_dir}")
        
        for file in os.listdir(self.video_dir):
            if Path(file).suffix in self.VIDEO_EXTENSIONS:
                full_path = os.path.join(self.video_dir, file)
                if os.path.isfile(full_path):
                    video_files.append(full_path)
        
        if not video_files:
            raise ValueError(f"No video files found in {self.video_dir}")
        
        return sorted(video_files)
    
    def get_queue(self) -> list[str]:
        """
        Get videos to process (excluding already-processed ones).
        
        Returns:
            List of video file paths to process
        """
        all_videos = self.discover_videos()
        queue = []
        
        for video_path in all_videos:
            filename = os.path.basename(video_path)
            if filename not in self.processed_videos:
                queue.append(video_path)
        
        return queue
    
    def create_label_window(self, video_path: str) -> tuple[Optional[dict], Optional[dict]]:
        """
        Launch labelling window for a video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            tuple: (frame_data dict, video_params dict) or (None, None) if cancelled
        
        Raises:
            ValueError: If video cannot be opened
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            window = EditWindow(video, video_path, self.target_labels)
            window.show()
            app.exec_()
            
            # Get results from window
            frame_data = window.frame_data
            return frame_data, None
        finally:
            video.release()
    
    def write_results(self, filename: str, frame_data: dict):
        """
        Append labelled frame data to CSV.
        
        Args:
            filename: Video filename (for CSV)
            frame_data: Dict of {frame_num: weight}
        """
        # Determine if we need to write header
        write_header = not self.csv_exists
        
        try:
            mode = 'a' if self.csv_exists else 'w'
            with open(self.csv_path, mode, newline='') as f:
                writer = csv.writer(f)
                
                if write_header:
                    writer.writerow(['filename', 'frame_number', 'weight'])
                
                # Write all labelled frames (non-zero weights)
                for frame_num, weight in sorted(frame_data.items()):
                    if weight != '0':  # Only write non-zero weights
                        writer.writerow([filename, frame_num, weight])
            
            self.csv_exists = True
        except Exception as e:
            raise ValueError(f"Failed to write results to CSV: {e}")
    
    def process_batch(self, video_queue: list[str]) -> int:
        """
        Process all videos in queue with per-video summaries.
        
        Args:
            video_queue: List of video file paths to process
        
        Returns:
            Number of successfully processed videos
        """
        total = len(video_queue)
        processed_count = 0
        
        for idx, video_path in enumerate(video_queue, 1):
            filename = os.path.basename(video_path)
            
            try:
                print(f"[{idx}/{total}] Processing: {filename}")
                
                # Open video and launch labelling window
                frame_data, _ = self.create_label_window(video_path)
                
                if frame_data is None:
                    print(f"Labelling cancelled for {filename}")
                    continue
                
                # Count labelled frames
                labelled_count = sum(1 for w in frame_data.values() if w != '0')
                
                # Show summary
                summary_msg = f"Video {idx}/{total}: {filename}\n\nLabelled {labelled_count} frames."
                
                # Ask user to continue or review
                reply = QMessageBox.question(
                    None,
                    "Video Complete",
                    summary_msg + "\n\nContinue to next video?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.No:
                    # User wants to review/edit current video
                    # Re-open window for this video
                    frame_data, _ = self.create_label_window(video_path)
                    if frame_data is None:
                        continue
                
                # Write results
                self.write_results(filename, frame_data)
                processed_count += 1
                print(f"✓ Completed: {filename}")
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
                QMessageBox.warning(
                    None,
                    "Error",
                    f"Error processing {filename}:\n{e}"
                )
                continue
        
        return processed_count
