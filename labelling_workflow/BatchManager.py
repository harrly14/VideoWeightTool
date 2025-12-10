import os
import cv2
import csv
import sys
from pathlib import Path
from typing import Optional
from RangeSelectionDialog import RangeSelectionDialog
from LabellingWindow import LabellingWindow, VideoCompleteDialog
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
            target_labels: Target number of labels per video
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
    
    def validate_all_videos(self, video_paths: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Pre-validate all videos can be opened.
        
        Args:
            video_paths: List of video file paths to validate
        
        Returns:
            tuple: (valid_paths, errors) where errors is list of (filename, error_msg)
        """
        valid = []
        errors = []
        
        for path in video_paths:
            filename = os.path.basename(path)
            video = cv2.VideoCapture(path)
            
            if video.isOpened():
                # check we can read at least one frame
                success, _ = video.read()
                video.release()
                
                if success:
                    valid.append(path)
                else:
                    errors.append((filename, "Cannot read frames from video"))
            else:
                # Try to get more diagnostic info
                if not os.path.exists(path):
                    errors.append((filename, "File does not exist"))
                else:
                    file_size = os.path.getsize(path)
                    if file_size == 0:
                        errors.append((filename, "File is empty"))
                    else:
                        errors.append((filename, f"Cannot open video (size: {file_size} bytes)"))
        
        return valid, errors
    
    def process_single_video(self, video_path: str, video_index: int, total_videos: int) -> tuple[Optional[dict], int, Optional[int], Optional[int]]:
        """
        Process a single video through staged workflow.
        
        Args:
            video_path: Path to video file
            video_index: Current video index (1-based)
            total_videos: Total videos in batch
        
        Returns:
            tuple: (frame_data dict or None, result_action, start_frame, end_frame)
                result_action: VideoCompleteDialog.EXIT_BATCH, REVIEW_LABELS, or NEXT_VIDEO
        """
        filename = os.path.basename(video_path)
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            # Stage 1: Range Selection Dialog
            range_dialog = RangeSelectionDialog(video, filename)
            if range_dialog.exec_() != RangeSelectionDialog.Accepted:
                # User cancelled range selection
                return None, VideoCompleteDialog.EXIT_BATCH, None, None
            
            start_frame, end_frame = range_dialog.get_range()
            
            # Stage 2: Labelling Window
            while True:
                # Reset video position
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                labelling_window = LabellingWindow(
                    video, filename, start_frame, end_frame,
                    self.target_labels, video_index, total_videos
                )
                labelling_window.show()
                app.exec_()
                
                frame_data = labelling_window.get_frame_data()
                result_action = labelling_window.get_result_action()
                
                if result_action == VideoCompleteDialog.REVIEW_LABELS:
                    # User wants to review - loop back to labelling window
                    continue
                else:
                    # EXIT_BATCH or NEXT_VIDEO
                    return frame_data, result_action, start_frame, end_frame
        
        finally:
            video.release()
    
    def write_results(self, filename: str, frame_data: dict, start_frame: int | None = None, end_frame: int | None = None):
        """
        Append labelled frame data to CSV.
        
        Args:
            filename: Video filename (for CSV)
            frame_data: Dict of {frame_num: weight}
            start_frame: Valid range start frame
            end_frame: Valid range end frame
        """
        write_header = not self.csv_exists
        
        try:
            mode = 'a' if self.csv_exists else 'w'
            with open(self.csv_path, mode, newline='') as f:
                writer = csv.writer(f)
                
                if write_header:
                    writer.writerow(['filename', 'frame_number', 'weight', 'start_frame', 'end_frame'])
                
                for frame_num, weight in sorted(frame_data.items()):
                    if weight != '0':
                        writer.writerow([filename, frame_num, weight, start_frame, end_frame])
            
            self.csv_exists = True
        except Exception as e:
            raise ValueError(f"Failed to write results to CSV: {e}")
    
    def process_batch(self, video_queue: list[str]) -> int:
        """
        Process all videos in queue using staged workflow.
        
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
                
                frame_data, result_action, start_frame, end_frame = self.process_single_video(video_path, idx, total)
                
                if frame_data is None or not frame_data:
                    print(f"No labels for {filename}")
                    if result_action == VideoCompleteDialog.EXIT_BATCH:
                        print("User exited batch")
                        break
                    continue
                
                self.write_results(filename, frame_data, start_frame, end_frame)
                processed_count += 1
                
                labelled_count = len(frame_data)
                print(f"✓ Completed: {filename} ({labelled_count} labels)")
                
                if result_action == VideoCompleteDialog.EXIT_BATCH:
                    print("User exited batch")
                    break
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {e}")
                QMessageBox.warning(
                    None,
                    "Error",
                    f"Error processing {filename}:\n{e}"
                )
                continue
        
        return processed_count
