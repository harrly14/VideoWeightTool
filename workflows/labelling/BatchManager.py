import os
import cv2
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from VideoSectionsManager import VideoSectionsManager
from LabellingWindow import LabellingWindow, VideoCompleteDialog
from PyQt5.QtWidgets import QApplication, QMessageBox

VIDEO_SECTIONS_JSON_PATH = Path("data/valid_video_sections.json")


class BatchManager:
    """Manage batch processing of multiple videos for labelling."""
    
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV'}
    
    def __init__(self, video_dir: str, csv_path: str | None, target_labels: int | None, rois_only: bool = False):
        """
        Initialize batch manager.
        
        Args:
            video_dir: Directory containing video files
            csv_path: Path to output CSV file (optional if rois-only)
            target_labels: Target number of labels per video (None in rois-only mode)
            rois_only: If True, skip CSV I/O and only perform ROI/valid-frame selection
        """
        self.video_dir = video_dir
        self.csv_path = csv_path
        self.target_labels = target_labels
        self.rois_only = rois_only
        self.processed_videos = set()
        
        self.video_sections: dict = {}
        self.last_roi: Optional[List[Tuple[int, int]]] = None  # For ROI carryover
        
        self._load_video_sections()
        
        if not self.rois_only and self.csv_path:
            if os.path.exists(self.csv_path):
                self._load_processed_videos()
    
    def _load_video_sections(self):
        if VIDEO_SECTIONS_JSON_PATH.exists():
            try:
                with open(VIDEO_SECTIONS_JSON_PATH, 'r') as f:
                    data = json.load(f)
                self.video_sections = data
                print(f"Loaded existing video sections from {VIDEO_SECTIONS_JSON_PATH}")
            except Exception as e:
                print(f"Warning: Could not load video sections: {e}")
                self.video_sections = {}
        else:
            self.video_sections = {}
    
    def _save_video_sections(self):
        # Ensure data directory exists
        VIDEO_SECTIONS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        self.video_sections["updated_at"] = datetime.now(timezone.utc).isoformat() + "Z"
        if "created_at" not in self.video_sections:
            self.video_sections["created_at"] = self.video_sections["updated_at"]
        
        if "video_ranges" not in self.video_sections:
            self.video_sections["video_ranges"] = {}
        if "rois" not in self.video_sections:
            self.video_sections["rois"] = {}
        
        self.video_sections["num_videos"] = len(self.video_sections.get("video_ranges", {}))
        
        try:
            with open(VIDEO_SECTIONS_JSON_PATH, 'w') as f:
                json.dump(self.video_sections, f, indent=2, sort_keys=True)
            print(f"Saved video sections to {VIDEO_SECTIONS_JSON_PATH}")
        except Exception as e:
            print(f"Warning: Could not save video sections: {e}")
    
    def _update_video_sections(self, filename: str, sections: List[Dict]):
        if "video_ranges" not in self.video_sections:
            self.video_sections["video_ranges"] = {}
        if "rois" not in self.video_sections:
            self.video_sections["rois"] = {}
        
        # Calculate overall range from sections
        if sections:
            overall_start = min(s['start_frame'] for s in sections)
            overall_end = max(s['end_frame'] for s in sections)
            
            self.video_sections["video_ranges"][filename] = {
                "start_frame": overall_start,
                "end_frame": overall_end
            }
            
            self.video_sections["rois"][filename] = {
                "sections": [
                    {
                        "quad": s['quad'],
                        "start_frame": s['start_frame'],
                        "end_frame": s['end_frame']
                    }
                    for s in sections
                ]
            }
        
        self._save_video_sections()
    
    def _load_processed_videos(self):
        try:
            csv_path = self.csv_path
            if csv_path is None:
                return
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                first_row = next(reader, None)
                if first_row and first_row[0].lower() in ['filename', 'frame_number']:
                    for row in reader:
                        if row:
                            self.processed_videos.add(row[0])
                else:
                    if first_row and first_row:
                        self.processed_videos.add(first_row[0])
                    for row in reader:
                        if row:
                            self.processed_videos.add(row[0])
        except Exception as e:
            print(f"Warning: Could not load processed videos from CSV: {e}")
    
    def discover_videos(self) -> list[str]:
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
    
    def get_videos_to_process(self) -> list[str]:
        all_videos = self.discover_videos()
        
        csv_processed = set(self.processed_videos)
        json_processed = set(self.video_sections.get("video_ranges", {}).keys())
        already_processed = csv_processed | json_processed
        
        queue = [
            video_path 
            for video_path in all_videos 
            if os.path.basename(video_path) not in already_processed
        ]
        
        return queue
    
    def validate_all_videos(self, video_paths: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
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
    
    def process_single_video(self, video_path: str, video_index: int, total_videos: int) -> tuple[Optional[dict], int, Optional[List[Dict]], Optional[List[Tuple[int, int]]]]:
        """
        Process a single video through staged workflow.
        
        Args:
            video_path: Path to video file
            video_index: Current video index (1-based)
            total_videos: Total videos in batch
        
        Returns:
            tuple: (frame_data dict or None, result_action, sections list, covered_ranges)
                result_action: VideoCompleteDialog.EXIT_BATCH, REVIEW_LABELS, or NEXT_VIDEO
                sections: List of section dicts with 'quad', 'start_frame', 'end_frame'
                covered_ranges: List of (start, end) tuples for frame sampling
        """
        filename = os.path.basename(video_path)
        
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Stage 1: Video Sections Manager (combined ROI + frame range selection)
            sections_manager = VideoSectionsManager(
                video, filename, total_frames,
                default_roi=self.last_roi  # Carryover from previous video
            )
            
            if sections_manager.exec_() != VideoSectionsManager.Accepted: # User cancelled section selection
                return None, VideoCompleteDialog.EXIT_BATCH, None, None
            
            sections = sections_manager.get_sections()
            covered_ranges = sections_manager.get_covered_frame_ranges()
            self.last_roi = sections_manager.get_last_roi()  # Save for carryover to next video
            
            if not sections or not covered_ranges:
                return None, VideoCompleteDialog.EXIT_BATCH, None, None
            
            # If running in rois-only mode, skip the labelling window
            if self.rois_only:
                try:
                    self._update_video_sections(filename, sections)
                    print(f"ROIS-only: saved sections for {filename}")
                    return None, VideoCompleteDialog.NEXT_VIDEO, sections, covered_ranges
                except Exception as e:
                    print(f"Error saving sections for {filename}: {e}")
                    return None, VideoCompleteDialog.NEXT_VIDEO, None, None
            
            overall_start = min(range[0] for range in covered_ranges)
            overall_end = max(range[1] for range in covered_ranges)

            if self.target_labels is None:
                raise ValueError("Target labels must be set for labelling")
            
            # Stage 2: Labelling Window (only samples from covered ranges)
            while True:
                # Reset video position
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                labelling_window = LabellingWindow(
                    video, filename, overall_start, overall_end,
                    self.target_labels, video_index, total_videos,
                    covered_ranges=covered_ranges
                )
                labelling_window.show()
                app.exec_()
                
                frame_data = labelling_window.get_frame_data()
                result_action = labelling_window.get_result_action()
                
                if result_action == VideoCompleteDialog.REVIEW_LABELS:
                    continue
                else:
                    # EXIT_BATCH or NEXT_VIDEO
                    return frame_data, result_action, sections, covered_ranges
        
        finally:
            video.release()
    
    def write_results(self, filename: str, frame_data: dict):
        if self.rois_only:
            # In rois-only mode we do not create or append to CSVs
            print(f"ROIS-only mode active: skipping CSV write for {filename}")
            return

        
        try:
            if self.csv_path is None:
                raise ValueError("CSV path is not configured")

            file_exists = os.path.exists(self.csv_path)
            mode = 'a' if file_exists else 'w'
            
            with open(self.csv_path, mode, newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow(['filename', 'frame_number', 'weight'])
                
                for frame_num, weight in sorted(frame_data.items()):
                    if weight != '0':
                        writer.writerow([filename, frame_num, weight])
            
        except Exception as e:
            raise ValueError(f"Failed to write results to CSV: {e}")
    
    def process_batch(self, video_queue: list[str]) -> int:
        total = len(video_queue)
        processed_count = 0
        
        for idx, video_path in enumerate(video_queue, 1):
            filename = os.path.basename(video_path)
            
            try:
                print(f"[{idx}/{total}] Processing: {filename}")
                
                frame_data, result_action, sections, covered_ranges = self.process_single_video(video_path, idx, total)
                
                # ROIs-only mode: sections are saved inside process_single_video; just count and report
                if self.rois_only:
                    if sections:
                        processed_count += 1
                        print(f"Completed: {filename} (ROIs saved, {len(sections)} ROI sections)")
                    else:
                        print(f"No ROIs saved for {filename}")
                    if result_action == VideoCompleteDialog.EXIT_BATCH:
                        print("User exited batch")
                        break
                    continue
                
                if frame_data is None or not frame_data:
                    print(f"No labels for {filename}")
                    if result_action == VideoCompleteDialog.EXIT_BATCH:
                        print("User exited batch")
                        break
                    continue
                
                self.write_results(filename, frame_data)
                
                if sections:
                    self._update_video_sections(filename, sections)
                
                processed_count += 1
                
                labelled_count = len(frame_data)
                print(f"Completed: {filename} ({labelled_count} labels, {len(sections) if sections is not None else 0} ROI sections)")
                
                if result_action == VideoCompleteDialog.EXIT_BATCH:
                    print("User exited batch")
                    break
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                QMessageBox.warning(
                    None,
                    "Error",
                    f"Error processing {filename}:\n{e}"
                )
                continue
        
        return processed_count