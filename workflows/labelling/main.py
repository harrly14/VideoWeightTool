#!/usr/bin/env python3

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # allows modules from other directories
import argparse
import cv2
from PyQt5.QtWidgets import QApplication, QMessageBox
from StartupDialog import StartupDialog
from BatchManager import BatchManager


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Video frame labelling tool with dynamic touchstone sampling"
    )
    parser.add_argument("-n", "--num-frames-per-video", type=int, help="Number of labels per video")
    parser.add_argument("--video-dir", type=str, help="Directory containing videos")
    parser.add_argument("--csv", type=str, help="Path to output CSV file")
    parser.add_argument(
        "--strict-validation", 
        action="store_true",
        help="Exit immediately if any video cannot be opened (default: warn and allow continue)"
    )
    parser.add_argument(
        "--rois-only",
        action="store_true",
        help="Only perform ROI and valid-frame selection; do not create CSVs or add touchstones (saves JSON only)"
    )
    
    args = parser.parse_args()
    return args


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    
    cli_args = parse_cli_args()

    strict_validation = cli_args.strict_validation if hasattr(cli_args, 'strict_validation') else False
    rois_only = cli_args.rois_only if hasattr(cli_args, 'rois_only') else False
    
    if rois_only:
        if cli_args.video_dir:
            video_dir = cli_args.video_dir
            num_frames_per_video = cli_args.num_frames_per_video if hasattr(cli_args, 'num_frames_per_video') else None
            csv_path = None
            print(f"Using CLI parameters in ROIS-only mode: num_frames_per_video={num_frames_per_video}, video_dir={video_dir}")
        else:
            QMessageBox.critical(
                None,
                "Error",
                "--rois-only requires --video-dir when running without the GUI"
            )
            sys.exit(1)
    else:
        if cli_args.num_frames_per_video and cli_args.video_dir and cli_args.csv:
            num_frames_per_video = cli_args.num_frames_per_video
            video_dir = cli_args.video_dir
            csv_path = cli_args.csv
            
            if not csv_path.lower().endswith('.csv'):
                csv_path += '.csv'
            
            print(f"Using CLI parameters: num_frames_per_video={num_frames_per_video}, video_dir={video_dir}, csv={csv_path}")
        else:
            dialog = StartupDialog()
            if dialog.exec_() != StartupDialog.Accepted:
                sys.exit(0)
            
            num_frames_per_video, video_dir, csv_path = dialog.get_config()
    
    if not video_dir or (not csv_path and not rois_only):
        QMessageBox.critical(None, "Error", "Invalid configuration")
        sys.exit(1)
    
    if not os.path.isdir(video_dir):
        QMessageBox.critical(None, "Error", f"Video directory does not exist: {video_dir}")
        sys.exit(1)
    
    try:
        manager = BatchManager(video_dir, csv_path, num_frames_per_video, rois_only=rois_only)
        queue = manager.get_videos_to_process()
        
        if not queue:
            QMessageBox.information(
                None,
                "No videos to process",
                f"All videos in {video_dir} have already been processed, or no videos found."
            )
            sys.exit(0)
        
        print(f"Found {len(queue)} videos to process")
        
        print("Validating videos...")
        valid_queue, errors = manager.validate_all_videos(queue)
        
        if errors:
            formatted_lines = [
                f"  • {fname}: {msg}" 
                for fname, msg in errors[:10]
            ]
            error_list = "\n".join(formatted_lines)

            if len(errors) > 10:
                error_list += f"\n  ... and {len(errors) - 10} more"
            
            if strict_validation:
                QMessageBox.critical(
                    None,
                    "Video Validation Failed",
                    f"The following videos cannot be opened:\n\n{error_list}\n\n"
                    "Exiting due to --strict-validation flag."
                )
                sys.exit(1)
            else:
                reply = QMessageBox.warning(
                    None,
                    "Video Validation Warnings",
                    f"The following videos cannot be opened and will be skipped:\n\n{error_list}\n\n"
                    f"{len(valid_queue)} videos are valid.\n\n"
                    "Continue with valid videos?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.No:
                    sys.exit(0)
                
                queue = valid_queue
        
        if not queue:
            QMessageBox.information(
                None,
                "No valid videos",
                "No valid videos to process after validation."
            )
            sys.exit(0)
        
        print(f"{len(queue)} valid videos ready to process")
        
        header = (
            f"Ready to process {len(queue)} video(s) (ROIs-only mode).\n\n" if rois_only else f"Ready to process {len(queue)} video(s) with {num_frames_per_video} labels per video.\n\n"
        )
        
        reply = QMessageBox.information(
            None,
            "Confirm",
            header
            + (f"Video sections will be saved to:\ndata/valid_video_sections.json\n\n" if rois_only else f"Output will be saved to:\n{csv_path}\n\n")
            + "Continue?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok
        )
        
        if reply == QMessageBox.Cancel:
            sys.exit(0)
        
        processed = manager.process_batch(queue)
        
        if rois_only:
            success_message = f"Successfully processed {processed} video(s).\n\nVideo sections saved to:\ndata/valid_video_sections.json"
        else:
            success_message = f"Successfully processed {processed} video(s).\n\nResults saved to:\n{csv_path}"
        
        QMessageBox.information(None, "Batch Complete", success_message)
        
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Batch processing failed:\n{e}")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
