#!/usr/bin/env python3

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import sys
import argparse
import cv2
from PyQt5.QtWidgets import QApplication, QMessageBox
from StartupDialog import StartupDialog
from BatchManager import BatchManager


def parse_cli_args():
    """Parse command-line arguments."""
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
    
    args = parser.parse_args()
    return args


def main():
    """Main entry point."""
    app = QApplication.instance() or QApplication(sys.argv)
    
    cli_args = parse_cli_args()
    
    
    strict_validation = cli_args.strict_validation if hasattr(cli_args, 'strict_validation') else False
    
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
    
    if not num_frames_per_video or not video_dir or not csv_path:
        QMessageBox.critical(None, "Error", "Invalid configuration")
        sys.exit(1)
    
    if not os.path.isdir(video_dir):
        QMessageBox.critical(None, "Error", f"Video directory does not exist: {video_dir}")
        sys.exit(1)
    
    try:
        manager = BatchManager(video_dir, csv_path, num_frames_per_video)
        
        queue = manager.get_queue()
        
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
            error_list = "\n".join([f"  • {fname}: {msg}" for fname, msg in errors[:10]])
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
        
        reply = QMessageBox.information(
            None,
            "Confirm",
            f"Ready to process {len(queue)} video(s) with {num_frames_per_video} labels per video.\n\n"
            f"Output will be saved to:\n{csv_path}\n\n"
            "Continue?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok
        )
        
        if reply == QMessageBox.Cancel:
            sys.exit(0)
        
        processed = manager.process_batch(queue)
        
        QMessageBox.information(
            None,
            "Batch Complete",
            f"Successfully processed {processed} video(s).\n\n"
            f"Results saved to:\n{csv_path}"
        )
        
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Batch processing failed:\n{e}")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
