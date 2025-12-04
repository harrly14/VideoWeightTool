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
    parser.add_argument("--n", type=int, help="Number of labels per video")
    parser.add_argument("--video-dir", type=str, help="Directory containing videos")
    parser.add_argument("--csv", type=str, help="Path to output CSV file")
    
    args = parser.parse_args()
    return args


def main():
    """Main entry point."""
    # Ensure QApplication exists
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Parse CLI arguments
    cli_args = parse_cli_args()
    
    # Get configuration
    if cli_args.n and cli_args.video_dir and cli_args.csv:
        # Use CLI arguments
        n = cli_args.n
        video_dir = cli_args.video_dir
        csv_path = cli_args.csv
        
        print(f"Using CLI parameters: n={n}, video_dir={video_dir}, csv={csv_path}")
    else:
        # Show startup dialog
        dialog = StartupDialog()
        if dialog.exec_() != StartupDialog.Accepted:
            sys.exit(0)
        
        n, video_dir, csv_path = dialog.get_config()
    
    # Validate inputs
    if not n or not video_dir or not csv_path:
        QMessageBox.critical(None, "Error", "Invalid configuration")
        sys.exit(1)
    
    if not os.path.isdir(video_dir):
        QMessageBox.critical(None, "Error", f"Video directory does not exist: {video_dir}")
        sys.exit(1)
    
    try:
        # Initialize batch manager
        manager = BatchManager(video_dir, csv_path, n)
        
        # Get videos to process
        queue = manager.get_queue()
        
        if not queue:
            QMessageBox.information(
                None,
                "No videos to process",
                f"All videos in {video_dir} have already been processed, or no videos found."
            )
            sys.exit(0)
        
        print(f"Found {len(queue)} videos to process")
        
        # Show confirmation
        reply = QMessageBox.information(
            None,
            "Confirm",
            f"Ready to process {len(queue)} video(s) with {n} labels per video.\n\n"
            f"Output will be saved to:\n{csv_path}\n\n"
            "Continue?",
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Ok
        )
        
        if reply == QMessageBox.Cancel:
            sys.exit(0)
        
        # Process batch
        processed = manager.process_batch(queue)
        
        # Summary
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
