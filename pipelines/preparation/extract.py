import os
import sys
import argparse
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

CNN_WIDTH = 256
CNN_HEIGHT = 64

def get_roi_for_frame(frame_num: int, roi_sections: List[Dict]) -> Optional[List[List[int]]]:
    """
    Find the correct ROI for a given frame number from a list of sections.
    
    Args:
        frame_num: The frame number to look up
        roi_sections: List of section dicts with 'quad', 'start_frame', 'end_frame'
    
    Returns:
        ROI quad coordinates list [[x,y],...], or None if frame not covered
    """
    if not roi_sections:
        return None
    
    for section in roi_sections:
        start = section.get('start_frame', 0)
        end = section.get('end_frame', float('inf'))
        if start <= frame_num <= end:
            return section.get('quad')
    
    return None

def extract_frames_for_video(video_path: Path, frames: List[int], roi_sections: List[Dict], output_folder: Path, video_name: str):
    print(f"Starting extraction for {video_name}...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    count = 0
    skipped_no_roi = 0
    
    frames = sorted(list(set(frames)))
    
    for frame_num in frames:
        quad = get_roi_for_frame(frame_num, roi_sections)
        if quad is None:
            skipped_no_roi += 1
            continue
        
        # Quad is [[x,y], [x,y], [x,y], [x,y]]
        pts = [(int(p[0]), int(p[1])) for p in quad]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_num} from {video_name}")
            continue

        try:
            src_pts = np.float32(pts)
            dst_pts = np.float32([[0, 0], [CNN_WIDTH, 0], [CNN_WIDTH, CNN_HEIGHT], [0, CNN_HEIGHT]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, M, (CNN_WIDTH, CNN_HEIGHT), flags=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"Warning: Could not warp quad ROI for frame {frame_num}: {e}. Skipping frame.")
            continue
        
        out_name = f"{video_name}_{frame_num}.jpg"
        out_path = output_folder / out_name
        cv2.imwrite(str(out_path), warped)
        count += 1
        
    cap.release()
    if skipped_no_roi > 0:
        print(f"Warning: Skipped {skipped_no_roi} frames with no ROI coverage for {video_name}")
    print(f"Finished extraction for {video_name}: {count} frames.")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos using ROI definitions without GUI.")
    parser.add_argument('--csv', '-c', default='data/all_data.csv', help='Path to labels CSV')
    parser.add_argument('--video_dir', '-v', default='data/raw_videos', help='Directory containing video files')
    parser.add_argument('--output_dir', '-o', default='data/images/', help='Output directory for images')
    parser.add_argument('--roi_file', '-r', default='data/valid_video_sections.json', help='Path to JSON file containing ROI information')
    
    args = parser.parse_args()
    
    # Use absolute paths or relative to CWD, assuming running from project root
    csv_path = Path(args.csv)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    roi_path = Path(args.roi_file)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    if not video_dir.exists():
        print(f"Error: Video directory not found at {video_dir}")
        sys.exit(1)
        
    if not roi_path.exists():
        print(f"Error: ROI file not found at {roi_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    unique_videos = df['filename'].unique()
    
    print(f"Loading ROIs from {roi_path}...")
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)
    
    all_rois = roi_data.get("rois", {})
    
    print(f"Found {len(unique_videos)} videos in CSV.")
    
    processed_count = 0
    skipped_count = 0
    
    for i, video_name in enumerate(unique_videos):
        print(f"Processing ({i+1}/{len(unique_videos)}): {video_name}")
        video_path = video_dir / video_name
        
        if not video_path.exists():
            print(f"  Skipping: File not found in {video_dir}")
            skipped_count += 1
            continue
            
        if video_name not in all_rois:
            print(f"  Skipping: No ROI definition found in {roi_path}")
            skipped_count += 1
            continue
            
        roi_sections = all_rois[video_name].get("sections", [])
        if not roi_sections:
            print(f"  Skipping: ROI definition is empty")
            skipped_count += 1
            continue

        video_curr_df = df[df['filename'] == video_name]
        frames = video_curr_df['frame_number'].tolist()
        
        extract_frames_for_video(video_path, frames, roi_sections, output_dir, video_name)
        processed_count += 1

    print(f"\nAll processing complete. Processed {processed_count}, Skipped {skipped_count}.")

if __name__ == "__main__":
    main()
