import os
import pandas as pd

def cleanup_images(csv_path, images_dir):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    print(f"Reading valid ranges from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        
        # Create a dictionary to store valid ranges for each video
        # Assuming one range per video, or taking the min start and max end if multiple entries exist?
        # The CSV structure seems to have multiple rows per video, but start_frame and end_frame seem consistent for a video.
        # Let's verify if start/end frame are consistent per video or per row.
        # Based on previous read_file output:
        # GH010167.MP4,1243,6.825,1243,19399
        # GH010167.MP4,2042,7.471,1243,19399
        # It seems consistent. But to be safe, I'll group by filename and take the min start and max end, 
        # OR just check if the frame is valid for *any* row corresponding to that video?
        # Actually, the user said "check this csv ... for frames that are outside the start/end frame index range".
        # This implies that for a given row, the frame_number must be within start_frame and end_frame.
        # But for images, we just have the image file. We don't know which specific "row" it belongs to if there were multiple ranges.
        # However, looking at the data, it seems `start_frame` and `end_frame` define the valid segment of the video *globally* for that video file in this dataset context.
        # So I will assume that for a given video filename, there is a global valid range defined by the min(start_frame) and max(end_frame) across all rows for that video.
        # Or more likely, start_frame and end_frame are the same for all rows of a video.
        
        # Let's aggregate ranges per video
        video_ranges = {}
        for index, row in df.iterrows():
            filename = row['filename']
            start = row['start_frame']
            end = row['end_frame']
            
            if filename not in video_ranges:
                video_ranges[filename] = {'start': start, 'end': end}
            else:
                # If there are multiple ranges, we might need to be careful.
                # But based on the sample, it looks like metadata for the video clip.
                # Let's assume the widest range if they differ, or just take the first one found.
                # Actually, if I look at the previous output:
                # GH010168.MP4,161,7.502,0,19399
                # GH010168.MP4,154,7.499,0,19399
                # It seems consistent.
                pass
        
        print(f"Found ranges for {len(video_ranges)} videos.")
        
        files = os.listdir(images_dir)
        removed_count = 0
        kept_count = 0
        
        for file in files:
            if not file.endswith('.jpg'):
                continue
                
            # Parse filename: VideoName.MP4_FrameNumber.jpg
            # We need to be careful about the split.
            # The format seems to be: {video_filename}_{frame_number}.jpg
            # video_filename ends with .MP4
            
            try:
                # Split from the right to get the frame number
                base_name = os.path.splitext(file)[0] # Remove .jpg
                parts = base_name.rsplit('_', 1)
                
                if len(parts) != 2:
                    print(f"Skipping file with unexpected format: {file}")
                    continue
                    
                video_name = parts[0]
                frame_number_str = parts[1]
                
                if not frame_number_str.isdigit():
                     print(f"Skipping file with non-numeric frame number: {file}")
                     continue
                     
                frame_number = int(frame_number_str)
                
                if video_name in video_ranges:
                    valid_range = video_ranges[video_name]
                    if not (valid_range['start'] <= frame_number <= valid_range['end']):
                        print(f"Removing {file}: Frame {frame_number} outside range [{valid_range['start']}, {valid_range['end']}]")
                        os.remove(os.path.join(images_dir, file))
                        removed_count += 1
                    else:
                        kept_count += 1
                else:
                    # If video is not in CSV, should we remove it?
                    # The user said "make sure that all images ... are in the correct range".
                    # If we don't know the range, we can't verify.
                    # But usually, if it's not in the dataset CSV, it might be extraneous.
                    # However, to be safe and strictly follow "outside the start/end frame index range",
                    # I will only remove if I KNOW it is outside the range.
                    # If I don't have a range, I'll skip it (or maybe warn).
                    # Let's assume we only clean images for videos we know about.
                    # print(f"Skipping {file}: Video {video_name} not found in CSV.")
                    pass

            except Exception as e:
                print(f"Error processing {file}: {e}")
                
        print(f"Finished cleanup. Removed {removed_count} images. Kept {kept_count} images.")

    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    # Adjust paths
    workspace_root = os.getcwd()
    csv_path = os.path.join(workspace_root, "data/all_data.csv")
    images_dir = os.path.join(workspace_root, "data/images")
    
    cleanup_images(csv_path, images_dir)
