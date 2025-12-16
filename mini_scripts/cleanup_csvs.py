import pandas as pd
import os

def cleanup_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Processing {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # Ensure columns exist
        required_columns = ['frame_number', 'start_frame', 'end_frame']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping {file_path}: Missing required columns.")
            return

        initial_count = len(df)
        
        # Filter rows
        # Keep rows where frame_number is within [start_frame, end_frame]
        # Note: Assuming inclusive range based on typical usage, but user said "outside the start/end frame index range"
        # Usually start/end are inclusive boundaries.
        
        mask = (df['frame_number'] >= df['start_frame']) & (df['frame_number'] <= df['end_frame'])
        cleaned_df = df[mask]
        
        final_count = len(cleaned_df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            cleaned_df.to_csv(file_path, index=False)
            print(f"Removed {removed_count} rows from {file_path}. (Original: {initial_count}, New: {final_count})")
        else:
            print(f"No rows removed from {file_path}.")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    files_to_clean = [
        "data/all_data.csv",
        "data/labels/train_labels.csv",
        "data/labels/val_labels.csv",
        "data/labels/test_labels.csv"
    ]
    
    # Adjust paths to be absolute or relative to workspace root
    workspace_root = os.getcwd()
    
    for rel_path in files_to_clean:
        abs_path = os.path.join(workspace_root, rel_path)
        cleanup_csv(abs_path)
