import pandas as pd
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def check_duplicates(df):
    duplicates = df[df.duplicated(subset=['filename', 'frame_number'], keep=False)]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate rows:")
        print("-" * 40)
        print(duplicates.sort_values(by=['filename', 'frame_number']).to_string(index=False))
        print("-" * 40)
        return duplicates
    return None

def check_erratic_fluctuations(df, threshold):
    # the idea: 
    # Group by filename, sort each group by frame_num, then compute the absolute diff between 
    # consecutive label values. Flag any row where the jump exceeds threshold. Worth noting: 
    # the first frame of a video should be exempt since there's no prior frame to compare against — 
    # you don't want cross-video diffs leaking in either, so grouping strictly by filename before diffing is important.

    # Use df.groupby('filename')['weight'].diff().abs(). 
    # Don't forget to fill the NaN at the start of each group with 0 so you don't accidentally flag the first frame of every video.

def validate_format(df, min_val, max_val):
    # two checks: regex match for X.XXX format 
    # and a configurable range check (e.g. [min_val, max_val] passed as args or set in a config).
    # Both checks should tag rows with a specific reason so the user knows what they're fixing.

def handle_flagged(flagged_df):
    # After all three validation checks run, collect all flagged rows into one unified df (with a reason column), then prompt:
    # "Flagged X rows. How would you like to proceed?"
    # 1. Fix interactively (step through each row in CLI)
    # 2. Print summary and exit
    # 3. Export flagged_rows.csv and exit
    # For option 1, show the row, the reason it was flagged, and let the user type a corrected value or skip it. 
    # Skipped rows get dropped from the df before splitting. 
    # After interactive fixing, re-run validation to confirm no new issues were introduced.

    # add a "Bulk Auto-Fix" option for fluctuations.
    # The Logic: If a weight jumps from 7.450 -> 7.850 -> 7.451, 
    # it is almost certainly a single-frame glitch. 
    # You could offer to automatically interpolate (average) that middle frame based on its neighbors rather than making the user type it in manually.

def split_data(df, train=0.75, val=0.125, test=0.125, waste_threshold=0.10)
    print("Attempting to split data...")
    print(f"Target splits: {train*100}% train, {val*100}% validation, {test*100}% test")
    print("To avoid ")
    # Group by filename so whole videos are never split across sets. 
    df = df.groupby(by='filename')
    # Assign each video as a unit to train/val/test. 
    # Because video lengths vary, perfect ratios are unlikely
    # so, sort videos by frame count descending, assign each to whichever split is furthest below its target).
    # After the initial assignment, compute actual vs. target ratios and the percentage of total frames 
    # that ended up unassigned (if any). If wasted frames exceed waste_threshold, trigger recommend_labeling().
    # Before the final assignment, check if swapping two similar-sized videos between 
    # Val and Test brings you closer to the 12.5% target. This is much better than "wasting" data by leaving frames unassigned.
    
    # unique_videos = df['filename'].unique()
    # # Shuffle the videos, not the dataframe rows
    # np.random.seed(42)
    # np.random.shuffle(unique_videos)

    # handle the actual split
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]

    return train_df, val_df, test_df


def recommend_labeling(df, unlabeled_videos, waste_threshold):
    # (Called when waste exceeds threshold)
    # The goal is to recommend which currently-unlabeled videos the user should label to simultaneously optimize across three axes:
    # 1. Reduce waste / improve split ratio fit 
        # identify unlabeled videos whose frame counts would slot neatly into whichever split is currently undershooting its target. 
        # A video that brings val from 11.2% → 12.4% is a better pick than one that overshoots to 14%.
    # 2. Improve per-digit-position class balance 
        # for each digit position (ones, tenths, hundredths, thousandths), compute the current distribution across train/val/test. 
        # Score each unlabeled video by how much its label distribution would reduce imbalance if added. 
        # Since unlabeled videos don't have labels yet, you'd need to either ask the user to quick-label a sample, 
        # or skip this axis for truly unlabeled data and focus on the frame-count axes.
    # 3. Maximize data utilization 
        # prioritize recommendations where adding a video meaningfully increases total labeled frames in use vs. just nudging ratios.
    # Combine these into a single recommendation score per video (weighted sum, with configurable weights). 
    # Present the top N recommendations with a plain-language explanation: 
    # "Labeling video_042.mp4 (320 frames) would bring your val split to 12.6% and fill a gap in the tenths=3 class."
    # Considerations: 
    # Video diversity: 
        # a long video with 500 frames of label 1.234 is less valuable than a shorter video with varied labels. 
        # Factor in label entropy of candidate videos if any partial labels exist.
    # Diminishing returns:
        # if the user follows recommendation #1, recommendations #2-N should update dynamically rather than being computed all at once upfront.
    # User confirmation gate:
        # before proceeding to split after recommendations are shown, always confirm the user is happy with the final video-to-split assignments.

    # gemini says: 
    # The "Anchor" Frame Approach: 
    # Since you can't know the labels of unlabeled videos, use the research log approach we talked about earlier. 
    # If the user says "Video_X starts at 6.8kg and ends at 7.2kg," the script can assume a linear distribution to estimate which digits it will provide.
    
    # Axis of Optimization: 
    # Focus on Position Balance first. 
    # If your heatmap shows zero 9s, and an unlabeled video is from a time of high swarm activity, that video gets a massive "Priority Bonus."

def balance_classes(train_df, val_df, test_df)
    # After splitting, check class distribution at each digit position independently. 
    # If any position is significantly skewed within a split, apply stratified oversampling 
    # (or undersampling, configurable) at the frame level — not the video level, 
    # since we're already past the video-grouping step. 
    # Oversampling here means duplicating individual frames, which is fine for a CRNN training set.

    # Warning: Do not oversample your Validation or Test sets. 
    # You should only ever oversample the Train set. 
    # Your Test set must remain "pure" to give you an honest accuracy score.


if __name__ == "__main__":
    # this is all used to be in split_data and needs fixing
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    expected_columns = ['filename','frame_number','weight']
    actual_columns = df.columns.tolist()

    if actual_columns != expected_columns:
        print("DataFrame columns do not match.")
        print(f"Expected columns: {expected_columns}")
        print(f"Actual columns: {actual_columns}")
        print("CSV columns must match the expected columns exactly. Exiting...")
        sys.exit(0)

    n = len(df)
    print(f"Total samples: {n}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
        
    # ==========================================
    duplicates_df = check_duplicates(df)
    check_erratic_fluctuations(df, )
    validate_format(df, )
    handle_flagged(flagged_df)

    train_df, val_df, test_df = split_data(.....)
    balance_classes(...)

    print(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
    print(f"Val:   {len(val_df)} ({len(val_df)/n:.1%})")
    print(f"Test:  {len(test_df)} ({len(test_df)/n:.1%})")

    train_df.to_csv(output_path / 'train_labels.csv', index=False)
    val_df.to_csv(output_path / 'val_labels.csv', index=False)
    test_df.to_csv(output_path / 'test_labels.csv', index=False)
    
    print(f"Saved splits to {output_path}")