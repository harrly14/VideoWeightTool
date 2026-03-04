import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

def check_duplicates(df, n):
    duplicates = df[df.duplicated(subset=['filename', 'frame_number'], keep=False)]
    if not duplicates.empty:
        display_rows = n
        print(f"Found {len(flagged)} duplicate rows")
        print(f"Printing the first {display_rows}:")
        print("-" * 40)
        print(flagged.sort_values(by=['filename', 'frame_number']).head(display_rows).to_string(index=False))
        print("-" * 40)
        return flagged
    return None

def check_erratic_fluctuations(df, threshold=0.4, n):
    df.sort_values(by=['filename', 'frame_number'], inplace=True)
    df['abs_diff'] = df.groupby('filename')['weight'].diff().abs()

    flagged = df[df['abs_diff'].notna() & (df['abs_diff'] >= threshold)].copy()

    if not flagged.empty:
        display_rows = n
        print(f"Found {len(flagged)} rows with erratic fluctuations")
        print(f"Printing the first {display_rows}:")
        print("-" * 40)
        print(flagged.sort_values(by=['filename', 'frame_number']).head(display_rows).to_string(index=False))
        print("-" * 40)
        return flagged
    return None

def validate_format(df, min_val=6, max_val=8, n):
    pattern = re.compile(r'^\d\.\d{3}$')
    flagged =  df[~df['filename'].str.contains(pattern) &
                   ~(df['value'].between(min_val, max_val))]
    
    if not flagged.empty:
        display_rows = n
        print(f"Found {len(flagged)} rows with an invalid format (or out weight out of range)")
        print(f"Printing the first {display_rows}:")
        print("-" * 40)
        print(flagged.sort_values(by=['filename', 'frame_number']).head(display_rows).to_string(index=False))
        print("-" * 40)
        return flagged
    return None

def _handle_duplicate(df, flagged_df, row):
    fname, fnum = row['filename'], row['frame_number']
    dupes = df[(df['filename'] == fname) & (df['frame_number'] == fnum)]

    print(f"\n{'=' * 50}")
    print(f"DUPLICATE: filename={fname}, frame_number={fnum}")
    print(f"{'=' * 50}")

    dupe_indices = dupes.index.tolist()
    for i, (idx, r) in enumerate(dupes.iterrows(), 1):
        print(f"  [{i}] index={idx}  weight={r['weight']}")

    print("\nOptions:")
    print(f" 1-{len(dupes)}: Keep that row, drop the others")
    print(" t: Type a custom weight (keep first row, drop rest)")
    print(" f: Change frame numbers to resolve conflict")

    choice = input("Choice: ").strip().lower()

    if choice == 't':
        val = input("Enter weight value: ").strip()
        try:
            weight = float(val)
        except ValueError:
            print("Invalid number. Skipping this group.")
            return df, flagged_df
        keep_idx = dupe_indices[0]
        df.at[keep_idx, 'weight'] = weight
        drop_indices = dupe_indices[1:]
        df = df.drop(index=drop_indices)
    elif choice == 'f':
        print("Enter new frame number for each row (blank = keep current):")
        for i, (idx, r) in enumerate(dupes.iterrows(), 1):
            new_fn = input(f"  [{i}] frame_number={int(r['frame_number'])}: ").strip()
            if new_fn:
                try:
                    df.at[idx, 'frame_number'] = int(new_fn)
                except ValueError:
                    print(f"    Invalid number, keeping {int(r['frame_number'])}")
    else:
        try:
            pick = int(choice)
            if 1 <= pick <= len(dupe_indices):
                keep_idx = dupe_indices[pick - 1]
                drop_indices = [i for i in dupe_indices if i != keep_idx]
                df = df.drop(index=drop_indices)
            else:
                print("Out of range. Skipping this group.")
                return df, flagged_df
        except ValueError:
            print("Invalid input. Skipping this group.")
            return df, flagged_df

    # Remove all flagged entries for this duplicate group
    flagged_df = flagged_df[
        ~((flagged_df['filename'] == fname) & (flagged_df['frame_number'] == fnum))
    ]
    return df, flagged_df


def _handle_fluctuation(df, flagged_df, row):
    fname, fnum = row['filename'], row['frame_number']
    match = df[(df['filename'] == fname) & (df['frame_number'] == fnum)]
    if match.empty:
        flagged_df = flagged_df[
            ~((flagged_df['filename'] == fname) & (flagged_df['frame_number'] == fnum))
        ]
        return df, flagged_df

    row_idx = match.index[0]

    # Show surrounding context from the same video
    video_df = df[df['filename'] == fname].sort_values('frame_number')
    video_indices = video_df.index.tolist()
    pos = video_indices.index(row_idx)
    start = max(0, pos - 3)
    end = min(len(video_indices), pos + 4)
    context_indices = video_indices[start:end]

    print(f"\n{'=' * 50}")
    print(f"FLUCTUATION: filename={fname}, frame_number={fnum}, weight={row['weight']}")
    print(f"{'=' * 50}")
    print("Surrounding rows:")
    for ci in context_indices:
        marker = " >>>" if ci == row_idx else "    "
        r = df.loc[ci]
        print(f"{marker} frame={int(r['frame_number'])}  weight={r['weight']}")

    # Compute interpolated value from neighbours
    prev_idx = video_indices[pos - 1] if pos > 0 else None
    next_idx = video_indices[pos + 1] if pos < len(video_indices) - 1 else None
    if prev_idx is not None and next_idx is not None:
        avg = (df.at[prev_idx, 'weight'] + df.at[next_idx, 'weight']) / 2
    elif prev_idx is not None:
        avg = df.at[prev_idx, 'weight']
    else:
        avg = df.at[next_idx, 'weight'] if next_idx is not None else None

    print("\nOptions:")
    print("  k: Keep / unflag")
    if avg is not None:
        print(f"  i: Interpolate (replace with {avg:.3f})")
    print("  t: Type a custom weight")
    print("  d: Drop this row")

    choice = input("Choice: ").strip().lower()

    if choice == 'k':
        pass
    elif choice == 'i' and avg is not None:
        df.at[row_idx, 'weight'] = round(avg, 3)
    elif choice == 't':
        val = input("Enter weight value: ").strip()
        try:
            df.at[row_idx, 'weight'] = float(val)
        except ValueError:
            print("Invalid number. Skipping.")
            return df, flagged_df
    elif choice == 'd':
        df = df.drop(index=row_idx)
    else:
        print("Invalid input. Skipping.")
        return df, flagged_df

    flagged_df = flagged_df[
        ~((flagged_df['filename'] == fname) & (flagged_df['frame_number'] == fnum))
    ]
    return df, flagged_df


def _handle_invalid_format(df, flagged_df, row):
    fname, fnum = row['filename'], row['frame_number']
    match = df[(df['filename'] == fname) & (df['frame_number'] == fnum)]
    if match.empty:
        flagged_df = flagged_df[
            ~((flagged_df['filename'] == fname) & (flagged_df['frame_number'] == fnum))
        ]
        return df, flagged_df

    row_idx = match.index[0]

    print(f"\n{'=' * 50}")
    print(f"INVALID FORMAT: filename={fname}, frame_number={fnum}, weight={df.at[row_idx, 'weight']}")
    print(f"{'=' * 50}")
    print("\nOptions:")
    print("  t: Type a corrected weight")
    print("  d: Drop this row")
    print("  k: Keep / unflag")

    choice = input("Choice: ").strip().lower()

    if choice == 't':
        val = input("Enter weight value: ").strip()
        try:
            df.at[row_idx, 'weight'] = float(val)
        except ValueError:
            print("Invalid number. Skipping.")
            return df, flagged_df
    elif choice == 'd':
        df = df.drop(index=row_idx)
    elif choice == 'k':
        pass
    else:
        print("Invalid input. Skipping.")
        return df, flagged_df

    flagged_df = flagged_df[
        ~((flagged_df['filename'] == fname) & (flagged_df['frame_number'] == fnum))
    ]
    return df, flagged_df


def handle_flagged(df,flagged_df, output_path):
    print(f"Flagged {len(flagged_df)} rows. How would you like to proceed?")
    print("1. Fix interactively (step through each row in CLI)")
    print("2. Print summary and exit")
    print("3. Export flagged_rows.csv and exit")

    choice = input("Select option: ").strip()

    if choice == '1':
        while not flagged_df.empty:
            row = flagged_df.iloc[0]
            reason = row['flag_reason']
            remaining = len(flagged_df)
            print(f"\n[{remaining} flagged row(s) remaining]")

            if reason == 'duplicate':
                df, flagged_df = _handle_duplicate(df, flagged_df, row)
            elif reason == 'large fluctuation':
                df, flagged_df = _handle_fluctuation(df, flagged_df, row)
            elif reason == 'invalid format/out of range':
                df, flagged_df = _handle_invalid_format(df, flagged_df, row)
            else:
                print(f"Unknown flag reason: {reason}. Removing from flagged list.")
                flagged_df = flagged_df.iloc[1:]

        print("\nAll flagged rows resolved.")

    elif choice == '2':
        print("\n")
        print("-" * 40)
        print(flagged.sort_values(by=['filename', 'frame_number']).to_string(index=False))
        print("-" * 40)
        print("\n")
        sys.exit()
    elif choice == '3':
        filename = output_path / 'flagged_rows.csv'
        flagged_df.to_csv(filename, index=False)
        print(f"Saved splits to {filename}")
        sys.exit()
    else:
        print("Invalid choice. Please try again.")
        input("Press Enter to continue...")

def split_data(df, train=0.75, val=0.125, test=0.125, waste_threshold=0.10):
    """
    Split a labelled frame DataFrame into train, validation, and test sets,
    keeping all frames from the same video in the same split to avoid data leakage.

    Uses a two-phase greedy algorithm to get as close to the target ratios as
    possible while minimising discarded frames:

      Phase 1: largest-first, whole videos only.
        Videos are sorted by frame count descending and assigned whole to
        whichever split has the largest remaining deficit, provided the video
        fits within that split's remaining budget. The scan restarts after each
        successful assignment so no valid placement is missed.

      Phase 2: smallest-first, trim to fit.
        Videos that were too large to fit whole into any remaining budget slot
        are sorted ascending and used to top up each split. A video is trimmed
        to exactly the frames needed; the leftover tail is discarded. Working
        smallest-first ensures the trimmed tail is as small as possible.

    If the percentage of unused frames is larger than the waste threshold,
    recommend_labeling() is triggered to help get closer to target splits.
    """
    print("Attempting to split data...")
    print(f"Target splits: {train*100}% train, {val*100}% validation, {test*100}% test")

    groups = df.groupby(by='filename')
    video_sizes = groups.size().reset_index(name='frame_count')
    video_sizes = video_sizes.sort_values('frame_count', ascending=False).reset_index(drop=True)

    total_frames = video_sizes['frame_count'].sum()
    split_targets = {
        'train': round(train * total_frames),
        'val':   round(val * total_frames),
        'test':  round(test * total_frames),
    }
    remaining_budget = split_targets.copy()

    # each entry in buckets is (filename, frame_number). 
    # if frame_number=None, it means take the whole video
    buckets = {'train': [], 'val': [], 'test': []} 

    # =========== Phase 1 (largest-first) ===========
    unassigned = video_sizes.sort_values('frame_count', ascending=False).copy()

    still_fitting = True
    while still_fitting:
        still_fitting = False
        used_indices = []

        for i, row in unassigned.iterrows():
            filename, frame_count = row['filename'], row['frame_count']

            # find the split with the biggest deficit that can fit this video
            eligible = {}
            for s in remaining_budget:
                if remaining_budget[s] >= frame_count:
                    eligible[s] = remaining_budget[s]
            if not eligible:
                continue # video is too big for any split
                
            best_split = max(eligible, key=eligible.get)
            buckets[best_split].append((filename, None))
            remaining_budget[best_split] -= frame_count
            used_indices.append(i)
            still_fitting = True

        unassigned = unassigned.drop(index=used_indices)
            
    # =========== Phase 2 (smallest-first) ===========
    unassigned = video_sizes.sort_values('frame_count', ascending=False).copy()

    for split in ('train', 'val', 'test'):
        for i, row in unassigned.iterrows():
            if remaining_budget[split] <= 0: break

            filename, frame_count = row['filename'], row['frame_count']
            take = min(frame_count, remaining_budget[split])
            if take == frame_count: n_taken_frames = None
            else: n_taken_frames = take
            buckets[split].append((filename, n_taken_frames))
            remaining_budget[split] -= take

    # =========== processing n stuff ===========
    assigned_counts = {}
    for s in split_targets:
        assigned_counts[s] = split_targets[s] - remaining_budget[s]
    total_assigned = sum(assigned_counts.values())

    wasted_frames = total_frames - total_assigned
    wasted_percent = wasted_frames/total_frames if total_frames > 0 else 0

    for split in ('train', 'val', 'train'):
        actual_percent = assigned_counts[split] / total_frames * 100
        target_percent = split_targets[split] / total_frames * 100
        print(f"{split:>5}: {actual_percent:.1f}% (target {target_percent:.1f}%)")

    if wasted_percent > waste_threshold: 
        print(f"  WARNING: {wasted_frames*100:.1f}% of frames unassigned (threshold: {waste_threshold*100:.1f}%)")
        print("Triggering recommended labelling...")
        recommend_labeling(df, wasted_frames, waste_threshold)

    result = {}
    for split_name, entries in buckets.items():
        parts = []
        for filename, num_frames in entries:
            video_df = groups.get_group(filename)
            if num_frames is not None:
                video_df = video_df.iloc[:num_frames]
            parts.append(video_df)
        if parts:
            result[split_name] = pd.concat(parts).reset_index(drop=True)
        else:
            result[split_name] = pd.DataFrame(columns=df.columns)

    train_df = result['train']
    val_df = result['val']
    test_df = result['test']

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

def balance_classes(dfs):
    # need to come back to this after moving from CRNN to CNN


if __name__ == "__main__":
    # add args: csvpath, output path, display row num, waste_threshold
    # prompt for those args if not given, or use default values

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    expected_columns = ['filename','frame_number','weight']
    actual_columns = df.columns.tolist()

    if actual_columns != expected_columns:
        print("DataFrame columns do not match.")
        print(f"Expected columns: {expected_columns}")
        print(f"Actual columns: {actual_columns}")
        print("CSV columns must match the expected columns exactly. Exiting...")
        sys.exit()

    n = len(df)
    print(f"Total samples: {n}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    flagged = pd.DataFrame()

    for func, reason in zip(
        [check_duplicates, check_erratic_fluctuations, validate_format],
        ['duplicate', 'large fluctuation', 'invalid format/out of range']
    ):
        temp_df = func(df, n=10)
        if temp_df is not None: 
            temp_df['flag_reason'] = reason
            flagged = pd.concat([flagged, temp_df], ignore_index=True)

    handle_flagged(df, flagged, output_path)

    train_df, val_df, test_df = split_data(df)
    dfs = [train_df, val_df, test_df]
    balance_classes(dfs)

    print(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
    print(f"Val:   {len(val_df)} ({len(val_df)/n:.1%})")
    print(f"Test:  {len(test_df)} ({len(test_df)/n:.1%})")

    train_df.to_csv(output_path / 'train_labels.csv', index=False)
    val_df.to_csv(output_path / 'val_labels.csv', index=False)
    test_df.to_csv(output_path / 'test_labels.csv', index=False)
    
    print(f"Saved splits to {output_path}")