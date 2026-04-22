import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def check_duplicates(df, n=10):
    flagged = df[df.duplicated(subset=['filename', 'frame_number'], keep=False)].copy()
    if not flagged.empty:
        print(f"Found {len(flagged)} duplicate rows")
        print(f"Printing the first {n}:")
        print("-" * 40)
        print(flagged.sort_values(by=['filename', 'frame_number']).head(n).to_string(index=False))
        print("-" * 40)
        return flagged
    return None

def check_erratic_fluctuations(df, threshold=0.4, n=10):
    temp = df.sort_values(by=['filename', 'frame_number']).copy()
    temp['abs_diff'] = temp.groupby('filename')['weight'].diff().abs()

    flagged = temp[temp['abs_diff'].notna() & (temp['abs_diff'] >= threshold)].copy()

    if not flagged.empty:
        print(f"Found {len(flagged)} rows with erratic fluctuations")
        print(f"Printing the first {n}:")
        print("-" * 40)
        print(flagged.sort_values(by=['filename', 'frame_number']).head(n).to_string(index=False))
        print("-" * 40)
        return flagged.drop(columns=['abs_diff'])
    return None

def validate_format(df, min_val=6, max_val=8, n=10, weight_raw=None):
    pattern = re.compile(r'^\d\.\d{3}$')
    if weight_raw is not None:
        weight_as_str = pd.Series(weight_raw).astype(str).str.strip()
    else:
        numeric_for_format = pd.to_numeric(df['weight'], errors='coerce')
        weight_as_str = numeric_for_format.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    in_format = weight_as_str.str.match(pattern)
    numeric_weight = pd.to_numeric(df['weight'], errors='coerce')
    in_range = numeric_weight.between(min_val, max_val)
    flagged = df[~in_format | ~in_range].copy()
    
    if not flagged.empty:
        print(f"Found {len(flagged)} rows with an invalid format (or out weight out of range)")
        print(f"Printing the first {n}:")
        print("-" * 40)
        print(flagged.sort_values(by=['filename', 'frame_number']).head(n).to_string(index=False))
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
        print(flagged_df.sort_values(by=['filename', 'frame_number']).to_string(index=False))
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

    return df, flagged_df

def _weight_to_digits(weight):
    try:
        value = float(weight)
    except (TypeError, ValueError):
        return None

    as_digits = f"{value:.3f}".replace('.', '')
    if len(as_digits) != 4 or not as_digits.isdigit():
        return None
    return as_digits

def _digit_hist_from_df(df):
    hist = np.zeros(10, dtype=np.int64)
    for weight in df['weight'].tolist():
        digits = _weight_to_digits(weight)
        if digits is None:
            continue
        for d in digits:
            hist[int(d)] += 1
    return hist

def _build_row_strat_labels(df):
    numeric_weight = pd.to_numeric(df['weight'], errors='coerce')
    seq_labels = numeric_weight.map(lambda x: f"{x:.3f}" if pd.notna(x) else 'nan')

    counts = seq_labels.value_counts(dropna=False)
    rare = set(counts[counts < 2].index.tolist())
    labels = seq_labels.where(~seq_labels.isin(rare), other='other').fillna('other')
    return labels

def _split_rows_stratified(df, train, val, test, seed):
    if df.empty:
        empty = pd.DataFrame(columns=df.columns)
        return empty.copy(), empty.copy(), empty.copy()

    labels = _build_row_strat_labels(df)
    holdout_ratio = val + test

    if holdout_ratio <= 0:
        return df.copy().reset_index(drop=True), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    try:
        train_df, holdout_df = train_test_split(
            df,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        train_df, holdout_df = train_test_split(
            df,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=None,
        )

    if len(holdout_df) == 0:
        return train_df.reset_index(drop=True), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    if val <= 0:
        return train_df.reset_index(drop=True), pd.DataFrame(columns=df.columns), holdout_df.reset_index(drop=True)
    if test <= 0:
        return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True), pd.DataFrame(columns=df.columns)

    val_share_of_holdout = val / (val + test)
    holdout_labels = _build_row_strat_labels(holdout_df)

    try:
        val_df, test_df = train_test_split(
            holdout_df,
            test_size=(1 - val_share_of_holdout),
            random_state=seed,
            stratify=holdout_labels,
        )
    except ValueError:
        val_df, test_df = train_test_split(
            holdout_df,
            test_size=(1 - val_share_of_holdout),
            random_state=seed,
            stratify=None,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def split_data(df, train=0.75, val=0.125, test=0.125, waste_threshold=0.10, seed=42):
    """
        Split a labelled frame DataFrame into train, validation, and test sets
        using row-level stratified splitting (no video-level grouping).

        The split stratifies on weight sequence labels where rare labels are
        bucketed into an 'other' class for stability.

        Note: waste_threshold is accepted for API compatibility but is not used
        in row-level splitting because all rows are assigned.
    """
    print("Attempting to split data...")
    print(f"Target splits: {train*100:.1f}% train, {val*100:.1f}% validation, {test*100:.1f}% test")

    ratio_sum = train + val + test
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0 (got {ratio_sum:.4f})")

    if df.empty:
        empty = pd.DataFrame(columns=df.columns)
        return empty.copy(), empty.copy(), empty.copy()

    train_df, val_df, test_df = _split_rows_stratified(df, train, val, test, seed)

    total_rows = len(df)
    ratios = {'train': train, 'val': val, 'test': test}
    result = {'train': train_df, 'val': val_df, 'test': test_df}

    print("Split summary:")
    for split_name in ('train', 'val', 'test'):
        actual_percent = (len(result[split_name]) / total_rows) * 100 if total_rows > 0 else 0
        target_percent = ratios[split_name] * 100
        print(f"{split_name:>5}: {actual_percent:.1f}% (target {target_percent:.1f}%)")

    return train_df, val_df, test_df

def balance_classes(dfs, split_names=('train', 'val', 'test')):
    print("\nDigit-class balance report:")
    print("-" * 60)

    reports = {}
    for split_name, split_df in zip(split_names, dfs):
        hist = _digit_hist_from_df(split_df)
        total = int(hist.sum())
        nonzero = hist[hist > 0]
        imbalance_ratio = float(nonzero.max() / nonzero.min()) if len(nonzero) > 0 else float('inf')

        reports[split_name] = {
            'hist': hist,
            'total_digit_samples': total,
            'imbalance_ratio': imbalance_ratio,
        }

        print(f"{split_name.upper()} | total digit labels: {total} | imbalance ratio: {imbalance_ratio:.2f}")
        print("  " + " ".join([f"{d}:{hist[d]}" for d in range(10)]))

    print("-" * 60)
    return reports

def _parse_args():
    parser = argparse.ArgumentParser(description="Validate and split labelled frame data.")
    parser.add_argument('--csv-path', default='data/all_data.csv', help='Input CSV with filename,frame_number,weight columns')
    parser.add_argument('--output-dir', default='data/labels', help='Directory to save split CSV files')
    parser.add_argument('--display-rows', type=int, default=10, help='Rows to print for each validation warning')
    parser.add_argument('--fluctuation-threshold', type=float, default=0.4, help='Abs weight delta threshold to flag')
    parser.add_argument('--min-weight', type=float, default=6.0, help='Minimum valid weight')
    parser.add_argument('--max-weight', type=float, default=8.0, help='Maximum valid weight')
    parser.add_argument('--train', type=float, default=0.75, help='Train ratio')
    parser.add_argument('--val', type=float, default=0.125, help='Validation ratio')
    parser.add_argument('--test', type=float, default=0.125, help='Test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic split tie-breaks')
    parser.add_argument('--waste-threshold', type=float, default=0.10, help='Unused-frame warning threshold')
    parser.add_argument('--skip-interactive-flags', action='store_true', help='Do not prompt for flagged rows; continue split')
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_dir)

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'weight': 'string'})

    expected_columns = ['filename','frame_number','weight']
    actual_columns = df.columns.tolist()

    if actual_columns != expected_columns:
        print("DataFrame columns do not match.")
        print(f"Expected columns: {expected_columns}")
        print(f"Actual columns: {actual_columns}")
        print("CSV columns must match the expected columns exactly. Exiting...")
        sys.exit()

    weight_raw = df['weight'].copy()
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    n = len(df)
    print(f"Total samples: {n}")
    
    output_path.mkdir(parents=True, exist_ok=True)

    flagged = pd.DataFrame()

    duplicate_rows = check_duplicates(df, n=args.display_rows)
    if duplicate_rows is not None:
        duplicate_rows['flag_reason'] = 'duplicate'
        flagged = pd.concat([flagged, duplicate_rows], ignore_index=True)

    fluctuation_rows = check_erratic_fluctuations(
        df,
        threshold=args.fluctuation_threshold,
        n=args.display_rows,
    )
    if fluctuation_rows is not None:
        fluctuation_rows['flag_reason'] = 'large fluctuation'
        flagged = pd.concat([flagged, fluctuation_rows], ignore_index=True)

    invalid_rows = validate_format(
        df,
        min_val=args.min_weight,
        max_val=args.max_weight,
        n=args.display_rows,
        weight_raw=weight_raw,
    )
    if invalid_rows is not None:
        invalid_rows['flag_reason'] = 'invalid format/out of range'
        flagged = pd.concat([flagged, invalid_rows], ignore_index=True)

    if not flagged.empty:
        if args.skip_interactive_flags:
            flagged_file = output_path / 'flagged_rows.csv'
            flagged.to_csv(flagged_file, index=False)
            print(f"Flagged rows saved to {flagged_file}; continuing split without interactive fixes.")
        else:
            df, _ = handle_flagged(df, flagged, output_path)

    train_df, val_df, test_df = split_data(
        df,
        train=args.train,
        val=args.val,
        test=args.test,
        waste_threshold=args.waste_threshold,
        seed=args.seed,
    )
    dfs = [train_df, val_df, test_df]
    balance_classes(dfs)

    print(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
    print(f"Val:   {len(val_df)} ({len(val_df)/n:.1%})")
    print(f"Test:  {len(test_df)} ({len(test_df)/n:.1%})")

    train_df.to_csv(output_path / 'train_labels.csv', index=False)
    val_df.to_csv(output_path / 'val_labels.csv', index=False)
    test_df.to_csv(output_path / 'test_labels.csv', index=False)
    
    print(f"Saved splits to {output_path}")