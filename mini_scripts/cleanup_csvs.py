"""mini_scripts/cleanup_csvs.py

Clean CSV files by removing rows whose `frame_number` falls outside the
per-video ranges defined in `data/all_data.json` (preferred). Falls back to
filtering using `start_frame`/`end_frame` columns if JSON is missing.

Usage:

# Dry-run, default (will only print summary)
python mini_scripts/cleanup_csvs.py --csv data/labels/train_labels.csv

# In-place cleanup with backup
python mini_scripts/cleanup_csvs.py --csv data/labels/train_labels.csv --inplace --backup

# Write to a separate cleaned file
python mini_scripts/cleanup_csvs.py --csv data/labels/train_labels.csv --out cleaned.csv
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd


def load_json_ranges(json_path: Path) -> Optional[Dict[str, Tuple[int, int]]]:
    if not json_path.exists():
        return None
    try:
        with json_path.open("r") as f:
            data = json.load(f)
        vr = data.get("video_ranges", {})
        return {k: (int(v["min_frame"]), int(v["max_frame"])) for k, v in vr.items()}
    except Exception:
        return None


def cleanup_csv(csv_path: Path, json_path: Path, inplace: bool = False, backup: bool = True, dry_run: bool = True):
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    json_ranges = load_json_ranges(json_path)

    df = pd.read_csv(csv_path)
    initial_count = len(df)

    if json_ranges is not None:
        # Filter using JSON per-video ranges
        def row_in_range(row):
            video = row['filename']
            frame = int(row['frame_number'])
            if video not in json_ranges:
                # If no range for video, keep row (safe default)
                return True
            start, end = json_ranges[video]
            return start <= frame <= end

        mask = df.apply(row_in_range, axis=1)
        source = f"json:{json_path}"
    elif 'start_frame' in df.columns and 'end_frame' in df.columns:
        mask = (df['frame_number'] >= df['start_frame']) & (df['frame_number'] <= df['end_frame'])
        source = f"csv(start/end)"
    else:
        print(f"No JSON ranges ({json_path}) and CSV lacks start_frame/end_frame columns. Skipping {csv_path}.")
        return

    cleaned_df = df[mask]
    final_count = len(cleaned_df)
    removed_count = initial_count - final_count

    print(f"{csv_path}: source={source}  original={initial_count}  kept={final_count}  removed={removed_count}")

    if dry_run:
        print("Dry-run: no files written. Use --inplace or --out to save changes.")
        return

    # Write
    if inplace:
        if backup:
            bak = csv_path.with_suffix(csv_path.suffix + ".bak")
            shutil.copy2(csv_path, bak)
            print(f"Backup written to {bak}")
        cleaned_df.to_csv(csv_path, index=False)
        print(f"Wrote cleaned CSV in-place: {csv_path}")
    else:
        out = csv_path.parent / (csv_path.stem + ".cleaned" + csv_path.suffix)
        cleaned_df.to_csv(out, index=False)
        print(f"Wrote cleaned CSV to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CSV label files using authoritative video ranges")
    parser.add_argument("--csv", required=False, action='append', help="CSV file(s) to clean. If omitted, defaults to a few commonly used files.")
    parser.add_argument("--json", default="data/all_data.json", help="Path to all_data.json produced by extractor (preferred)")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original CSV file(s).")
    parser.add_argument("--backup", action="store_true", help="Create a .bak backup when --inplace is used")
    parser.add_argument("--out", required=False, help="Write cleaned CSV to this path (overrides default <csv>.cleaned.csv)")
    parser.add_argument("--execute", action="store_true", help="Actually write changes. Default is dry-run")

    args = parser.parse_args()

    json_path = Path(args.json)

    default_files = [
        Path("data/all_data.csv"),
        Path("data/labels/train_labels.csv"),
        Path("data/labels/val_labels.csv"),
        Path("data/labels/test_labels.csv"),
    ]

    csv_files = [Path(p) for p in args.csv] if args.csv else default_files

    for csv_path in csv_files:
        if args.out:
            # If a single out is specified, use it for the first CSV; otherwise fallback to <csv>.cleaned.csv
            out = Path(args.out)
            if len(csv_files) > 1:
                print("Warning: --out specified but multiple CSVs requested; using default .cleaned suffix for others")
                out = None
        else:
            out = None

        dry_run = not args.execute

        cleanup_csv(csv_path, json_path, inplace=args.inplace, backup=args.backup, dry_run=dry_run)

