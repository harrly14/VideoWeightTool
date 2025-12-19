"""mini_scripts/cleanup_images.py

Improved cleanup script:
- Prefers `data/all_data.json` for authoritative video ranges (video_ranges.min_frame/max_frame)
- Falls back to CSV (`start_frame`/`end_frame` or inferred min/max `frame_number` per video)
- Dry-run by default; use --execute to actually delete/move files
- Supports moving removed files to a safe directory instead of deleting
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

VIDEO_FRAME_RE = re.compile(r"(?P<video>.+)_(?P<frame>\d+)\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE)


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


def load_csv_ranges(csv_path: Path) -> Optional[Dict[str, Tuple[int, int]]]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)

    # Prefer explicit per-row start_frame/end_frame columns if present
    if "start_frame" in df.columns and "end_frame" in df.columns:
        ranges = {}
        grouped = df.groupby("filename")
        for name, g in grouped:
            start = int(g["start_frame"].mode().iloc[0]) if not g["start_frame"].isnull().all() else int(g["start_frame"].min())
            end = int(g["end_frame"].mode().iloc[0]) if not g["end_frame"].isnull().all() else int(g["end_frame"].max())
            ranges[name] = (start, end)
        return ranges

    # Fallback: infer min/max frame number from labeled rows
    if "frame_number" in df.columns:
        ranges = {}
        grouped = df.groupby("filename")
        for name, g in grouped:
            start = int(g["frame_number"].min())
            end = int(g["frame_number"].max())
            ranges[name] = (start, end)
        return ranges

    return None


def main():
    parser = argparse.ArgumentParser(description="Cleanup image files outside per-video ranges")
    parser.add_argument("--csv", required=True, help="Path to labels CSV used to identify videos (e.g., data/labels/train_labels.csv)")
    parser.add_argument("--images-dir", default="data/images", help="Directory containing extracted images")
    parser.add_argument("--json", default="data/all_data.json", help="Path to all_data.json produced by extractor (preferred)")
    parser.add_argument("--exts", default="jpg,jpeg,png", help="Comma-separated list of image extensions to consider")
    parser.add_argument("--execute", action="store_true", help="Actually delete/move files; default is dry-run")
    parser.add_argument("--move-to", default=None, help="If set and --execute, move removed files to this directory instead of deleting")
    parser.add_argument("--delete-unknown", action="store_true", help="Also delete images whose video is not present in ranges (be careful)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_dir = Path(args.images_dir)
    json_path = Path(args.json)

    exts = {e.lower() for e in args.exts.split(",")}

    json_ranges = load_json_ranges(json_path)
    if json_ranges is not None:
        ranges = json_ranges
        source = f"json:{json_path}"
    else:
        csv_ranges = load_csv_ranges(csv_path)
        if csv_ranges is None:
            print("Could not determine video ranges from JSON or CSV. Aborting.")
            return
        ranges = csv_ranges
        source = f"csv:{csv_path}"

    total = 0
    candidates = []  # list of Path
    to_remove = []

    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        total += 1
        m = VIDEO_FRAME_RE.match(p.name)
        if not m:
            # filename doesn't match expected pattern; keep by default
            if args.verbose:
                print(f"SKIP (no-match): {p.name}")
            continue
        video = m.group("video")
        frame = int(m.group("frame"))
        ext = m.group("ext").lower()
        if ext not in exts:
            continue

        candidates.append((p, video, frame))

        if video in ranges:
            start, end = ranges[video]
            if frame < start or frame > end:
                to_remove.append((p, video, frame, start, end))
        else:
            # Unknown video: delete only if --delete-unknown
            if args.delete_unknown:
                to_remove.append((p, video, frame, None, None))

    print(f"Scanned {total} files in {images_dir}. Considered {len(candidates)} image files matching pattern.")
    print(f"Found {len(to_remove)} files outside declared ranges (source={source}).")

    if len(to_remove) == 0:
        print("Nothing to do.")
        return

    if not args.execute:
        print("Dry-run mode (use --execute to remove/move files). Example files to remove:")
        for p, video, frame, start, end in to_remove[:20]:
            if start is None:
                print(f"  {p.name}  (unknown video)")
            else:
                print(f"  {p.name}  (frame {frame} outside {start}-{end})")
        print("...")
        return

    # Execute removals
    if args.move_to:
        move_to = Path(args.move_to)
        move_to.mkdir(parents=True, exist_ok=True)
    deleted = 0
    moved = 0
    for p, video, frame, start, end in to_remove:
        try:
            if args.move_to:
                dest = move_to / p.name
                shutil.move(str(p), str(dest))
                moved += 1
                if args.verbose:
                    print(f"MOVED {p} -> {dest}")
            else:
                p.unlink()
                deleted += 1
                if args.verbose:
                    print(f"DELETED {p}")
        except Exception as e:
            print(f"Error removing {p}: {e}")

    print(f"Done. Deleted: {deleted}, Moved: {moved}.")
