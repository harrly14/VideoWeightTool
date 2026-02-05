#!/usr/bin/env python3
"""Dataset statistics for VideoWeightTool

Collects simple stats from label CSVs and checks for missing images.
This file was created and coded by Raptor Mini 
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from PIL import Image


def find_csvs(labels_dir: Path):
    candidates = sorted(labels_dir.glob("*.csv"))
    return candidates


def read_csv(path: Path):
    if pd is not None:
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
            return None
    # fallback to simple csv reader
    import csv
    rows = []
    try:
        with path.open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                rows.append(r)
        return rows
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}")
        return None


def compute_stats(labels_dir: Path, images_dir: Path, sample_images: int = 200):
    csvs = find_csvs(labels_dir)
    if not csvs:
        print(f"No CSV label files found in {labels_dir}")
        return None

    total_rows = 0
    per_file = {}
    combined = []

    for p in csvs:
        data = read_csv(p)
        if data is None:
            continue
        if pd is not None and isinstance(data, pd.DataFrame):
            rows = data.to_dict(orient="records")
            cnt = len(data)
        else:
            rows = data
            cnt = len(rows)
        total_rows += cnt
        per_file[p.name] = cnt
        combined.extend(rows)

    stats = {
        "total_rows": total_rows,
        "by_file": per_file,
        "unique_videos": None,
        "weight_stats": None,
        "missing_images": {"checked": 0, "missing": 0},
        "image_sizes_sampled": {},
    }

    # derive simple stats if columns exist
    if combined and isinstance(combined[0], dict):
        cols = set(combined[0].keys())
    else:
        cols = set()

    if "weight" in cols:
        try:
            vals = [float(r["weight"]) for r in combined if r.get("weight") not in (None, "")]
            if vals:
                import statistics
                stats["weight_stats"] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": statistics.mean(vals),
                    "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                }
        except Exception:
            pass

    if "filename" in cols and "frame_number" in cols:
        vids = set(r["filename"] for r in combined if r.get("filename") not in (None, ""))
        stats["unique_videos"] = len(vids)

        exts = [".jpg", ".jpeg", ".png"]
        sampled = 0
        for r in combined:
            fname = r.get("filename")
            frame = r.get("frame_number")
            if fname is None or frame in (None, ""):
                continue
            keybase = f"{fname}_{int(float(frame))}"
            stats["missing_images"]["checked"] += 1
            exists = False
            for e in exts:
                if (images_dir / f"{keybase}{e}").exists():
                    exists = True
                    if Image is not None and sampled < sample_images:
                        try:
                            p = images_dir / f"{keybase}{e}"
                            with Image.open(p) as im:
                                size = f"{im.width}x{im.height}"
                                stats["image_sizes_sampled"][size] = stats["image_sizes_sampled"].get(size, 0) + 1
                                sampled += 1
                        except Exception:
                            stats["image_sizes_sampled"]["unreadable"] = stats["image_sizes_sampled"].get("unreadable", 0) + 1
                    break
            if not exists:
                stats["missing_images"]["missing"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--labels-dir", default="data/labels", help="Directory containing label CSVs")
    parser.add_argument("--images-dir", default="data/images", help="Directory containing extracted images")
    parser.add_argument("--sample-images", type=int, default=200, help="How many images to sample for size distribution")
    parser.add_argument("--out", default=None, help="Write JSON summary to this file")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)

    stats = compute_stats(labels_dir, images_dir, args.sample_images)
    if stats is None:
        parser.exit(2)

    print("\nDATASET STATS\n" + "=" * 40)
    print(f"Total label rows: {stats['total_rows']}")
    print("By file:")
    for k, v in stats["by_file" ].items():
        print(f"  {k}: {v}")
    if stats["unique_videos"] is not None:
        print(f"Unique videos: {stats['unique_videos']}")
    if stats["weight_stats"]:
        w = stats["weight_stats"]
        print(f"Weight: min={w['min']:.3f} max={w['max']:.3f} mean={w['mean']:.3f} stdev={w['stdev']:.3f}")
    miss = stats["missing_images"]
    print(f"Missing images: {miss['missing']} / {miss['checked']} checked")

    if stats["image_sizes_sampled"]:
        print("Image sizes (sampled):")
        for size, cnt in stats["image_sizes_sampled"].items():
            print(f"  {size}: {cnt}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"Wrote JSON report to {outp}")


if __name__ == "__main__":
    main()
