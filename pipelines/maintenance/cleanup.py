#!/usr/bin/env python3
"""Cleanup label CSVs by removing rows that reference missing images.

This script is intentionally conservative: by default it performs a dry-run and prints
how many rows would be removed. Use --execute and optionally --inplace to apply changes.
This file was created and coded by Raptor Mini 
"""

import argparse
import csv
import shutil
from pathlib import Path


def find_csvs(labels_dir: Path):
    return sorted(labels_dir.glob("*.csv"))


def read_csv_rows(path: Path):
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return reader.fieldnames, rows


def filter_rows(rows, images_dir: Path, exts):
    kept = []
    removed = 0
    for r in rows:
        fname = r.get("filename")
        frame = r.get("frame_number")
        if not fname or frame in (None, ""):
            # keep rows we can't validate
            kept.append(r)
            continue
        base = f"{fname}_{int(float(frame))}"
        exists = any((images_dir / f"{base}{e}").exists() for e in exts)
        if exists:
            kept.append(r)
        else:
            removed += 1
    return kept, removed


def write_csv(path: Path, fieldnames, rows):
    tmp = path.with_suffix(path.suffix + ".cleaned")
    with tmp.open("w", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return tmp


def main():
    parser = argparse.ArgumentParser(description="Cleanup label CSVs by removing rows referencing missing images")
    parser.add_argument("--labels-dir", default="data/labels", help="Directory containing label CSVs")
    parser.add_argument("--images-dir", default="data/images", help="Directory containing extracted images")
    parser.add_argument("--exts", default="jpg,jpeg,png", help="Image extensions to look for (comma-separated)")
    parser.add_argument("--execute", action="store_true", help="Apply changes (by default do a dry-run)")
    parser.add_argument("--inplace", action="store_true", help="When used with --execute, overwrite original CSVs (creates .bak if --backup used)")
    parser.add_argument("--backup", action="store_true", help="Create a .bak copy when using --inplace")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    exts = ["." + e.strip().lstrip('.') for e in args.exts.split(",") if e.strip()]

    csvs = find_csvs(labels_dir)
    if not csvs:
        print(f"No CSVs found in {labels_dir}")
        return 2

    total_removed = 0
    for p in csvs:
        fieldnames, rows = read_csv_rows(p)
        kept, removed = filter_rows(rows, images_dir, exts)
        total_removed += removed
        print(f"{p.name}: {removed} rows would be removed (out of {len(rows)})")
        if args.execute and removed > 0:
            cleaned_tmp = write_csv(p, fieldnames, kept)
            if args.inplace:
                if args.backup:
                    bak = p.with_suffix(p.suffix + ".bak")
                    shutil.copy2(p, bak)
                shutil.move(str(cleaned_tmp), str(p))
                print(f"Overwrote {p} with cleaned data")
            else:
                outp = p.with_suffix(p.suffix + ".cleaned.csv")
                shutil.move(str(cleaned_tmp), str(outp))
                print(f"Wrote cleaned CSV to {outp}")

    print(f"Total rows removed (or would be removed without --execute): {total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
