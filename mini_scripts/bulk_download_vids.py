#!/usr/bin/env python3
"""
Download one mp4 file from each scale directory in Swarm Assembly 2025/S02/
"""

import os
import shutil
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Source and destination directories
SOURCE_BASE = Path("/home/sully/Shares/peleg-group-1/Swarm Assembly 2025/S02")
DEST_DIR = Path.home() / "Downloads" / "scale_videos"

# Create destination directory if it doesn't exist
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Sanity checks
print(f"Source base: {SOURCE_BASE}")
print(f"Source exists: {SOURCE_BASE.exists()}")
print(f"Destination: {DEST_DIR}")
print(f"Destination writable: {os.access(DEST_DIR, os.W_OK)}")
print()

print(f"Scanning {SOURCE_BASE} for scale directories...")
print(f"Destination: {DEST_DIR}\n")

# Progress tracking globals
print_lock = Lock()
file_lines = {}  # Track which line each file is on
total_files = 0

def update_progress(filename, status):
    """Thread-safe progress update for a specific file."""
    with print_lock:
        if filename not in file_lines:
            return
        line_num = file_lines[filename]
        lines_up = total_files - line_num
        # Move cursor up, clear line, print, move back down
        sys.stdout.write(f"\033[{lines_up}A")  # Move up
        sys.stdout.write(f"\033[2K")            # Clear line
        sys.stdout.write(f"{status}\n")         # Print status
        sys.stdout.write(f"\033[{lines_up - 1}B")  # Move back down
        sys.stdout.flush()

def copy_file(source_file, dest_file, dated_folder_name):
    """Copy a single file and return status"""
    filename = source_file.name
    
    # Check if file already exists
    if dest_file.exists():
        update_progress(filename, f"[{filename[:30]}] Skipped (already exists)")
        return 'skipped'
    
    # Copy the file
    try:
        update_progress(filename, f"[{filename[:30]}] Copying...")
        shutil.copy2(source_file, dest_file)
        update_progress(filename, f"[{filename[:30]}] Completed")
        return 'copied'
    except Exception as e:
        update_progress(filename, f"[{filename[:30]}] Failed: {str(e)[:50]}")
        return 'failed'

# Iterate through dated subdirectories (0728, 0729, etc.)
# Collect files to copy
files_to_copy = []

for dated_folder in sorted(SOURCE_BASE.iterdir()):
    if not dated_folder.is_dir():
        continue
    
    scale_dir = dated_folder / "scale"
    
    if not scale_dir.exists():
        print(f"[SKIP] {dated_folder.name}: no 'scale' subdirectory")
        continue
    
    # Debug: check if scale_dir resolves correctly
    print(f"[DEBUG] {dated_folder.name}: checking scale dir: {scale_dir} (exists: {scale_dir.exists()})")
    
    # Find first mp4 file in the scale directory
    # Check both .mp4 and .MP4 extensions (case-insensitive)
    mp4_files = list(scale_dir.glob("*.mp4")) + list(scale_dir.glob("*.MP4"))
    
    # Debug: show what files we found
    if not mp4_files:
        all_files = list(scale_dir.glob("*"))
        print(f"[DEBUG] {dated_folder.name}: found files: {[f.name for f in all_files]}")
        print(f"[SKIP] {dated_folder.name}: no mp4/MP4 files in scale directory")
        continue
    
    # Use first mp4 file (filesystem order)
    source_file = mp4_files[0]
    dest_file = DEST_DIR / source_file.name
    
    files_to_copy.append((source_file, dest_file, dated_folder.name))

# Initialize progress display
total_files = len(files_to_copy)
print(f"\nProcessing {total_files} files...\n")

if total_files == 0:
    print("No files to copy. Exiting.")
    exit(0)

for i, (source_file, dest_file, folder_name) in enumerate(files_to_copy):
    filename = source_file.name
    file_lines[filename] = i
    print(f"[{filename[:30]}] Waiting...")

# Copy files in parallel (4 concurrent copies)
print(f"\nStarting parallel copy with 4 workers...\n")
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(copy_file, source, dest, folder_name)
        for source, dest, folder_name in files_to_copy
    ]
    
    # Collect results and count them
    results = []
    for future in as_completed(futures):
        result = future.result()
        results.append(result)

# Count results
copied_count = results.count('copied')
skipped_count = results.count('skipped')
failed_count = results.count('failed')

print(f"\n--- Summary ---")
print(f"Copied: {copied_count}")
print(f"Skipped (already exist): {skipped_count}")
print(f"Failed: {failed_count}")
