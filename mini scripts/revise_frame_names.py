import os
import sys
import re

if len(sys.argv) < 3:
    print("You must input a file directory and vid name")
    exit(1)

vid_dir = sys.argv[1]
vid_name = sys.argv[2]

for file in os.listdir(vid_dir):
    file_path = os.path.join(vid_dir, file)
    if not os.path.isfile(file_path):
        continue
    m = re.search(r'frame(\d+)\.jpg$', file)
    if not m:
        continue
    num = m.group(1)  # preserves leading zeros
    new_fname = f"{vid_name}{num}.jpg"
    dst_path = os.path.join(vid_dir, new_fname)
    if os.path.exists(dst_path):
        print(f"Skipping {file}: target exists {new_fname}")
        continue
    os.rename(file_path, dst_path)
    print(f"Renamed {file} -> {new_fname}")