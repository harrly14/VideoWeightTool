import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)
import cv2
import sys
import re

output_folder = "data/images"
os.makedirs(output_folder, exist_ok=True)

vid_dir = "videos"

for filename in os.listdir(vid_dir):
    match = re.search(r'(071\d_0\d)', filename)
    if not match:
        print(f"Skipping {filename}: does not match expected pattern")
        continue
    vid_name = match.group(1)

    vid = cv2.VideoCapture(os.path.join(vid_dir, filename))

    if not vid.isOpened():
        print(f"Error: Failed to open video:\n{filename}")
        vid.release()
        sys.exit(1)

    count, success = 0, True
    while success:
        success, image = vid.read() 
        if success: 
            file_path = os.path.join(output_folder, f"{vid_name}_{count}.jpg")
            cv2.imwrite(file_path, image) 
            count += 1

    vid.release()