import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

if len(sys.argv) < 2:
    print("You must enter your desired video name as an argument")
    sys.exit(1)

vid_name = sys.argv[1]

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

video_filter = "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV);;All Files (*)"
video_path, _ = QFileDialog.getOpenFileName(
    None,
    "Select video file",
    os.getcwd(),
    video_filter
)

if not video_path:
    QMessageBox.information(None, "No file selected", "No video selected. Exiting...")
    sys.exit(0)

if not os.path.isfile(video_path):
    QMessageBox.critical(None, "File error", f"Selected file does not exist:\n{video_path}")
    sys.exit(1)

vid = cv2.VideoCapture(video_path)

if not vid.isOpened():
    QMessageBox.critical(None, "OpenCV error", f"Failed to open video:\n{video_path}")
    vid.release()
    sys.exit(1)

output_folder = "/home/sully/Desktop/projects/VideoWeightTool/data/val/images/"

if not output_folder:
    QMessageBox.information(None, "No folder selected", "No folder for the output files selected. Exiting...")
    sys.exit(0)

count, success = 0, True
while success:
    success, image = vid.read() 
    if success: 
        file_path = os.path.join(output_folder, f"{vid_name}_{count}.jpg")
        cv2.imwrite(file_path, image) 
        count += 1

vid.release()