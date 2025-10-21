import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)
import csv
import cv2
import sys
import subprocess
import re
import ffmpeg
from datetime import datetime
from Window import EditWindow
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

print(ffmpeg.__file__)

def apply_edits_and_save(video_path, video_params, output_folder, progress_callback=None):
    file_name = "tmp_edited_video.mp4"
    output_path = os.path.join(output_folder, file_name)
    if os.path.exists(output_path): 
        os.remove(output_path)

    if progress_callback:
        progress_callback(0)
    
    stream = ffmpeg.input(video_path)
    probe = ffmpeg.probe(video_path)

    if video_params.trim_end is not None:
        start_frame = video_params.trim_start
        end_frame = video_params.trim_end
        
        probe = ffmpeg.probe(video_path)
        fps = eval(probe['streams'][0]['r_frame_rate'])

        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time

        # setpts updates the timestamps
        stream = stream.trim(start=start_time, end=end_time).setpts('PTS-STARTPTS') 
    else: 
        duration = float(probe['streams'][0]['duration'])

    if video_params.crop_coords: 
        x, y, w, h = video_params.crop_coords
        stream = ffmpeg.filter(stream, 'crop', w, h, x, y)

    # Convert integer values to floats for ffmpeg
    # Brightness: -255 to 255 range to -1 to 1 range
    # Saturation: 0-300 integer to 0-3 float (100 = 1.0)
    # Contrast: 0-200 integer to 0-2 float (100 = 1.0)
    adj_brightness = video_params.brightness / 255
    adj_saturation = video_params.saturation / 100
    adj_contrast = video_params.contrast / 100

    stream = ffmpeg.filter(stream, 'eq', 
                           brightness=adj_brightness,
                           contrast=adj_contrast,
                           saturation=adj_saturation)

    stream = ffmpeg.output(stream, output_path,
                           vcodec='libx264', 
                           acodec='aac', 
                           preset='ultrafast', # faster compression, slightly larger files
                           crf=25,
                           threads=0)

    try:
        cmd = ffmpeg.compile(stream, overwrite_output=True)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        for line in process.stderr: # type: ignore
            if progress_callback:
                time_match = re.search(r'time=(\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
                if time_match:
                    hours, minutes, seconds = time_match.groups()
                    current_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    progress_percent = int((current_time / duration) * 100)
                    progress_percent = min(progress_percent, 99)

                    should_continue = progress_callback(progress_percent)
                    if not should_continue:
                        process.terminate()
                        process.wait()
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        return None
        process.wait()

        if process.returncode != 0:
            QMessageBox.warning(None, "Error", f"FFmpeg error: return code {process.returncode}")
            return None

        if progress_callback:
            progress_callback(100)

        # rename temp file
        try:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            final_name = f"{base_name}_edited.mp4"
            final_path = os.path.join(output_folder,final_name)
            os.replace(output_path, final_path)
            return final_path
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Could not rename output file: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
    except Exception as e:
        QMessageBox.warning(None, "Error", f"Error processing video: {e}")
        return None

def launch_editing_window(video, output_csv):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = EditWindow(video, output_csv)
    window.show()
    app.exec_()

    return window.video_params

if __name__ == "__main__":
    #ensure a QApplication exists before making any QWidgets
    app = QApplication.instance() or QApplication(sys.argv)

    # choose file
    options = QFileDialog.Options()

    video_filter = "Video Files (*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV);;All Files (*)"
    video_path, selected_filter = QFileDialog.getOpenFileName(
        None,
        "Select video file",
        os.getcwd(),
        video_filter,
        options=options
    )
    if not video_path:
        QMessageBox.information(None, "No file selected", "No video selected. Exiting...")
        sys.exit(0)

    scale_video = cv2.VideoCapture(video_path)

    if not scale_video.isOpened():
        QMessageBox.critical(None, "Error", "Cannot open the video file.")
        scale_video.release()
        cv2.destroyAllWindows()
        sys.exit(1)

    # choose output directory
    base_outputs_dir = "outputs"
    os.makedirs(base_outputs_dir, exist_ok=True)
    default_dir = os.path.join(os.getcwd(), base_outputs_dir)
    output_parent = QFileDialog.getExistingDirectory(None, "Select, output folder", default_dir,
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)

    if not output_parent:
        QMessageBox.information(None, "No folder selected", "No folder for the output files selected. Exiting...")
        sys.exit(0)    

    video_base = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(output_parent, f"{video_base}_{timestamp}")

    os.makedirs(output_folder, exist_ok=True)

    updated_params = None
    output_csv = os.path.join(output_folder, "weights.csv")

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_num", "weight"])

        total_frames = int(scale_video.get(cv2.CAP_PROP_FRAME_COUNT))  
        for frame in range(total_frames):
            writer.writerow([frame, 0])
        updated_params = launch_editing_window(scale_video, output_csv)
        
    if updated_params.trim_end is not None:
        start_row = updated_params.trim_start + 1
        end_row = updated_params.trim_end + 1
        
        backup_csv = output_csv + 'bak'
        os.rename(output_csv, backup_csv)

        try:
            with open(backup_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)

                for i, row in enumerate(reader):
                    if i == 0: # keep header row
                        writer.writerow(row)
                    elif start_row <= i <= end_row:
                        writer.writerow(row)
            
            os.remove(backup_csv)

        except Exception as e:
            if os.path.exists(backup_csv):
                os.rename(backup_csv, output_csv)
            QMessageBox.warning(None, "Error", f"An error occurred: {e}")


    save_y_n = QMessageBox.question(None, 'Message', 'Would you like  to save the video with your edits?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

    if save_y_n == QMessageBox.Yes:
        if updated_params is None:
            QMessageBox.warning(None, "No edits", "No edit params returned. Not saving.")
        else: 
            progress = QProgressDialog("Processing video...", "Cancel", 0, 100)
            progress.setWindowTitle("Saving video")
            progress.setMinimumDuration(0)
            progress.setValue(0)

            progress.show()
            QApplication.processEvents()
            # a better approach here would be to use a worker thread, but i dont know how to do that so this is what ya get
            def update_progress(value):
                progress.setValue(value)
                progress.setLabelText(f"Processing video... {value}%")
                QApplication.processEvents()
                return not progress.wasCanceled()
            
            edited_video_path = apply_edits_and_save(video_path, updated_params, output_folder, progress_callback=update_progress)

            progress.close()

            if edited_video_path is not None:
                QMessageBox.information(None, "Success", 
                    f"The updated video has been written to:\n{edited_video_path}")
            else:
                QMessageBox.warning(None, "Failed", "Failed to save the updated video.")
    else: 
        QMessageBox.information(None, "Not saved", f"Changes were not saved. Your csv can still be found in {output_csv}")

    try:
        if scale_video is not None:
            scale_video.release()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass