import csv
from PyQt5.QtWidgets import QApplication, QInputDialog

if __name__ == '__main__':
   app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    QInputDialog.getText(self, 'Input Dialog', 'Enter some text:')


    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if use_existing_csv == QMessageBox.No:
            writer.writerow(["frame_num", "weight"])
            total_frames = int(scale_video.get(cv2.CAP_PROP_FRAME_COUNT))  
            for frame in range(total_frames):
                writer.writerow([frame, 0])