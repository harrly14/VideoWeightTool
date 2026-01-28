from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

class StartupDialog(QDialog):
    """GUI to set up main.py arguments if they are missing"""
    
    def __init__(self, parent=None, rois_only: bool = False):
        super().__init__(parent)
        self.rois_only = rois_only
        self.setWindowTitle("Video Labelling Tool - Configuration")
        self.setMinimumWidth(500)
        self.resize(600, 300)
        
        self.num_frames_per_video = None
        self.video_dir = None
        self.csv_path = None
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("Video Batch Labelling Configuration")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        description = QLabel(
            "Configure the number of labels per video and select source/output locations.\n\n"
            "Frames will be evenly distributed throughout the valid frame range."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        num_frames_layout = QHBoxLayout()
        num_frames_label = QLabel("Labels per video:")
        self.num_frames_input = QLineEdit()
        self.num_frames_input.setPlaceholderText("e.g., 50")
        self.num_frames_input.setText("50")
        num_frames_layout.addWidget(num_frames_label)
        num_frames_layout.addWidget(self.num_frames_input)
        layout.addLayout(num_frames_layout)
        
        video_dir_layout = QHBoxLayout()
        video_dir_label = QLabel("Video folder:")
        self.video_dir_display = QLineEdit()
        self.video_dir_display.setReadOnly(True)
        self.video_dir_button = QPushButton("Browse...")
        self.video_dir_button.clicked.connect(self.pick_video_dir)
        video_dir_layout.addWidget(video_dir_label)
        video_dir_layout.addWidget(self.video_dir_display)
        video_dir_layout.addWidget(self.video_dir_button)
        layout.addLayout(video_dir_layout)
        
        csv_layout = QHBoxLayout()
        csv_label = QLabel("Output CSV:")
        self.csv_display = QLineEdit()
        self.csv_display.setReadOnly(True)
        self.csv_button = QPushButton("Browse...")
        self.csv_button.clicked.connect(self.pick_csv_file)
        csv_layout.addWidget(csv_label)
        csv_layout.addWidget(self.csv_display)
        csv_layout.addWidget(self.csv_button)
        layout.addLayout(csv_layout)
        
        # If running in rois-only mode, disable CSV selection
        if self.rois_only:
            self.csv_display.setText("N/A (rois-only mode)")
            self.csv_display.setReadOnly(True)
            self.csv_button.setEnabled(False)
        
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Start")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def pick_video_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select video folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.video_dir = folder
            self.video_dir_display.setText(folder)
    
    def pick_csv_file(self):
        csv_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select or create CSV file",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=QFileDialog.DontConfirmOverwrite  # Allow selecting existing file
        )
        if csv_path:
            if not csv_path.lower().endswith('.csv'):
                csv_path += '.csv'
            self.csv_path = csv_path
            self.csv_display.setText(csv_path)
    
    def validate_and_accept(self):
        if not self.rois_only:
            try:
                self.num_frames_per_video = int(self.num_frames_input.text().strip())
                if self.num_frames_per_video <= 0:
                    raise ValueError("must be positive")
            except ValueError as e:
                QMessageBox.warning(self, "Invalid input", f"Invalid value for labels per video: {e}")
                return
        else:
            # In rois-only mode, labels per video is optional
            try:
                val = self.num_frames_input.text().strip()
                self.num_frames_per_video = int(val) if val else None
            except ValueError:
                QMessageBox.warning(self, "Invalid input", "Invalid value for labels per video")
                return
        
        if not self.video_dir:
            QMessageBox.warning(self, "Missing selection", "Please select a video folder")
            return
        
        if not self.csv_path and not self.rois_only:
            QMessageBox.warning(self, "Missing selection", "Please select or specify a CSV file path")
            return
        
        self.accept()
    
    def get_config(self) -> tuple[int | None, str | None, str | None]:
        return self.num_frames_per_video, self.video_dir, self.csv_path
