from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt


class StartupDialog(QDialog):
    """Startup configuration dialog for batch labelling."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Labelling Tool - Configuration")
        self.setMinimumWidth(500)
        self.resize(600, 300)
        
        self.n_value = None
        self.video_dir = None
        self.csv_path = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Title and description
        title = QLabel("Video Batch Labelling Configuration")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        description = QLabel(
            "Configure the number of labels per video and select source/output locations.\n\n"
            "n: Number of frames to label per video.\n"
            "Frames will be evenly distributed throughout the valid frame range."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # n input
        n_layout = QHBoxLayout()
        n_label = QLabel("Labels per video (n):")
        self.n_input = QLineEdit()
        self.n_input.setPlaceholderText("e.g., 50")
        self.n_input.setText("50")
        n_layout.addWidget(n_label)
        n_layout.addWidget(self.n_input)
        layout.addLayout(n_layout)
        
        # Video directory picker
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
        
        # CSV file picker
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
        
        # Buttons
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
        """Open folder selection dialog for video directory."""
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
        """Open file selection dialog for CSV output."""
        csv_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select or create CSV file",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=QFileDialog.DontConfirmOverwrite  # Allow selecting existing file
        )
        if csv_path:
            self.csv_path = csv_path
            self.csv_display.setText(csv_path)
    
    def validate_and_accept(self):
        """Validate inputs and accept dialog."""
        # Validate n
        try:
            self.n_value = int(self.n_input.text().strip())
            if self.n_value <= 0:
                raise ValueError("n must be positive")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", f"Invalid value for n: {e}")
            return
        
        # Validate video directory
        if not self.video_dir:
            QMessageBox.warning(self, "Missing selection", "Please select a video folder")
            return
        
        # Validate CSV path
        if not self.csv_path:
            QMessageBox.warning(self, "Missing selection", "Please select or specify a CSV file path")
            return
        
        self.accept()
    
    def get_config(self) -> tuple[int | None, str | None, str | None]:
        """Return configuration values."""
        return self.n_value, self.video_dir, self.csv_path
