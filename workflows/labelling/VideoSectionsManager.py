"""
Video Sections Manager - Coordinates definition of multiple ROI sections per video.
Allows users to define multiple ROI+frame-range pairs until the video is fully covered
or they choose to proceed with partial coverage.
"""

import cv2
from typing import Optional, List, Dict, Tuple
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QMessageBox, QListWidget, QListWidgetItem, QFrame
)
from PyQt5.QtCore import Qt

from workflows.labelling.ROISelectionDialog import ROISelectionDialog


class VideoSectionsManager(QDialog):
    """
    Manager dialog for defining multiple ROI sections for a single video.
    Coordinates the creation of sections and tracks coverage.
    """
    
    def __init__(self, video_capture, video_filename: str, 
                 total_frames: int,
                 default_roi: Optional[List[Tuple[int, int]]] = None,
                 parent=None):
        """
        Initialize video sections manager.
        
        Args:
            video_capture: cv2.VideoCapture object (already opened)
            video_filename: Name of video file for display
            total_frames: Total number of frames in video
            default_roi: Previous ROI to use as default for first section
            parent: Parent widget
        """
        super().__init__(parent)
        self.video = video_capture
        self.video_filename = video_filename
        self.total_frames = total_frames
        self.default_roi = default_roi
        
        # List of defined sections: [{'quad': [...], 'start_frame': int, 'end_frame': int}, ...]
        self.sections: List[Dict] = []
        
        # Track the last ROI for carryover to next section
        self.last_roi = default_roi
        
        self.setWindowTitle(f"Define ROI Sections - {video_filename}")
        self.setMinimumSize(500, 400)
        self.resize(600, 500)
        
        self.setup_ui()
        self._update_display()
    
    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()
        
        # Header
        header_label = QLabel(
            f"<b>{self.video_filename}</b><br>"
            f"Total frames: 0 - {self.total_frames - 1} ({self.total_frames} frames)"
        )
        header_label.setStyleSheet("font-size: 13px; padding: 5px;")
        layout.addWidget(header_label)
        
        # Instructions
        instructions = QLabel(
            "Define one or more ROI sections. Each section specifies an ROI (region of interest) "
            "and the frame range where that ROI is valid. Sections cannot overlap."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(instructions)
        
        # Sections list
        sections_group = QGroupBox("Defined Sections")
        sections_layout = QVBoxLayout()
        
        self.sections_list = QListWidget()
        self.sections_list.setMinimumHeight(150)
        sections_layout.addWidget(self.sections_list)
        
        # Section buttons
        section_btn_layout = QHBoxLayout()
        
        self.add_section_btn = QPushButton("Add Section")
        self.add_section_btn.setStyleSheet("background-color: #4a7a4a; color: white;")
        self.add_section_btn.clicked.connect(self._add_section)
        section_btn_layout.addWidget(self.add_section_btn)
        
        self.remove_section_btn = QPushButton("Remove Selected")
        self.remove_section_btn.clicked.connect(self._remove_section)
        self.remove_section_btn.setEnabled(False)
        section_btn_layout.addWidget(self.remove_section_btn)
        
        section_btn_layout.addStretch()
        sections_layout.addLayout(section_btn_layout)
        
        sections_group.setLayout(sections_layout)
        layout.addWidget(sections_group)
        
        # Coverage info
        coverage_group = QGroupBox("Coverage")
        coverage_layout = QVBoxLayout()
        
        self.coverage_label = QLabel()
        self.coverage_label.setWordWrap(True)
        coverage_layout.addWidget(self.coverage_label)
        
        self.uncovered_label = QLabel()
        self.uncovered_label.setWordWrap(True)
        self.uncovered_label.setStyleSheet("color: #a00;")
        coverage_layout.addWidget(self.uncovered_label)
        
        coverage_group.setLayout(coverage_layout)
        layout.addWidget(coverage_group)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        button_layout.addStretch()
        
        self.proceed_button = QPushButton("Proceed with Current Sections")
        self.proceed_button.setStyleSheet("background-color: #4a7a4a; color: white; padding: 8px 16px;")
        self.proceed_button.clicked.connect(self._on_proceed)
        self.proceed_button.setEnabled(False)
        button_layout.addWidget(self.proceed_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # Connect selection change
        self.sections_list.itemSelectionChanged.connect(self._on_selection_changed)
    
    def _get_uncovered_ranges(self) -> List[Tuple[int, int]]:
        """
        Calculate uncovered frame ranges.
        
        Returns:
            List of (start, end) tuples for uncovered ranges
        """
        if not self.sections:
            return [(0, self.total_frames - 1)]
        
        # Sort sections by start frame
        sorted_sections = sorted(self.sections, key=lambda s: s['start_frame'])
        
        uncovered = []
        current_pos = 0
        
        for section in sorted_sections:
            if section['start_frame'] > current_pos:
                uncovered.append((current_pos, section['start_frame'] - 1))
            current_pos = max(current_pos, section['end_frame'] + 1)
        
        if current_pos < self.total_frames:
            uncovered.append((current_pos, self.total_frames - 1))
        
        return uncovered
    
    def _get_next_available_range(self) -> Optional[Tuple[int, int]]:
        """
        Get the next available frame range for a new section.
        
        Returns:
            (start, end) tuple or None if fully covered
        """
        uncovered = self._get_uncovered_ranges()
        if uncovered:
            return uncovered[0]
        return None
    
    def _get_total_covered_frames(self) -> int:
        """Get total number of covered frames."""
        return sum(s['end_frame'] - s['start_frame'] + 1 for s in self.sections)
    
    def _update_display(self):
        """Update the sections list and coverage info."""
        # Update sections list
        self.sections_list.clear()
        for i, section in enumerate(self.sections):
            frame_count = section['end_frame'] - section['start_frame'] + 1
            item_text = (
                f"Section {i+1}: Frames {section['start_frame']} - {section['end_frame']} "
                f"({frame_count} frames)"
            )
            item = QListWidgetItem(item_text)
            self.sections_list.addItem(item)
        
        # Update coverage info
        covered = self._get_total_covered_frames()
        coverage_pct = (covered / self.total_frames * 100) if self.total_frames > 0 else 0
        self.coverage_label.setText(
            f"Covered: {covered} / {self.total_frames} frames ({coverage_pct:.1f}%)"
        )
        
        # Update uncovered info
        uncovered = self._get_uncovered_ranges()
        if uncovered:
            uncovered_text = "Uncovered ranges: " + ", ".join([
                f"{start}-{end}" for start, end in uncovered
            ])
            self.uncovered_label.setText(uncovered_text)
            self.uncovered_label.show()
        else:
            self.uncovered_label.setText("All frames covered!")
            self.uncovered_label.setStyleSheet("color: #0a0;")
        
        # Update button states
        self.proceed_button.setEnabled(len(self.sections) > 0)
        
        # Check if there's room for more sections
        next_range = self._get_next_available_range()
        self.add_section_btn.setEnabled(next_range is not None)
        if next_range is None:
            self.add_section_btn.setText("All Frames Covered")
        else:
            self.add_section_btn.setText("Add Section")
    
    def _on_selection_changed(self):
        """Handle section selection change."""
        self.remove_section_btn.setEnabled(
            len(self.sections_list.selectedItems()) > 0
        )
    
    def _add_section(self):
        """Open dialog to add a new section."""
        next_range = self._get_next_available_range()
        if next_range is None:
            QMessageBox.information(
                self,
                "Fully Covered",
                "All frames are already covered by existing sections."
            )
            return
        
        available_start, available_end = next_range
        
        # Reset video position
        self.video.set(cv2.CAP_PROP_POS_FRAMES, available_start)

        
        
        dialog = ROISelectionDialog(
            self.video,
            self.video_filename,
            available_start,
            available_end,
            default_roi=self.last_roi,
            section_number=len(self.sections) + 1,
            existing_sections=self.sections,
            parent=self
        )
        
        if dialog.exec_() == QDialog.Accepted:
            section = dialog.get_section()
            if section:
                self.sections.append(section)
                self.last_roi = section['quad']  # Carryover for next section
                self._update_display()
                
    def _remove_section(self):
        """Remove the selected section."""
        selected = self.sections_list.selectedItems()
        if not selected:
            return
        
        index = self.sections_list.row(selected[0])
        if 0 <= index < len(self.sections):
            del self.sections[index]
            self._update_display()
    
    def _on_proceed(self):
        """Confirm and proceed with current sections."""
        if not self.sections:
            QMessageBox.warning(
                self,
                "No Sections",
                "Please define at least one ROI section before proceeding."
            )
            return
        
        # Warn if there are uncovered ranges
        uncovered = self._get_uncovered_ranges()
        if uncovered:
            total_uncovered = sum(end - start + 1 for start, end in uncovered)
            uncovered_text = ", ".join([f"{start}-{end}" for start, end in uncovered])
            
            reply = QMessageBox.question(
                self,
                "Uncovered Frames",
                f"There are {total_uncovered} uncovered frames that will be skipped:\n"
                f"{uncovered_text}\n\n"
                "Proceed anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        self.accept()
    
    def get_sections(self) -> List[Dict]:
        """
        Get all defined sections.
        
        Returns:
            List of section dicts with 'quad', 'start_frame', 'end_frame'
        """
        return self.sections
    
    def get_last_roi(self) -> Optional[List[Tuple[int, int]]]:
        """Get the last used ROI for carryover to next video."""
        return self.last_roi
    
    def get_covered_frame_ranges(self) -> List[Tuple[int, int]]:
        """
        Get list of covered frame ranges (for frame sampling).
        
        Returns:
            List of (start, end) tuples
        """
        return [(s['start_frame'], s['end_frame']) for s in self.sections]
