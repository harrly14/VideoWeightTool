"""
Divider Dialog - Allows users to place three vertical divider lines on a warped ROI preview
to partition the digit display into four regions (ones, tenths, hundredths, thousandths).
"""

import cv2
import numpy as np
from typing import Optional, List
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

from core.config import CNN_WIDTH, CNN_HEIGHT, NUM_DIVIDERS

DIVIDER_COLOR = QColor(0, 255, 0)
DIVIDER_COLOR_ALPHA = QColor(0, 255, 0, 120)
MIN_DIVIDER_SPACING = 3

# Pixel radius for grabbing a divider line (in display/scaled space)
GRAB_RADIUS = 10

DEFAULT_SCALE = 3


class DividerCanvas(QWidget):
    """
    Custom widget that displays a warped ROI image and allows the user
    to drag three vertical divider lines.
    """

    def __init__(self, bgr_image: np.ndarray, default_dividers: Optional[List[int]] = None, parent=None):
        """
        Args:
            bgr_image: The warped CLAHE image in BGR format, shape (CNN_HEIGHT, CNN_WIDTH, 3).
            default_dividers: Optional list of NUM_DIVIDERS x-coordinates in canvas space (0..CNN_WIDTH).
        """
        super().__init__(parent)

        self.source_width = bgr_image.shape[1]  # CNN_WIDTH
        self.source_height = bgr_image.shape[0]  # CNN_HEIGHT

        # Convert BGR -> RGB -> QImage -> QPixmap
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        self._source_pixmap = QPixmap.fromImage(
            QImage(rgb.data, w, h, c * w, QImage.Format_RGB888).copy()
        )

        # Divider positions in source (canvas) coordinates
        if default_dividers and len(default_dividers) == NUM_DIVIDERS:
            self.dividers = sorted(list(default_dividers))
        else:
            # default (evenly spaced) dividers
            step = self.source_width / (NUM_DIVIDERS + 1)
            self.dividers = [int(round(i * step)) for i in range(1, NUM_DIVIDERS + 1)]

        self._clamp_dividers()

        self._dragging_index: Optional[int] = None

        self.setMinimumSize(self.source_width * 2, self.source_height * 2)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def get_dividers(self) -> List[int]:
        return sorted(self.dividers)

    # ------------------------------------------------------------------ geometry helpers
    def _display_scale(self) -> float:
        """Current uniform scale factor from source to display."""
        sx = self.width() / self.source_width
        sy = self.height() / self.source_height
        return min(sx, sy)

    def _display_offset(self) -> tuple:
        scale = self._display_scale()
        dw = scale * self.source_width
        dh = scale * self.source_height
        return ((self.width() - dw) / 2, (self.height() - dh) / 2)

    def _source_x_to_display(self, sx: float) -> float:
        scale = self._display_scale()
        ox, _ = self._display_offset()
        return ox + sx * scale

    def _display_x_to_source(self, dx: float) -> float:
        scale = self._display_scale()
        ox, _ = self._display_offset()
        return (dx - ox) / scale

    # ------------------------------------------------------------------ divider clamping
    def _clamp_dividers(self):
        """Enforce ordering, bounds and minimum spacing."""
        # Clamp to valid range
        for i in range(NUM_DIVIDERS):
            self.dividers[i] = max(MIN_DIVIDER_SPACING,
                                   min(self.source_width - MIN_DIVIDER_SPACING,
                                       self.dividers[i]))
        self.dividers.sort()

        # Enforce minimum spacing
        for i in range(1, NUM_DIVIDERS):
            if self.dividers[i] - self.dividers[i - 1] < MIN_DIVIDER_SPACING:
                self.dividers[i] = self.dividers[i - 1] + MIN_DIVIDER_SPACING
        # Final bounds check after spacing push
        for i in range(NUM_DIVIDERS):
            self.dividers[i] = max(MIN_DIVIDER_SPACING,
                                   min(self.source_width - MIN_DIVIDER_SPACING,
                                       self.dividers[i]))

    # ------------------------------------------------------------------ painting
    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        scale = self._display_scale()
        ox, oy = self._display_offset()

        # Draw scaled image
        dest_w = int(self.source_width * scale)
        dest_h = int(self.source_height * scale)
        scaled_pm = self._source_pixmap.scaled(dest_w, dest_h, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap(int(ox), int(oy), scaled_pm)

        # Draw divider lines
        pen = QPen(DIVIDER_COLOR, 2, Qt.PenStyle.SolidLine)
        painter.setPen(pen)

        top_y = int(oy)
        bot_y = int(oy + dest_h)

        for i, sx in enumerate(self.dividers):
            dx = int(self._source_x_to_display(sx))
            painter.drawLine(dx, top_y, dx, bot_y)

            # Draw a small handle circle at the midpoint
            mid_y = (top_y + bot_y) // 2
            painter.setBrush(DIVIDER_COLOR_ALPHA)
            painter.drawEllipse(QPoint(dx, mid_y), 6, 6)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.end()

    # ------------------------------------------------------------------ mouse interaction
    def _nearest_divider(self, display_x: float) -> Optional[int]:
        best_idx = None
        best_dist = float('inf')
        for i, sx in enumerate(self.dividers):
            dx = self._source_x_to_display(sx)
            dist = abs(display_x - dx)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_dist <= GRAB_RADIUS:
            return best_idx
        return None

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._nearest_divider(event.x())
            if idx is not None:
                self._dragging_index = idx
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._dragging_index is not None:
            new_sx = self._display_x_to_source(event.x())
            new_sx = int(round(new_sx))

            idx = self._dragging_index

            # Compute allowed bounds for this divider based on neighbours
            left_bound = MIN_DIVIDER_SPACING
            right_bound = self.source_width - MIN_DIVIDER_SPACING

            if idx > 0:
                left_bound = self.dividers[idx - 1] + MIN_DIVIDER_SPACING
            if idx < NUM_DIVIDERS - 1:
                right_bound = self.dividers[idx + 1] - MIN_DIVIDER_SPACING

            new_sx = max(left_bound, min(right_bound, new_sx))
            self.dividers[idx] = new_sx
            self.update()
            event.accept()
        else:
            idx = self._nearest_divider(event.x())
            if idx is not None:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_index is not None:
            self._dragging_index = None
            self._clamp_dividers()
            self.update()

            idx = self._nearest_divider(event.x())
            if idx is not None:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()

class DividerDialog(QDialog):
    """
    Dialog that presents a warped CLAHE ROI preview and lets the user
    position three vertical dividers to partition digits.
    """

    def __init__(self, bgr_image: np.ndarray,
                 default_dividers: Optional[List[int]] = None,
                 parent=None):
        """
        Args:
            bgr_image: Warped CLAHE image, shape (CNN_HEIGHT, CNN_WIDTH, 3), BGR.
            default_dividers: Previous divider positions for carryover.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Set Digit Dividers")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)

        # Default window size: 3× source
        default_w = CNN_WIDTH * DEFAULT_SCALE + 40  # padding
        default_h = CNN_HEIGHT * DEFAULT_SCALE + 120
        self.resize(default_w, default_h)
        self.setMinimumSize(CNN_WIDTH * 2 + 40, CNN_HEIGHT * 2 + 120)

        self._setup_ui(bgr_image, default_dividers)

    def _setup_ui(self, bgr_image, default_dividers):
        layout = QVBoxLayout()

        instructions = QLabel(
            "Drag the four green dividers to separate the four digit regions and the decimal point"
            "(ones | tenths | hundredths | thousandths)."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-size: 12px; padding: 4px; color: #ccc;")
        layout.addWidget(instructions)

        self.canvas = DividerCanvas(bgr_image, default_dividers)
        layout.addWidget(self.canvas, stretch=1)

        btn_layout = QHBoxLayout()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet("background-color: #7a4a4a; color: white; padding: 8px 16px;")
        self.cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_button)

        btn_layout.addStretch()

        self.reset_button = QPushButton("Reset to Even")
        self.reset_button.setStyleSheet("padding: 8px 16px;")
        self.reset_button.clicked.connect(self._reset_dividers)
        btn_layout.addWidget(self.reset_button)

        self.confirm_button = QPushButton("Confirm Dividers")
        self.confirm_button.setStyleSheet("background-color: #4a7a4a; color: white; padding: 8px 16px;")
        self.confirm_button.clicked.connect(self.accept)
        self.confirm_button.setDefault(True)
        btn_layout.addWidget(self.confirm_button)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _reset_dividers(self):
        step = self.canvas.source_width / (NUM_DIVIDERS + 1)
        self.canvas.dividers = [int(round(i * step)) for i in range(1, NUM_DIVIDERS + 1)]
        self.canvas._clamp_dividers()
        self.canvas.update()

    def get_dividers(self) -> List[int]:
        return self.canvas.get_dividers()
