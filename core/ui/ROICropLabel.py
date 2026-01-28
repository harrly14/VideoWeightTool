"""
ROI Selection Dialog for selecting the scale display region using a perspective quad.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QMessageBox, QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPolygon, QBrush
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QTimer


CNN_WIDTH = 256
CNN_HEIGHT = 64


class ROICropLabel(QLabel):
    """Label widget supporting interactive quad ROI selection"""

    crop_selected = pyqtSignal(list)
    HANDLE_RADIUS = 8
    
    def __init__(self) -> None:
        super().__init__()
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        self.crop_points: Optional[List[Tuple[int, int]]] = None 
        
        self.original_w: int = 0
        self.original_h: int = 0
        self.setMouseTracking(True)
        
        self.action = None
        self.drag_start_pos = None
        self.active_corner_index = None
        
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.pan_dragging = False
        self.pan_drag_start = None
        self.pan_drag_start_offset = None

        # Default standard ROI (will be adjusted based on video)
        self.std_x = 625
        self.std_y = 300
        self.std_w = 893
        self.std_h = 223
    
    def set_standard_roi(self, x: int, y: int, w: int, h: int) -> None:
        self.std_x, self.std_y, self.std_w, self.std_h = x, y, w, h
    
    def set_original_size(self, w: int, h: int) -> None:
        self.original_w = w
        self.original_h = h
        if self.crop_points is None:
            self.reset_crop_to_standard()
    
    def reset_crop_to_standard(self):
        if not self.original_w or not self.original_h:
            return
        x, y, w, h = self.std_x, self.std_y, self.std_w, self.std_h
        
        # Clamp to video dimensions
        x = min(x, self.original_w - w - 1)
        y = min(y, self.original_h - h - 1)
        
        self.crop_points = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]
        self.crop_selected.emit(self.crop_points)
        self.update()

    def set_points(self, points: List[Tuple[int, int]]):
        if points and len(points) == 4:
            self.crop_points = [(int(p[0]), int(p[1])) for p in points]
            self.crop_selected.emit(self.crop_points)
            self.update()
    
    def get_points(self) -> Optional[List[List[int]]]:
        """Get current ROI as list of [x,y] points"""
        if self.crop_points:
            return [[p[0], p[1]] for p in self.crop_points]
        return None
    
    def setFullPixmap(self, pixmap: QPixmap) -> None:
        """Set the full-resolution pixmap and apply zoom/pan."""
        if pixmap is None:
            return
        
        self.original_pixmap = pixmap
        new_w = pixmap.width()
        new_h = pixmap.height()
        
        if new_w != self.original_w or new_h != self.original_h:
            self.zoom_level = 1.0
            self.pan_offset_x = 0
            self.pan_offset_y = 0
        
        self.original_w = new_w
        self.original_h = new_h
        
        self._apply_zoom_pan()
    
    def _apply_zoom_pan(self) -> None:
        """Apply current zoom and pan settings to display the pixmap."""
        if not hasattr(self, 'original_pixmap') or self.original_pixmap is None:
            return
        
        if self.width() <= 0 or self.height() <= 0:
            return
        
        if self.zoom_level <= 1.0:
            # No zoom - scale to fit
            self.pan_offset_x, self.pan_offset_y = 0, 0
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)
            return
        
        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()
        
        visible_w = max(1, int(img_w / self.zoom_level))
        visible_h = max(1, int(img_h / self.zoom_level))
        
        max_pan_x = max(0, img_w - visible_w)
        max_pan_y = max(0, img_h - visible_h)
        self.pan_offset_x = max(0, min(self.pan_offset_x, max_pan_x))
        self.pan_offset_y = max(0, min(self.pan_offset_y, max_pan_y))
        
        x = int(self.pan_offset_x)
        y = int(self.pan_offset_y)
        
        cropped = self.original_pixmap.copy(x, y, visible_w, visible_h)
        
        scaled = cropped.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled)
    
    def set_zoom(self, zoom_level: float, pan_x: Optional[float] = None, pan_y: Optional[float] = None) -> None:
        if not self.original_w or not self.original_h:
            return
        self.zoom_level = max(1.0, min(zoom_level, 10.0))
        
        if pan_x is not None:
            max_pan_x = max(0, self.original_w - self.original_w / self.zoom_level)
            self.pan_offset_x = max(0, min(pan_x, max_pan_x))
        if pan_y is not None:
            max_pan_y = max(0, self.original_h - self.original_h / self.zoom_level)
            self.pan_offset_y = max(0, min(pan_y, max_pan_y))
        self._apply_zoom_pan()
    
    def _get_display_params(self):
        if not self.original_w or not self.original_h:
            return None
        label_w, label_h = self.width(), self.height()
        visible_w = self.original_w / self.zoom_level
        visible_h = self.original_h / self.zoom_level
        scale = min(label_w / visible_w, label_h / visible_h)
        display_w, display_h = int(visible_w * scale), int(visible_h * scale)
        offset_x, offset_y = (label_w - display_w) // 2, (label_h - display_h) // 2
        return scale, offset_x, offset_y

    def _img_pt_to_label(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        params = self._get_display_params()
        if not params:
            return (0, 0)
        scale, off_x, off_y = params
        x, y = pt
        lx = int((x - self.pan_offset_x) * scale) + off_x
        ly = int((y - self.pan_offset_y) * scale) + off_y
        return (lx, ly)

    def _label_pt_to_img(self, pos: QPoint) -> Tuple[int, int]:
        params = self._get_display_params()
        if not params:
            return (0, 0)
        scale, off_x, off_y = params
        ix = int((pos.x() - off_x) / scale + self.pan_offset_x)
        iy = int((pos.y() - off_y) / scale + self.pan_offset_y)
        return (ix, iy)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.crop_points:
            return

        mouse_pos = event.pos()
        clicked_corner_idx = -1
        
        for i, pt in enumerate(self.crop_points):
            lx, ly = self._img_pt_to_label(pt)
            if abs(mouse_pos.x() - lx) < self.HANDLE_RADIUS and abs(mouse_pos.y() - ly) < self.HANDLE_RADIUS:
                clicked_corner_idx = i
                break
        
        if clicked_corner_idx != -1:
            self.action = 'resize'
            self.active_corner_index = clicked_corner_idx
            self.drag_start_pos = mouse_pos
            return

        poly = QPolygon()
        for pt in self.crop_points:
            poly.append(QPoint(*self._img_pt_to_label(pt)))
            
        if poly.containsPoint(mouse_pos, Qt.FillRule.OddEvenFill):
            self.action = 'move'
            self.drag_start_pos = mouse_pos
            self.initial_points = list(self.crop_points)
        else:
            if self.zoom_level > 1.0:
                self.pan_dragging = True
                self.pan_drag_start = event.pos()
                self.pan_drag_start_offset = (self.pan_offset_x, self.pan_offset_y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if not self.crop_points:
            return
        
        if self.pan_dragging and self.pan_drag_start:
            dx_screen = event.pos().x() - self.pan_drag_start.x()
            dy_screen = event.pos().y() - self.pan_drag_start.y()
            params = self._get_display_params()
            if params:
                scale, _, _ = params
                self.pan_offset_x = max(0, min(self.pan_drag_start_offset[0] - dx_screen/scale, self.original_w - self.original_w/self.zoom_level))
                self.pan_offset_y = max(0, min(self.pan_drag_start_offset[1] - dy_screen/scale, self.original_h - self.original_h/self.zoom_level))
                self._apply_zoom_pan()
            return

        if not self.action:
            hover_handle = False
            for pt in self.crop_points:
                lx, ly = self._img_pt_to_label(pt)
                if abs(event.pos().x() - lx) < self.HANDLE_RADIUS and abs(event.pos().y() - ly) < self.HANDLE_RADIUS:
                    hover_handle = True
            
            if hover_handle:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                poly = QPolygon([QPoint(*self._img_pt_to_label(p)) for p in self.crop_points])
                if poly.containsPoint(event.pos(), Qt.FillRule.OddEvenFill):
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        curr_ix, curr_iy = self._label_pt_to_img(event.pos())
        curr_ix = max(0, min(curr_ix, self.original_w - 1))
        curr_iy = max(0, min(curr_iy, self.original_h - 1))

        if self.action == 'resize' and self.active_corner_index is not None:
            pts = list(self.crop_points)
            pts[self.active_corner_index] = (curr_ix, curr_iy)
            self.crop_points = pts
            
        elif self.action == 'move':
            start_ix, start_iy = self._label_pt_to_img(self.drag_start_pos)
            dx = curr_ix - start_ix
            dy = curr_iy - start_iy
            
            new_pts = []
            for (ix, iy) in self.initial_points:
                new_pts.append((
                    max(0, min(ix + dx, self.original_w - 1)),
                    max(0, min(iy + dy, self.original_h - 1))
                ))
            self.crop_points = new_pts

        self.crop_selected.emit(self.crop_points)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self.action = None
        self.active_corner_index = None
        self.pan_dragging = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom."""
        if not self.original_w or not self.original_h:
            event.ignore()
            return
        
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        
        zoom_factor = 1.15 if delta > 0 else 0.85
        new_zoom = self.zoom_level * zoom_factor
        new_zoom = max(1.0, min(new_zoom, 10.0))
        
        if abs(new_zoom - self.zoom_level) < 0.001:
            event.accept()
            return
        
        # Get mouse position in image coordinates before zoom
        mouse_pos = event.pos()
        img_x, img_y = self._label_pt_to_img(mouse_pos)
        
        # Calculate new pan to keep mouse point stationary
        params = self._get_display_params()
        if params:
            scale, off_x, off_y = params
            if scale > 0:
                rel_x = (mouse_pos.x() - off_x) / scale
                rel_y = (mouse_pos.y() - off_y) / scale
                
                new_visible_w = self.original_w / new_zoom
                new_visible_h = self.original_h / new_zoom
                
                visible_w = self.original_w / self.zoom_level
                visible_h = self.original_h / self.zoom_level
                
                if visible_w > 0 and visible_h > 0:
                    new_pan_x = img_x - (rel_x * new_visible_w / visible_w)
                    new_pan_y = img_y - (rel_y * new_visible_h / visible_h)
                    
                    max_pan_x = max(0, self.original_w - new_visible_w)
                    max_pan_y = max(0, self.original_h - new_visible_h)
                    self.pan_offset_x = max(0, min(new_pan_x, max_pan_x))
                    self.pan_offset_y = max(0, min(new_pan_y, max_pan_y))
        
        self.zoom_level = new_zoom
        self._apply_zoom_pan()
        event.accept()
    
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_zoom_pan()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.crop_points:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        poly_points = [QPoint(*self._img_pt_to_label(pt)) for pt in self.crop_points]
        
        painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.PenStyle.SolidLine))
        painter.drawPolygon(QPolygon(poly_points))
        
        fill_color = QColor(0, 255, 0, 30)
        path = QPolygon(poly_points)
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(path)
        
        painter.setBrush(QColor(255, 255, 0))
        r = self.HANDLE_RADIUS
        for point in poly_points:
            painter.drawEllipse(point, r, r)