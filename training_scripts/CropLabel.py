"""
Shared CNNCropLabel widget for interactive ROI selection (Perspective Warp Edition).
Used by extract_frames.py and process_video.py.
"""
from typing import Optional, Tuple, List
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygon, QBrush
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

CNN_WIDTH = 256
CNN_HEIGHT = 64

class CNNCropLabel(QLabel):
    # Emits a list of 4 points: [(x,y), (x,y), (x,y), (x,y)]
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
        self.active_corner_index = None # 0=TL, 1=TR, 2=BR, 3=BL
        
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.pan_dragging = False
        self.pan_drag_start = None
        self.pan_drag_start_offset = None

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
        
        self.crop_points = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]
        self.crop_selected.emit(self.crop_points)
        self.update()

    def set_points(self, points: List[Tuple[int, int]]):
        """Load an existing quad from file"""
        if len(points) == 4:
            self.crop_points = points
            self.crop_selected.emit(self.crop_points)
            self.update()
    
    def set_zoom(self, zoom_level: float, pan_x: Optional[float] = None, pan_y: Optional[float] = None) -> None:
        if not self.original_w or not self.original_h: return
        self.zoom_level = max(1.0, min(zoom_level, 10.0))
        
        if pan_x is not None:
            max_pan_x = max(0, self.original_w - self.original_w / self.zoom_level)
            self.pan_offset_x = max(0, min(pan_x, max_pan_x))
        if pan_y is not None:
            max_pan_y = max(0, self.original_h - self.original_h / self.zoom_level)
            self.pan_offset_y = max(0, min(pan_y, max_pan_y))
        self.update()
    
    def _get_display_params(self):
        if not self.original_w or not self.original_h: return None
        label_w, label_h = self.width(), self.height()
        visible_w = self.original_w / self.zoom_level
        visible_h = self.original_h / self.zoom_level
        scale = min(label_w / visible_w, label_h / visible_h)
        display_w, display_h = int(visible_w * scale), int(visible_h * scale)
        offset_x, offset_y = (label_w - display_w) // 2, (label_h - display_h) // 2
        return scale, offset_x, offset_y

    def _img_pt_to_label(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        params = self._get_display_params()
        if not params: return (0, 0)
        scale, off_x, off_y = params
        x, y = pt
        lx = int((x - self.pan_offset_x) * scale) + off_x
        ly = int((y - self.pan_offset_y) * scale) + off_y
        return (lx, ly)

    def _label_pt_to_img(self, pos: QPoint) -> Tuple[int, int]:
        params = self._get_display_params()
        if not params: return (0, 0)
        scale, off_x, off_y = params
        ix = int((pos.x() - off_x) / scale + self.pan_offset_x)
        iy = int((pos.y() - off_y) / scale + self.pan_offset_y)
        return (ix, iy)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton: return
        if not self.crop_points: return

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
        if not self.crop_points: return
        
        if self.pan_dragging and self.pan_drag_start:
            dx_screen = event.pos().x() - self.pan_drag_start.x()
            dy_screen = event.pos().y() - self.pan_drag_start.y()
            params = self._get_display_params()
            if params:
                scale, _, _ = params
                self.pan_offset_x = max(0, min(self.pan_drag_start_offset[0] - dx_screen/scale, self.original_w - self.original_w/self.zoom_level))
                self.pan_offset_y = max(0, min(self.pan_drag_start_offset[1] - dy_screen/scale, self.original_h - self.original_h/self.zoom_level))
                parent = self.parent()
                if parent and hasattr(parent, 'on_pan_drag'): parent.on_pan_drag(self.pan_offset_x, self.pan_offset_y)
                self.update()
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

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.crop_points: return

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
        for i, point in enumerate(poly_points):
            painter.drawEllipse(point, r, r)
            # Optional: Label corners 0,1,2,3 for debugging
            # painter.drawText(point.x()+10, point.y(), str(i))