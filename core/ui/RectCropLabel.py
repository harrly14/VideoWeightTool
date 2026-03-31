from PyQt5.QtGui import QPainter, QPen, QColor, QPolygon
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

class RectCropLabel(QLabel): 
    warp_changed = pyqtSignal(list)
    HANDLE_RADIUS = 8

    def __init__(self):
        super().__init__()
        self.original_w = None
        self.original_h = None
        self.warp_mode = False
        self.warp_quad = None
        self.active_warp_handle = None

    def set_warp_mode(self, enabled):
        self.warp_mode = bool(enabled)
        self.active_warp_handle = None
        self.update()

    def ensure_warp_quad(self):
        if self.warp_quad is not None:
            return
        if not self.original_w or not self.original_h:
            return
        self.warp_quad = [
            (0, 0),
            (self.original_w - 1, 0),
            (self.original_w - 1, self.original_h - 1),
            (0, self.original_h - 1),
        ]
        self.warp_changed.emit(list(self.warp_quad))
        self.update()

    def set_warp_quad(self, quad):
        if quad is None:
            self.warp_quad = None
            self.warp_changed.emit([])
            self.update()
            return
        if len(quad) != 4:
            return
        normalized = []
        for point in quad:
            try:
                x = int(point[0])
                y = int(point[1])
            except (TypeError, ValueError, IndexError):
                return
            normalized.append(self._clamp_image_point(x, y))
        self.warp_quad = normalized
        self.warp_changed.emit(list(self.warp_quad))
        self.update()

    def clear_warp_quad(self):
        self.warp_quad = None
        self.active_warp_handle = None
        self.warp_changed.emit([])
        self.update()

    def get_warp_quad(self):
        if self.warp_quad is None:
            return None
        return list(self.warp_quad)

    def set_original_size(self, w, h):
        self.original_w = w
        self.original_h = h

    def mousePressEvent(self, event):
        if not self.warp_mode:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.warp_quad is None:
            self.ensure_warp_quad()
            return
        self.active_warp_handle = self._find_warp_handle(event.pos())

    def mouseMoveEvent(self, event):
        if not self.warp_mode:
            return
        if self.active_warp_handle is None or self.warp_quad is None:
            return
        img_x, img_y = self._label_point_to_image(event.pos())
        quad = list(self.warp_quad)
        quad[self.active_warp_handle] = self._clamp_image_point(img_x, img_y)
        self.warp_quad = quad
        self.warp_changed.emit(list(self.warp_quad))
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.warp_mode:
            return
        self.active_warp_handle = None
    
    def paintEvent(self, event):
        super().paintEvent(event)

        if self.warp_mode and self.warp_quad is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            label_quad = [self._image_point_to_label(x, y) for x, y in self.warp_quad]
            polygon = QPolygon(label_quad)

            painter.setPen(QPen(QColor(80, 220, 80), 2, Qt.PenStyle.SolidLine))
            painter.drawPolygon(polygon)

            # Draw a simple warp grid to show perspective mapping.
            painter.setPen(QPen(QColor(80, 220, 80, 140), 1, Qt.PenStyle.DashLine))
            for t in (0.2, 0.4, 0.6, 0.8):
                left_pt = self._interp_point(label_quad[0], label_quad[3], t)
                right_pt = self._interp_point(label_quad[1], label_quad[2], t)
                top_pt = self._interp_point(label_quad[0], label_quad[1], t)
                bottom_pt = self._interp_point(label_quad[3], label_quad[2], t)
                painter.drawLine(left_pt, right_pt)
                painter.drawLine(top_pt, bottom_pt)

            painter.setPen(QPen(QColor(240, 200, 40), 2, Qt.PenStyle.SolidLine))
            painter.setBrush(QColor(240, 200, 40, 180))
            for point in label_quad:
                painter.drawEllipse(point, self.HANDLE_RADIUS, self.HANDLE_RADIUS)

    def _get_display_params(self):
        if not self.original_w or not self.original_h:
            return None

        label_w = self.width()
        label_h = self.height()
        display_ratio = min(label_w / self.original_w, label_h / self.original_h)
        display_w = int(self.original_w * display_ratio)
        display_h = int(self.original_h * display_ratio)
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2
        return (display_ratio, display_w, display_h, offset_x, offset_y)

    def _label_point_to_image(self, point):
        params = self._get_display_params()
        if params is None:
            return (0, 0)
        display_ratio, display_w, display_h, offset_x, offset_y = params
        adj_x = point.x() - offset_x
        adj_y = point.y() - offset_y
        adj_x = max(0, min(adj_x, display_w))
        adj_y = max(0, min(adj_y, display_h))

        img_x = int(adj_x / display_ratio)
        img_y = int(adj_y / display_ratio)
        return self._clamp_image_point(img_x, img_y)

    def _image_point_to_label(self, x, y):
        params = self._get_display_params()
        if params is None:
            return QPoint(0, 0)
        display_ratio, _, _, offset_x, offset_y = params
        label_x = int(x * display_ratio) + offset_x
        label_y = int(y * display_ratio) + offset_y
        return QPoint(label_x, label_y)

    def _clamp_image_point(self, x, y):
        if not self.original_w or not self.original_h:
            return (int(x), int(y))
        clamped_x = max(0, min(int(x), self.original_w - 1))
        clamped_y = max(0, min(int(y), self.original_h - 1))
        return (clamped_x, clamped_y)

    def _find_warp_handle(self, mouse_point):
        if self.warp_quad is None:
            return None
        for idx, (x, y) in enumerate(self.warp_quad):
            handle_pt = self._image_point_to_label(x, y)
            dx = mouse_point.x() - handle_pt.x()
            dy = mouse_point.y() - handle_pt.y()
            if (dx * dx + dy * dy) <= (self.HANDLE_RADIUS * self.HANDLE_RADIUS * 3):
                return idx
        return None

    def _interp_point(self, p0, p1, t):
        x = int(p0.x() + (p1.x() - p0.x()) * t)
        y = int(p0.y() + (p1.y() - p0.y()) * t)
        return QPoint(x, y)
