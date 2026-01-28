from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QPoint, QEvent

class ZoomPanLabel(QLabel):
    """Label widget supporting zoom and pan for frame visibility."""
    
    MIN_ZOOM = 1.0
    MAX_ZOOM = 10.0
    ZOOM_IN_FACTOR = 1.15
    ZOOM_OUT_FACTOR = 0.85
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.original_pixmap = None
        self.original_w = 0
        self.original_h = 0
        
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        
        self.pan_dragging = False
        self.pan_drag_start = None
        self.pan_drag_start_offset = None
    
    def setFullPixmap(self, pixmap: QPixmap) -> None:
        """Use this instead of setPixmap() to enable zoom/pan functionality."""
        if pixmap is None:
            return
        
        new_w = pixmap.width()
        new_h = pixmap.height()
        
        dimensions_changed = (new_w != self.original_w or new_h != self.original_h)
        
        self.original_pixmap = pixmap
        self.original_w = new_w
        self.original_h = new_h
        
        if dimensions_changed:
            self.zoom_level = 1.0
            self.pan_offset_x = 0.0
            self.pan_offset_y = 0.0
        
        self._apply_zoom_pan()
    
    def _apply_zoom_pan(self) -> None:
        if self.original_pixmap is None:
            return
        
        if self.width() <= 0 or self.height() <= 0:
            return
        
        if self.zoom_level <= 1.0: # no zoom, just scale to fit
            self.pan_offset_x, self.pan_offset_y = 0.0, 0.0
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        
        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()
        
        visible_w = int(img_w / self.zoom_level)
        visible_h = int(img_h / self.zoom_level)

        visible_w = max(1, visible_w)
        visible_h = max(1, visible_h)
        
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

        if self.pan_dragging:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
    
    def set_zoom(self, zoom_level: float, pan_x: float | None = None, pan_y: float | None = None) -> None:
        if not self.original_w or not self.original_h:
            return
        self.zoom_level = max(1.0, min(zoom_level, 10.0))
        
        # no clamping here - _apply_zoom_pan handles it
        if pan_x is not None:
            self.pan_offset_x = pan_x
        if pan_y is not None:
            self.pan_offset_y = pan_y
        
        self._apply_zoom_pan()
    
    def _get_display_params(self) -> tuple[float, int, int, float, float] | None:
        """Calculate display scaling and offset parameters."""
        if not self.original_w or not self.original_h:
            return None
        
        label_w = self.width()
        label_h = self.height()
        
        visible_w = self.original_w / self.zoom_level
        visible_h = self.original_h / self.zoom_level

        if visible_w <= 0 or visible_h <=0:
            return None
        
        scale_w = label_w / visible_w
        scale_h = label_h / visible_h
        scale = min(scale_w, scale_h)
        
        display_w = int(visible_w * scale)
        display_h = int(visible_h * scale)
        
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2
        
        return scale, offset_x, offset_y, visible_w, visible_h
    
    def _label_to_image_pt(self, pos: QPoint) -> tuple[int, int] | tuple[None, None]:
        """Convert label position to image coordinates."""
        params = self._get_display_params()
        if not params:
            return None, None
        
        scale, off_x, off_y, _, _ = params
        
        # Convert screen -> crop-relative
        ix_rel = (pos.x() - off_x) / scale
        iy_rel = (pos.y() - off_y) / scale
        
        # Add pan offset
        ix = int(ix_rel + self.pan_offset_x)
        iy = int(iy_rel + self.pan_offset_y)
        
        return ix, iy
        
    def _get_scale_factor(self) -> float | None:
        if not self.original_w or not self.original_h:
            return None
        
        label_w = self.width()
        label_h = self.height()
        
        visible_w = self.original_w / self.zoom_level
        visible_h = self.original_h / self.zoom_level
        
        if visible_w <= 0 or visible_h <= 0:
            return None
        
        scale_w = label_w / visible_w
        scale_h = label_h / visible_h
        return min(scale_w, scale_h)

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        if not self.original_w or not self.original_h:
            event.ignore()
            return
        
        # determine zoom direction
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
            
        if delta > 0:
            new_zoom = self.zoom_level * self.ZOOM_IN_FACTOR
        else:
            new_zoom = self.zoom_level * self.ZOOM_OUT_FACTOR
        
        new_zoom = max(self.MIN_ZOOM, min(new_zoom, self.MAX_ZOOM))
        
        if abs(new_zoom - self.zoom_level) < 0.001:
            event.accept()
            return
        
        mouse_pos = event.pos()
        img_x, img_y = self._label_to_image_pt(mouse_pos)
        
        if img_x is not None and img_y is not None:
            # calculate new pan to keep mouse point stationary
            new_visible_w = self.original_w / new_zoom
            new_visible_h = self.original_h / new_zoom
            
            params = self._get_display_params()
            if params:
                scale, off_x, off_y, visible_w, visible_h = params
                rel_x = (mouse_pos.x() - off_x) / scale if scale > 0 else 0
                rel_y = (mouse_pos.y() - off_y) / scale if scale > 0 else 0
                
                self.pan_offset_x = img_x - (rel_x * new_visible_w / visible_w) if visible_w > 0 else 0
                self.pan_offset_y = img_y - (rel_y * new_visible_h / visible_h) if visible_h > 0 else 0
        
        self.zoom_level = new_zoom
        self._apply_zoom_pan()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.zoom_level > self.MIN_ZOOM:
            self.pan_dragging = True
            self.pan_drag_start = event.pos()
            self.pan_drag_start_offset = (self.pan_offset_x, self.pan_offset_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle panning drag."""
        if self.pan_dragging and self.pan_drag_start and self.pan_drag_start_offset:
            scale = self._get_scale_factor()
            if scale and scale > 0:
                dx_screen = event.pos().x() - self.pan_drag_start.x()
                dy_screen = event.pos().y() - self.pan_drag_start.y()
                
                # convert screen delta to image delta (inverted)
                dx_img = -dx_screen / scale
                dy_img = -dy_screen / scale
                
                self.pan_offset_x = self.pan_drag_start_offset[0] + dx_img
                self.pan_offset_y = self.pan_drag_start_offset[1] + dy_img
                
                self._apply_zoom_pan()
            event.accept()
        else:
            if self.zoom_level > self.MIN_ZOOM:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Stop panning."""
        if self.pan_dragging:
            self.pan_dragging = False
            self.pan_drag_start = None
            self.pan_drag_start_offset = None
            if self.zoom_level > self.MIN_ZOOM:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_zoom_pan()
    
