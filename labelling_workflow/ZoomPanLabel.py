from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

class ZoomPanLabel(QLabel):
    """Label widget supporting zoom and pan for frame visibility without crop functionality."""
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_w = 0
        self.original_h = 0
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        # Store the original full-resolution pixmap
        self.original_pixmap = None
        
        self.pan_dragging = False
        self.pan_drag_start = None
        self.pan_drag_start_offset = None
        
        self.setMouseTracking(True)
    
    def set_original_size(self, w: int, h: int) -> None:
        """Set original image dimensions."""
        self.original_w = w
        self.original_h = h
    
    def setFullPixmap(self, pixmap: QPixmap) -> None:
        """
        Set the full-resolution pixmap and apply zoom/pan.
        Use this instead of setPixmap() to enable zoom/pan functionality.
        """
        self.original_pixmap = pixmap
        self.original_w = pixmap.width()
        self.original_h = pixmap.height()
        self._apply_zoom_pan()
    
    def _apply_zoom_pan(self) -> None:
        """Apply current zoom and pan settings to display the pixmap."""
        if self.original_pixmap is None:
            return
        
        if self.zoom_level <= 1.0:
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)
            return
        
        img_w = self.original_pixmap.width()
        img_h = self.original_pixmap.height()
        
        visible_w = int(img_w / self.zoom_level)
        visible_h = int(img_h / self.zoom_level)
        
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
    
    def set_zoom(self, zoom_level: float, pan_x: float | None = None, pan_y: float | None = None) -> None:
        """Set zoom level and optional pan offset."""
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
    
    def _get_display_params(self) -> tuple[float, int, int, float, float] | None:
        """Calculate display scaling and offset parameters."""
        if not self.original_w or not self.original_h:
            return None
        
        label_w = self.width()
        label_h = self.height()
        
        visible_w = self.original_w / self.zoom_level
        visible_h = self.original_h / self.zoom_level
        
        scale = min(label_w / visible_w, label_h / visible_h)
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
        
        scale, off_x, off_y, visible_w, visible_h = params
        
        ix_rel = int((pos.x() - off_x) / scale)
        iy_rel = int((pos.y() - off_y) / scale)
        
        ix = ix_rel + int(self.pan_offset_x)
        iy = iy_rel + int(self.pan_offset_y)
        
        return ix, iy
    
    def mousePressEvent(self, event):
        """Handle mouse press for pan drag."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        if self.zoom_level > 1.0:
            self.pan_dragging = True
            self.pan_drag_start = event.pos()
            self.pan_drag_start_offset = (self.pan_offset_x, self.pan_offset_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for pan dragging."""
        if not self.original_w or not self.original_h:
            return
        
        if self.pan_dragging and self.pan_drag_start and self.pan_drag_start_offset:
            dx_screen = event.pos().x() - self.pan_drag_start.x()
            dy_screen = event.pos().y() - self.pan_drag_start.y()
            
            params = self._get_display_params()
            if params:
                scale, off_x, off_y, visible_w, visible_h = params
                dx_img = -dx_screen / scale
                dy_img = -dy_screen / scale
                
                new_pan_x = self.pan_drag_start_offset[0] + dx_img
                new_pan_y = self.pan_drag_start_offset[1] + dy_img
                
                max_pan_x = max(0, self.original_w - visible_w)
                max_pan_y = max(0, self.original_h - visible_h)
                
                self.pan_offset_x = max(0, min(new_pan_x, max_pan_x))
                self.pan_offset_y = max(0, min(new_pan_y, max_pan_y))
                
                self._apply_zoom_pan()
            return
        
        if self.zoom_level > 1.0:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to end pan drag."""
        if self.pan_dragging:
            self.pan_dragging = False
            self.pan_drag_start = None
            self.pan_drag_start_offset = None
            if self.zoom_level > 1.0:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        if not self.original_w or not self.original_h:
            return
        
        zoom_factor = 1.05 if event.angleDelta().y() > 0 else 0.95
        new_zoom = self.zoom_level * zoom_factor
        
        params = self._get_display_params()
        if params:
            scale, off_x, off_y, visible_w, visible_h = params
            mouse_img_x, mouse_img_y = self._label_to_image_pt(event.pos())
            
            if mouse_img_x is not None and mouse_img_y is not None:
                new_visible_w = self.original_w / new_zoom
                new_visible_h = self.original_h / new_zoom
                
                new_pan_x = mouse_img_x - (new_visible_w / 2)
                new_pan_y = mouse_img_y - (new_visible_h / 2)
                
                self.set_zoom(new_zoom, new_pan_x, new_pan_y)
            else:
                self.set_zoom(new_zoom)
        
        event.accept()
    
    def resizeEvent(self, event):
        """Handle resize to reapply zoom/pan."""
        super().resizeEvent(event)
        self._apply_zoom_pan()
    
    def reset_zoom(self):
        """Reset zoom to 1.0 and pan to origin."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self._apply_zoom_pan()
