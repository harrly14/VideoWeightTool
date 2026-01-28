from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal


CNN_WIDTH = 256
CNN_HEIGHT = 64

class RectCropLabel(QLabel): 
    crop_selected = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.crop_start = None
        self.crop_end = None
        self.cropping = None
        self.cropping_enabled = True
        self.original_w = None
        self.original_h = None

    def set_original_size(self, w, h):
        self.original_w = w
        self.original_h = h

    def mousePressEvent(self, event):
        if not self.cropping_enabled:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.cropping = True
            self.crop_start = event.pos()
            self.crop_end = event.pos()
            self.update()
    def mouseMoveEvent(self, event):
        if not self.cropping_enabled:
            return
        if self.cropping:
            self.crop_end = event.pos()
            self.update()
    def mouseReleaseEvent(self, event):
        if not self.cropping_enabled:
            QMessageBox.warning(self, 'Warning', "You must first clear the current crop in order to crop again")
            return
        if self.cropping:
            self.cropping = False
            if self.crop_start and self.crop_end:
                x1, y1 = self.crop_start.x(), self.crop_start.y()
                x2, y2 = self.crop_end.x(), self.crop_end.y()
                x, y , w, h = self.make_rectangle(x1, y1, x2, y2)
                crop_coords = self.convert_to_image_coords(x, y, w, h)

                if crop_coords:
                    self.crop_selected.emit(*crop_coords)
            self.update()
    def make_rectangle(self, x1, y1, x2, y2):
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return (x, y, w, h)
    
    def paintEvent(self, event):
        super().paintEvent(event)

        if self.crop_start and self.crop_end:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            x, y, w, h = self.make_rectangle(self.crop_start.x(), self.crop_start.y(), self.crop_end.x(), self.crop_end.y())
            painter.drawRect(x, y, w, h)

    def convert_to_image_coords(self, x, y, w, h):
        if not self.original_w or not self.original_h:
            return None
        
        label_w = self.width()
        label_h = self.height()
        
        display_ratio = min(label_w / self.original_w, label_h / self.original_h)
        display_w = int(self.original_w * display_ratio)
        display_h = int(self.original_h * display_ratio)
        
        # adjust for padding around the image in the label
        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2
        adj_x = x - offset_x
        adj_y = y - offset_y
        
        adj_x = max(0, min(adj_x, display_w))
        adj_y = max(0, min(adj_y, display_h))
        adj_w = max(1, min(w, display_w - adj_x))
        adj_h = max(1, min(h, display_h - adj_y))
        
        img_x = int(adj_x / display_ratio)
        img_y = int(adj_y / display_ratio)
        img_w = int(adj_w / display_ratio)
        img_h = int(adj_h / display_ratio)
        
        img_x = max(0, min(img_x, self.original_w - 1))
        img_y = max(0, min(img_y, self.original_h - 1))
        img_w = max(1, min(img_w, self.original_w - img_x))
        img_h = max(1, min(img_h, self.original_h - img_y))
        
        return (img_x, img_y, img_w, img_h)
