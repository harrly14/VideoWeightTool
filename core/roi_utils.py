"""
Shared ROI utilities used by extraction, inference, and UI preview.
"""

import cv2
import numpy as np
from typing import List, Dict


from core.config import CNN_WIDTH, CNN_HEIGHT


def slice_roi_into_digits(canvas, dividers):
    assert len(dividers) == 3
    dividers = sorted(dividers) #just in case the user dragged the dividers out of order
    height = canvas.shape[0]
    width = canvas.shape[1]
    boundaries = [0] + dividers + [width]

    digits = []
    for i in range(len(boundaries) - 1):
        x_start = max(0,boundaries[i])
        x_end = min(width, boundaries[i+1])
        digit_crop = canvas[:, x_start:x_end]
        if digit_crop.size == 0:
            print(f"ERROR: Digit {i} has a width of 0.")
            return None
        digits.append(digit_crop)

    return digits

def get_roi_for_frame(frame_num: int, roi_sections: List[Dict]):
    if not roi_sections:
        return None

    for section in roi_sections:
        start = section.get('start_frame', 0)
        end = section.get('end_frame', float('inf'))
        if start <= frame_num <= end:
            return (section.get('quad'), section.get('dividers'))

    return None


def warp_roi_to_canvas(frame, roi_coords, target_width=CNN_WIDTH, target_height=CNN_HEIGHT):
    """
    Perspective-warp a quad ROI onto a black canvas of the given size,
    preserving the original aspect ratio and centering horizontally.
    Returns a BGR image of shape (target_height, target_width, 3)
    """
    if not roi_coords:
        return cv2.resize(frame, (target_width, target_height))

    try:
        pts = np.array(roi_coords, dtype="float32")

        width = max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3]))
        height = max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1]))
        aspect_ratio = width / height if height > 0 else 1.0

        new_height = target_height
        new_width = max(1, int(aspect_ratio * new_height))

        dst_pts = np.array([
            [0, 0], [new_width - 1, 0],
            [new_width - 1, new_height - 1], [0, new_height - 1],
            ], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (new_width, new_height), flags=cv2.INTER_LINEAR)

        # Center on black canvas
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        if new_width > target_width:
            warped = cv2.resize(warped, (target_width, target_height))
            new_width = target_width
        x_offset = (target_width - new_width) // 2
        canvas[:, x_offset:x_offset + new_width] = warped

        return canvas
    except Exception as e:
        print(f"ROI warp error: {e}")
        return cv2.resize(frame, (target_width, target_height))


def apply_clahe(frame, clip_limit=None, grid_size=None):
    """
    Apply CLAHE to a BGR frame, returning a BGR result.
    """
    from core.config import CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE

    if clip_limit is None:
        clip_limit = CLAHE_CLIP_LIMIT
    if grid_size is None:
        grid_size = CLAHE_GRID_SIZE

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
