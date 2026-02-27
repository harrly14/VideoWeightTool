"""
Shared ROI utilities used by extraction, inference, and UI preview.
"""

import cv2
import numpy as np
from typing import Optional, List, Dict

from core.config import CNN_WIDTH, CNN_HEIGHT


def get_roi_for_frame(frame_num: int, roi_sections: List[Dict]) -> Optional[List[List[int]]]:
    """
    Find the correct ROI for a given frame number from a list of sections.

    Args:
        frame_num: The frame number to look up
        roi_sections: List of section dicts with 'quad', 'start_frame', 'end_frame'

    Returns:
        ROI quad coordinates list [[x,y],...], or None if frame not covered
    """
    if not roi_sections:
        return None

    for section in roi_sections:
        start = section.get('start_frame', 0)
        end = section.get('end_frame', float('inf'))
        if start <= frame_num <= end:
            return section.get('quad')

    return None


def warp_roi_to_canvas(frame, roi_coords, target_width=CNN_WIDTH, target_height=CNN_HEIGHT):
    """
    Perspective-warp a quad ROI onto a black canvas of the given size,
    preserving the original aspect ratio and centering horizontally.

    Args:
        frame: BGR image (numpy array)
        roi_coords: list of 4 [x, y] points (quad corners)
        target_width: canvas width  (default: CNN_WIDTH from config)
        target_height: canvas height (default: CNN_HEIGHT from config)

    Returns:
        BGR image of shape (target_height, target_width, 3)
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

        dst_pts = np.float32([
            [0, 0], [new_width - 1, 0],
            [new_width - 1, new_height - 1], [0, new_height - 1],
        ])
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

    Uses project-wide defaults from config when parameters are not supplied.
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
