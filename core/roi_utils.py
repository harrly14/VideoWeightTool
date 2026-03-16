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


def apply_clahe(image, clip_limit=None, grid_size=None, color_order="bgr", **kwargs):
    """
    Apply CLAHE to a grayscale projection of an image.

    Args:
        image: Input image in grayscale (H, W) or color (H, W, 3).
        clip_limit: CLAHE clip limit override.
        grid_size: CLAHE tile grid override.
        color_order: Channel order for 3-channel inputs. One of:
            - "bgr" for OpenCV frames (default)
            - "rgb" for Albumentations/dataset path

    Returns:
        np.ndarray: (H, W, 1) grayscale image for single-channel model input.
    """
    from core.config import CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE

    if clip_limit is None:
        clip_limit = CLAHE_CLIP_LIMIT
    if grid_size is None:
        grid_size = CLAHE_GRID_SIZE

    if len(image.shape) == 3 and image.shape[2] == 3:
        if color_order == "rgb":
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif color_order == "bgr":
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported color_order '{color_order}'. Use 'rgb' or 'bgr'.")
    else:
        grayscale_image = image

    # OpenCV CLAHE requires 8-bit single-channel input.
    if grayscale_image.dtype != np.uint8:
        if np.issubdtype(grayscale_image.dtype, np.floating):
            max_val = float(np.nanmax(grayscale_image)) if grayscale_image.size else 0.0
            min_val = float(np.nanmin(grayscale_image)) if grayscale_image.size else 0.0
            # Support common float encodings: [0,1] and [0,255].
            if 0.0 <= min_val and max_val <= 1.0:
                grayscale_image = (grayscale_image * 255.0).round()
            grayscale_image = np.clip(grayscale_image, 0.0, 255.0).astype(np.uint8)
        else:
            grayscale_image = np.clip(grayscale_image, 0, 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(grayscale_image)
    
    return enhanced[:, :, np.newaxis]
