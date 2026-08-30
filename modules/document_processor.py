"""
Document Processor Module
--------------------------
Handles loading, resizing, deskewing, and enhancing uploaded ID card images
to prepare them for OCR extraction and face detection.
"""

import cv2
import numpy as np


def load_and_resize(img: np.ndarray, max_width: int = 1024) -> np.ndarray:
    """
    Resize the image proportionally so its width does not exceed max_width.
    This normalizes input sizes for consistent downstream processing.
    """
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def deskew(img_gray: np.ndarray) -> np.ndarray:
    """
    Detect the skew angle of the document using contour analysis
    and rotate it to straighten the text.

    Uses cv2.minAreaRect on the largest contour to determine the rotation angle.
    Only corrects angles within ±15° to avoid flipping already-aligned documents.
    """
    # Threshold to find edges
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img_gray

    # Use the largest contour to determine skew
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Normalize angle: minAreaRect returns angles in [-90, 0)
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct small skews (within ±15°)
    if abs(angle) > 15 or abs(angle) < 0.5:
        return img_gray

    # Rotate to deskew
    h, w = img_gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        img_gray, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


def enhance_for_ocr(img_gray: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding and bilateral filtering to improve
    text legibility for OCR.

    Returns a cleaned binary image optimized for text recognition.
    """
    # Bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(img_gray, 9, 75, 75)

    # Adaptive thresholding for robust binarization under uneven lighting
    enhanced = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    return enhanced


def process_document(img: np.ndarray) -> dict:
    """
    Full document preprocessing pipeline.

    Args:
        img: Raw BGR image (OpenCV format) of the ID card.

    Returns:
        Dictionary with:
            - 'color': Resized color image (for face detection)
            - 'gray': Grayscale version
            - 'deskewed': Deskewed grayscale image
            - 'enhanced': Enhanced binary image (for OCR)
    """
    # Step 1: Resize
    resized = load_and_resize(img)

    # Step 2: Grayscale
    gray = convert_to_grayscale(resized)

    # Step 3: Deskew
    deskewed = deskew(gray)

    # Step 4: Enhance for OCR
    enhanced = enhance_for_ocr(deskewed)

    return {
        "color": resized,
        "gray": gray,
        "deskewed": deskewed,
        "enhanced": enhanced,
    }
