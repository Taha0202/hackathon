"""
Shared utility functions for image format conversions and display helpers.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def bytes_to_cv2(uploaded_file) -> np.ndarray:
    """
    Convert a Streamlit UploadedFile (or any file-like object with .read())
    to an OpenCV BGR numpy array.
    """
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    uploaded_file.seek(0)  # Reset stream position for potential re-reads
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image. Ensure it is a valid JPG/PNG file.")
    return img


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV BGR numpy array to a PIL RGB Image
    suitable for display in Streamlit.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def draw_face_bbox(img: np.ndarray, bbox: tuple, color=(0, 255, 0), thickness=3) -> np.ndarray:
    """
    Draw a rectangle around the detected face region on a copy of the image.

    Args:
        img: OpenCV BGR image.
        bbox: Tuple of (x, y, w, h) for the face bounding box.
        color: BGR color for the rectangle (default: green).
        thickness: Line thickness in pixels.

    Returns:
        A copy of the image with the bounding box drawn.
    """
    annotated = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    return annotated


def resize_for_display(img: np.ndarray, max_width: int = 400) -> np.ndarray:
    """
    Resize an image proportionally so its width does not exceed max_width.
    Useful for consistent display in the Streamlit UI.
    """
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
