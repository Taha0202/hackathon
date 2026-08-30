"""
Face Verifier Module
---------------------
Detects and crops the face from an ID card image, then uses DeepFace
to compare it against an uploaded selfie. Returns match confidence.
"""

import cv2
import numpy as np
import tempfile
import os

# DeepFace is imported lazily to avoid slow startup
_deepface = None


def _get_deepface():
    """Lazily import DeepFace to avoid heavy startup cost on every module import."""
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


def get_cascade_path() -> str:
    """
    Get the path to the Haar cascade file.
    If cv2.data.haarcascades is not available, downloads it from OpenCV's official repo.
    """
    import urllib.request
    
    # Try using default cv2.data
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades') and cv2.data.haarcascades:
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(path):
                return path
    except Exception:
        pass

    # Fallback to downloading locally in the modules directory
    local_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(local_dir, "haarcascade_frontalface_default.xml")
    
    if not os.path.exists(local_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            # If download fails, try another mirror or raise error
            raise RuntimeError(f"Failed to download Haar cascade file: {str(e)}")
            
    return local_path


def detect_face_haar(img: np.ndarray) -> tuple:
    """
    Detect the largest face in an image using OpenCV's Haar Cascade.

    Args:
        img: BGR image (OpenCV format).

    Returns:
        Tuple (x, y, w, h) of the detected face bounding box,
        or None if no face is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade_path = get_cascade_path()
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return None

    # Return the largest face (highest area)
    largest = max(faces, key=lambda f: f[2] * f[3])
    return tuple(largest)


def crop_face(img: np.ndarray, bbox: tuple, padding: float = 0.2) -> np.ndarray:
    """
    Crop the face region from the image with optional padding.

    Args:
        img: BGR image.
        bbox: (x, y, w, h) bounding box.
        padding: Fraction of bbox dimensions to add as padding (default 20%).

    Returns:
        Cropped face image.
    """
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox

    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)

    return img[y1:y2, x1:x2]


def _save_temp_image(img: np.ndarray, prefix: str = "face") -> str:
    """Save an image to a temporary file and return the file path."""
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, f"{prefix}.jpg")
    cv2.imwrite(path, img)
    return path


def detect_face_deepface(img: np.ndarray) -> tuple:
    """
    Detect the largest face in an image using multiple DeepFace backends in sequence.
    Acts as a robust fallback for small, rotated, or low-contrast faces.
    """
    try:
        DeepFace = _get_deepface()
        temp_path = _save_temp_image(img, "detect_input")
        try:
            # Try SSD first (fastest), then MTCNN (robust for small faces), then MediaPipe
            for backend in ["ssd", "mtcnn", "mediapipe"]:
                try:
                    faces = DeepFace.extract_faces(
                        img_path=temp_path,
                        detector_backend=backend,
                        enforce_detection=True
                    )
                    if faces:
                        area = faces[0]["facial_area"]
                        return (area["x"], area["y"], area["w"], area["h"])
                except Exception:
                    continue
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception:
        pass
    return None


def verify_faces(
    id_card_img: np.ndarray,
    selfie_img: np.ndarray,
    model_name: str = "VGG-Face",
) -> dict:
    """
    Full face verification pipeline:
    1. Detect and crop face from the ID card.
    2. Compare it against the selfie using DeepFace.

    Args:
        id_card_img: BGR image of the ID card.
        selfie_img: BGR image of the selfie.
        model_name: DeepFace model to use (default: VGG-Face).

    Returns:
        Dictionary with:
            - 'verified': bool — whether faces match
            - 'confidence': float — match confidence percentage (0–100)
            - 'distance': float — raw distance metric
            - 'threshold': float — model's threshold for match
            - 'id_face_crop': ndarray — cropped face from ID card
            - 'id_face_bbox': tuple — (x,y,w,h) of detected face
            - 'error': str or None — error message if detection failed
    """
    DeepFace = _get_deepface()

    # Step 1: Detect face on the ID card
    bbox = None
    
    # Try Haar cascade first if CascadeClassifier exists
    if hasattr(cv2, 'CascadeClassifier'):
        try:
            bbox = detect_face_haar(id_card_img)
        except Exception:
            pass

    # Fall back to DeepFace detectors if Haar failed or is unavailable
    if bbox is None:
        try:
            bbox = detect_face_deepface(id_card_img)
        except Exception as e:
            return {
                "verified": False,
                "confidence": 0.0,
                "distance": None,
                "threshold": None,
                "id_face_crop": None,
                "id_face_bbox": None,
                "error": f"Face detection failed: {str(e)}",
            }

    if bbox is None:
        return {
            "verified": False,
            "confidence": 0.0,
            "distance": None,
            "threshold": None,
            "id_face_crop": None,
            "id_face_bbox": None,
            "error": "No face detected on the ID card. Please upload a clearer image.",
        }

    # Step 2: Crop the face
    id_face = crop_face(id_card_img, bbox)

    # Step 3: Save temps for DeepFace
    id_face_path = _save_temp_image(id_face, "id_face")
    selfie_path = _save_temp_image(selfie_img, "selfie")

    try:
        # Step 4: Run DeepFace verification
        # Use 'skip' backend since we already cropped the face region manually
        result = DeepFace.verify(
            img1_path=id_face_path,
            img2_path=selfie_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip",
        )

        # Step 5: Calculate confidence
        # DeepFace uses cosine distance for VGG-Face: lower = more similar
        distance = result.get("distance", 1.0)
        threshold = result.get("threshold", 0.4)
        verified = result.get("verified", False)

        # Convert distance to a confidence percentage
        # For cosine distance: confidence = (1 - distance) * 100, clamped to [0, 100]
        confidence = max(0.0, min(100.0, (1.0 - distance) * 100.0))

        return {
            "verified": verified,
            "confidence": round(confidence, 1),
            "distance": round(distance, 4),
            "threshold": threshold,
            "id_face_crop": id_face,
            "id_face_bbox": bbox,
            "error": None,
        }

    except Exception as e:
        return {
            "verified": False,
            "confidence": 0.0,
            "distance": None,
            "threshold": None,
            "id_face_crop": id_face,
            "id_face_bbox": bbox,
            "error": f"Face verification failed: {str(e)}",
        }

    finally:
        # Clean up temp files
        for path in [id_face_path, selfie_path]:
            try:
                os.remove(path)
            except OSError:
                pass
