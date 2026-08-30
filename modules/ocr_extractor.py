"""
OCR Extractor Module
---------------------
Uses EasyOCR to extract text from preprocessed document images
and parses structured fields (Name, Date of Birth, ID Number) via regex.
"""

import re
import easyocr
import numpy as np

# Module-level reader cache — initialized once and reused across calls
_reader = None


def _get_reader():
    """
    Lazily initialize and cache the EasyOCR Reader instance.
    This avoids reloading the model on every call (important for Streamlit reruns).
    """
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_raw_text(img: np.ndarray) -> list:
    """
    Run EasyOCR on the image and return raw results.

    Args:
        img: Preprocessed grayscale or enhanced image (numpy array).

    Returns:
        List of tuples: (bounding_box, text, confidence)
    """
    reader = _get_reader()
    results = reader.readtext(img, detail=1, paragraph=False)
    return results


def parse_name(raw_results: list) -> tuple:
    """
    Attempt to extract the person's name from OCR results.

    Looks for labels like 'Name', 'Surname', 'Given Name', 'Full Name'
    and returns the adjacent text value.

    Returns:
        (name_string, confidence) or (None, 0.0)
    """
    name_labels = re.compile(
        r"\b(name|surname|given\s*name|full\s*name|first\s*name|last\s*name|nom|prenom)\b",
        re.IGNORECASE,
    )

    texts = [(entry[1].strip(), entry[2]) for entry in raw_results]

    for i, (text, conf) in enumerate(texts):
        if name_labels.search(text):
            # The name value might be in the same field (after a colon/separator)
            # or in the next field
            parts = re.split(r"[:\-–]", text, maxsplit=1)
            if len(parts) > 1 and len(parts[1].strip()) > 1:
                return parts[1].strip(), conf

            # Check the next entry
            if i + 1 < len(texts):
                next_text, next_conf = texts[i + 1]
                if len(next_text) > 1 and not name_labels.search(next_text):
                    return next_text, next_conf

    # Fallback: look for a text entry that looks like a name (alphabetic, 2+ words)
    for text, conf in texts:
        cleaned = text.strip()
        if (
            re.match(r"^[A-Za-z\s\.\-]{4,40}$", cleaned)
            and len(cleaned.split()) >= 2
            and conf > 0.4
            and not re.search(r"\d", cleaned)
            and not name_labels.search(cleaned)
        ):
            return cleaned, conf

    return None, 0.0


def parse_date_of_birth(raw_results: list) -> tuple:
    """
    Extract a date of birth by matching common date patterns.

    Supports formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY

    Returns:
        (date_string, confidence) or (None, 0.0)
    """
    date_pattern = re.compile(
        r"\b(\d{1,2}[\\/\-\.]\d{1,2}[\\/\-\.]\d{2,4})\b"
        r"|"
        r"\b(\d{4}[\\/\-\.]\d{1,2}[\\/\-\.]\d{1,2})\b"
    )

    dob_label = re.compile(
        r"\b(date\s*of\s*birth|dob|birth\s*date|d\.o\.b|born|naissance)\b",
        re.IGNORECASE,
    )

    texts = [(entry[1].strip(), entry[2]) for entry in raw_results]

    # First pass: look for dates near a DOB label
    for i, (text, conf) in enumerate(texts):
        if dob_label.search(text):
            # Check in the same field
            match = date_pattern.search(text)
            if match:
                date_str = match.group(1) or match.group(2)
                return date_str, conf

            # Check adjacent fields
            for j in range(i + 1, min(i + 3, len(texts))):
                match = date_pattern.search(texts[j][0])
                if match:
                    date_str = match.group(1) or match.group(2)
                    return date_str, texts[j][1]

    # Second pass: return any date found
    for text, conf in texts:
        match = date_pattern.search(text)
        if match:
            date_str = match.group(1) or match.group(2)
            return date_str, conf

    return None, 0.0


def parse_id_number(raw_results: list) -> tuple:
    """
    Extract an ID/document number — typically an alphanumeric sequence
    of 6–12 characters.

    Returns:
        (id_string, confidence) or (None, 0.0)
    """
    id_label = re.compile(
        r"\b(id\s*no|id\s*number|document\s*no|passport\s*no|card\s*no|numero|number)\b",
        re.IGNORECASE,
    )

    id_pattern = re.compile(r"\b([A-Z0-9]{6,15})\b")

    texts = [(entry[1].strip(), entry[2]) for entry in raw_results]

    # First pass: look for ID numbers near a label
    for i, (text, conf) in enumerate(texts):
        if id_label.search(text):
            # Same field
            match = id_pattern.search(text)
            if match:
                return match.group(1), conf

            # Adjacent fields
            for j in range(i + 1, min(i + 3, len(texts))):
                match = id_pattern.search(texts[j][0])
                if match:
                    return match.group(1), texts[j][1]

    # Second pass: find any alphanumeric string that looks like an ID
    for text, conf in texts:
        match = id_pattern.search(text)
        if match:
            candidate = match.group(1)
            # Filter out common false positives (short words, common labels)
            if (
                len(candidate) >= 6
                and any(c.isdigit() for c in candidate)
                and conf > 0.3
            ):
                return candidate, conf

    return None, 0.0


def extract_fields(img: np.ndarray) -> dict:
    """
    Full OCR extraction pipeline: run OCR, then parse structured fields.

    Args:
        img: Preprocessed image (grayscale or enhanced).

    Returns:
        Dictionary with:
            - 'name': (value, confidence)
            - 'date_of_birth': (value, confidence)
            - 'id_number': (value, confidence)
            - 'raw_text': Full concatenated OCR output
            - 'raw_results': Raw EasyOCR results list
            - 'avg_confidence': Average confidence across all OCR detections
    """
    raw_results = extract_raw_text(img)

    if not raw_results:
        return {
            "name": (None, 0.0),
            "date_of_birth": (None, 0.0),
            "id_number": (None, 0.0),
            "raw_text": "",
            "raw_results": [],
            "avg_confidence": 0.0,
        }

    # Parse individual fields
    name = parse_name(raw_results)
    dob = parse_date_of_birth(raw_results)
    id_number = parse_id_number(raw_results)

    # Compute overall stats
    all_texts = [entry[1] for entry in raw_results]
    all_confs = [entry[2] for entry in raw_results]
    raw_text = " | ".join(all_texts)
    avg_confidence = sum(all_confs) / len(all_confs) if all_confs else 0.0

    return {
        "name": name,
        "date_of_birth": dob,
        "id_number": id_number,
        "raw_text": raw_text,
        "raw_results": raw_results,
        "avg_confidence": avg_confidence,
    }
