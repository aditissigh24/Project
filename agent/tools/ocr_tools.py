"""
OCR utilities using Tesseract via pytesseract.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytesseract
from PIL import Image

from config import settings


def page_needs_ocr(text: str) -> bool:
    """
    Heuristic to decide if a page needs OCR.
    Triggers OCR if extracted text is too short or has very low alphabetic ratio.
    """
    stripped = text.strip()
    if len(stripped) < settings.ocr_min_char_count:
        return True
    alpha_chars = sum(c.isalpha() for c in stripped)
    ratio = alpha_chars / max(len(stripped), 1)
    return ratio < settings.ocr_min_alpha_ratio


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Apply OpenCV preprocessing to improve OCR quality:
    - Convert to grayscale
    - Denoise
    - Binarize (Otsu thresholding)
    """
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def tesseract_ocr_page(image_bytes: bytes, lang: str = "eng") -> tuple[str, float]:
    """
    Run Tesseract OCR on an image (bytes).
    Returns (text, confidence) where confidence is 0.0-1.0.
    """
    preprocessed = _preprocess_image(image_bytes)
    pil_img = Image.fromarray(preprocessed)

    # Get detailed output with confidence data
    data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)

    words = []
    confidences = []
    for i, word in enumerate(data["text"]):
        conf = data["conf"][i]
        if isinstance(conf, (int, float)) and conf >= 0 and word.strip():
            words.append(word)
            confidences.append(float(conf))

    text = " ".join(words)
    avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return text, avg_confidence
