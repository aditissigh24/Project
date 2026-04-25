"""
PDF rendering and text extraction utilities.
Uses PyMuPDF (fitz) as the primary extractor.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def extract_text_pymupdf(pdf_path: str, page_num: int) -> str:
    """
    Extract embedded text from a single PDF page (0-indexed internally, 1-indexed externally).
    Returns empty string if no selectable text is present.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]  # convert 1-indexed to 0-indexed
        return page.get_text("text")
    finally:
        doc.close()


def render_page_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> "bytes":
    """
    Render a single PDF page to a PNG image in memory (bytes).
    page_num is 1-indexed.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        zoom = dpi / 72.0  # 72 DPI is the default PDF resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        return pix.tobytes("png")
    finally:
        doc.close()


def get_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()
