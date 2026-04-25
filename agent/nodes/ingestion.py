"""
Ingestion node — PDF rendering + text extraction + OCR fallback.

Reads: pdf_path from state
Writes to Mongo: pages collection (raw_text, ocr_used, confidence per page)
Returns to state: page_count, pages_ocr_used
"""

from __future__ import annotations

import asyncio
import logging

from agent.state import IngestionState
from agent.tools.ocr_tools import page_needs_ocr, tesseract_ocr_page
from agent.tools.pdf_tools import extract_text_pymupdf, get_page_count, render_page_to_image
from config import settings
import db.repositories as repo

logger = logging.getLogger(__name__)


async def _process_page(
    pdf_path: str,
    page_num: int,
    doc_id: str,
    sem: asyncio.Semaphore,
) -> tuple[bool, float]:
    """
    Process one page: try PyMuPDF first, fall back to Tesseract if needed.
    Returns (ocr_used, confidence).
    """
    async with sem:
        loop = asyncio.get_event_loop()

        # Run blocking PDF extraction in thread pool
        raw_text = await loop.run_in_executor(None, extract_text_pymupdf, pdf_path, page_num)

        ocr_used = False
        confidence = 1.0

        if page_needs_ocr(raw_text):
            logger.debug("Page %d needs OCR", page_num)
            image_bytes = await loop.run_in_executor(
                None, render_page_to_image, pdf_path, page_num, settings.ocr_dpi
            )
            ocr_text, confidence = await loop.run_in_executor(
                None, tesseract_ocr_page, image_bytes
            )
            raw_text = ocr_text
            ocr_used = True

        # Persist page immediately — do not accumulate in state
        repo.upsert_page(doc_id, page_num, raw_text, ocr_used, confidence)
        return ocr_used, confidence


async def ingest_node(state: IngestionState) -> dict:
    """
    LangGraph node: iterate all pages, extract text (OCR if needed),
    write each page to Mongo, return only counts.
    """
    pdf_path = state["pdf_path"]
    doc_id = state["doc_id"]

    loop = asyncio.get_event_loop()
    total_pages = await loop.run_in_executor(None, get_page_count, pdf_path)

    # Update document record with page count
    repo.upsert_document(doc_id, pdf_path, page_count=total_pages, status="INGESTING")

    sem = asyncio.Semaphore(settings.llm_semaphore)
    tasks = [
        _process_page(pdf_path, page_num, doc_id, sem)
        for page_num in range(1, total_pages + 1)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    pages_ocr_used = 0
    errors: list[str] = []
    for i, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            msg = f"Page {i} failed: {result}"
            logger.error(msg)
            errors.append(msg)
        elif result[0]:  # ocr_used
            pages_ocr_used += 1

    logger.info("Ingested %d pages (%d via OCR)", total_pages, pages_ocr_used)

    update: dict = {
        "page_count": total_pages,
        "pages_ocr_used": pages_ocr_used,
    }
    if errors:
        update["errors"] = errors
    return update
