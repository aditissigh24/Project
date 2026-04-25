"""
Finalize node — flip document status to READY, write run metrics summary.

Does NOT write any analysis artifacts — those are already in Mongo from their
respective producing nodes. This node is purely a bookkeeping step.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agent.state import IngestionState
import db.repositories as repo

logger = logging.getLogger(__name__)


async def finalize_node(state: IngestionState) -> dict:
    """
    LangGraph node: mark document as READY, persist run summary metrics.
    """
    doc_id = state["doc_id"]
    run_id = state.get("run_id", "unknown")

    run_data = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "page_count": state.get("page_count", 0),
        "pages_ocr_used": state.get("pages_ocr_used", 0),
        "segment_count": len(state.get("segment_ids", [])),
        "analyzed_count": len(state.get("analyzed_segment_ids", [])),
        "section_count": len(state.get("section_ids", [])),
        "chapter_count": len(state.get("chapter_ids", [])),
        "has_master_summary": state.get("has_master_summary", False),
        "contradiction_count": state.get("contradiction_count", 0),
        "errors": state.get("errors", []),
    }

    repo.upsert_run(doc_id, run_id, run_data)
    repo.set_document_status(doc_id, "READY")

    logger.info(
        "Document %s finalized: %d pages, %d segments, %d contradictions",
        doc_id,
        run_data["page_count"],
        run_data["segment_count"],
        run_data["contradiction_count"],
    )
    return {}
