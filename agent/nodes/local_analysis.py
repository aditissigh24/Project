"""
Local Analysis node — extract structured metadata from one segment.

Receives: {doc_id, segment_id} via Send (id-only payload)
Reads from Mongo: segments.text + segments.heading
Writes to Mongo: segment_analyses
Returns to state: analyzed_segment_ids=[segment_id]  (single-item list, merged by reducer)
"""

from __future__ import annotations

import logging

from agent.llm import get_llm, get_semaphore
from agent.prompts import LOCAL_ANALYSIS_PROMPT
from agent.schemas import SegmentAnalysis
from agent.tools.text_tools import count_tokens, enforce_token_budget
from config import settings
import db.repositories as repo

logger = logging.getLogger(__name__)


async def analyze_segment_node(state: dict) -> dict:
    """
    LangGraph node (invoked via Send): analyze one segment and write result to Mongo.

    `state` here is the Send payload: {doc_id, segment_id, run_id}
    plus any IngestionState fields LangGraph merges in.
    """
    doc_id = state["doc_id"]
    segment_id = state["segment_id"]

    # Skip if already analyzed (resume semantics)
    existing = repo.get_segment_analysis(doc_id, segment_id)
    if existing:
        logger.debug("Segment %s already analyzed, skipping", segment_id)
        return {"analyzed_segment_ids": [segment_id]}

    seg = repo.get_segment(doc_id, segment_id)
    if seg is None:
        logger.error("Segment %s not found in Mongo", segment_id)
        return {"analyzed_segment_ids": [], "errors": [f"Segment {segment_id} not found"]}

    heading = seg.get("heading", "Untitled")
    text = seg.get("text", "")

    # Enforce per-segment token budget
    text = enforce_token_budget(text, settings.max_segment_tokens)

    prompt = LOCAL_ANALYSIS_PROMPT.format(heading=heading, text=text)

    sem = get_semaphore()
    async with sem:
        llm = get_llm("heavy")
        structured_llm = llm.with_structured_output(SegmentAnalysis)
        try:
            analysis: SegmentAnalysis = await structured_llm.ainvoke(prompt)
            analysis_dict = analysis.model_dump()
        except Exception as e:
            logger.error("Analysis failed for segment %s: %s", segment_id, e)
            # Store a minimal placeholder so the pipeline continues
            analysis_dict = {
                "summary": f"[Analysis failed: {e}]",
                "key_entities": [],
                "key_claims": [],
                "decisions": [],
                "risks": [],
                "contradictions": [],
                "topics": [],
                "sentiment": "neutral",
            }

    repo.upsert_segment_analysis(doc_id, segment_id, analysis_dict)
    logger.debug("Analyzed segment %s", segment_id)

    return {"analyzed_segment_ids": [segment_id]}
