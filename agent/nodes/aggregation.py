"""
Aggregation nodes — hierarchical map-reduce: segments → sections → chapters → document.

Each node reads from Mongo, writes aggregated summaries to Mongo, returns ids to state.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict

from agent.llm import get_llm
from agent.prompts import (
    CHAPTER_AGGREGATE_PROMPT,
    DOCUMENT_AGGREGATE_PROMPT,
    SECTION_AGGREGATE_PROMPT,
)
from agent.schemas import (
    ChapterSummaryResponse,
    DocumentSummaryResponse,
    SectionSummaryResponse,
)
from agent.state import IngestionState
from agent.tools.text_tools import count_tokens, enforce_token_budget
from config import settings
import db.repositories as repo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass 1: Sections
# ---------------------------------------------------------------------------

async def aggregate_sections_node(state: IngestionState) -> dict:
    """
    Group segments by parent_section (or parent_chapter as fallback),
    aggregate their analysis summaries into SectionSummary objects.
    Writes to section_summaries collection.
    """
    doc_id = state["doc_id"]

    # Fetch all segments with hierarchy metadata
    segments = repo.get_segments(doc_id, fields=["segment_id", "heading", "parent_section", "parent_chapter", "type"])

    # Group segments: use (parent_section or heading) as section key
    groups: dict[str, list[str]] = defaultdict(list)
    heading_map: dict[str, str] = {}

    for seg in segments:
        seg_id = seg["segment_id"]
        seg_type = seg.get("type", "section")

        if seg_type in ("chapter", "section"):
            # This segment IS a section — group it alone
            key = seg["heading"]
        elif seg.get("parent_section"):
            key = seg["parent_section"]
        elif seg.get("parent_chapter"):
            key = seg["parent_chapter"]
        else:
            key = seg.get("heading", "Uncategorized")

        groups[key].append(seg_id)
        heading_map[key] = key  # heading == key for now

    section_ids = []
    llm = get_llm("heavy")
    structured_llm = llm.with_structured_output(SectionSummaryResponse)

    for heading, seg_ids in groups.items():
        section_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{heading}"))

        # Skip if already exists
        existing = repo.get_section_summary(doc_id, section_id)
        if existing:
            section_ids.append(section_id)
            continue

        # Fetch analysis summaries (only the summary field to stay within budget)
        analyses = [repo.get_segment_analysis(doc_id, sid) for sid in seg_ids]
        analyses = [a for a in analyses if a]

        if not analyses:
            logger.warning("No analyses found for section '%s'", heading)
            continue

        # Build aggregation prompt
        summaries_text = "\n\n---\n\n".join(
            f"Sub-section: {a.get('summary', '')}" for a in analyses
        )
        summaries_text = enforce_token_budget(summaries_text, settings.max_section_aggregate_tokens)

        prompt = SECTION_AGGREGATE_PROMPT.format(heading=heading, segment_summaries=summaries_text)

        try:
            response: SectionSummaryResponse = await structured_llm.ainvoke(prompt)
            section_data = {
                "section_id": section_id,
                "heading": heading,
                "summary": response.summary,
                "key_claims": response.key_claims,
                "risks": response.risks,
                "decisions": response.decisions,
                "entities": [e.model_dump() for e in response.entities],
                "child_segment_ids": seg_ids,
            }
        except Exception as e:
            logger.error("Section aggregation failed for '%s': %s", heading, e)
            section_data = {
                "section_id": section_id,
                "heading": heading,
                "summary": f"[Aggregation failed: {e}]",
                "key_claims": [],
                "risks": [],
                "decisions": [],
                "entities": [],
                "child_segment_ids": seg_ids,
            }

        repo.upsert_section_summary(doc_id, section_id, section_data)
        section_ids.append(section_id)
        logger.debug("Aggregated section '%s'", heading)

    logger.info("Aggregated %d sections for doc %s", len(section_ids), doc_id)
    return {"section_ids": section_ids}


# ---------------------------------------------------------------------------
# Pass 2: Chapters
# ---------------------------------------------------------------------------

async def aggregate_chapters_node(state: IngestionState) -> dict:
    """
    Group sections by parent_chapter (derived from segment hierarchy),
    aggregate into ChapterSummary objects.
    """
    doc_id = state["doc_id"]
    section_ids = state.get("section_ids", [])

    # Fetch all chapter groupings from segment metadata
    segments = repo.get_segments(doc_id, fields=["heading", "parent_chapter", "type"])

    # Build chapter -> section heading mapping
    chapter_to_sections: dict[str, list[str]] = defaultdict(list)
    for seg in segments:
        chapter = seg.get("parent_chapter") or seg.get("heading", "Main")
        sec = seg.get("heading", "")
        if sec and sec not in chapter_to_sections[chapter]:
            chapter_to_sections[chapter].append(sec)

    # If no chapters detected, treat all sections as one chapter
    if not chapter_to_sections or all(len(v) == 0 for v in chapter_to_sections.values()):
        chapter_to_sections = {"Document": [s for s in section_ids]}

    chapter_ids = []
    llm = get_llm("heavy")
    structured_llm = llm.with_structured_output(ChapterSummaryResponse)

    for chapter_heading, section_headings in chapter_to_sections.items():
        chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:chapter:{chapter_heading}"))

        existing = repo.get_chapter_summary(doc_id, chapter_id)
        if existing:
            chapter_ids.append(chapter_id)
            continue

        # Collect section summaries for this chapter
        all_sections = repo.list_section_summaries(doc_id, fields=["section_id", "heading", "summary"])
        relevant = [s for s in all_sections if s.get("heading") in section_headings or not section_headings]

        if not relevant:
            # Fallback: use all sections
            relevant = all_sections

        summaries_text = "\n\n---\n\n".join(
            f"Section: {s.get('heading', '')}\n{s.get('summary', '')}" for s in relevant
        )
        summaries_text = enforce_token_budget(summaries_text, settings.max_chapter_aggregate_tokens)

        prompt = CHAPTER_AGGREGATE_PROMPT.format(heading=chapter_heading, section_summaries=summaries_text)

        try:
            response: ChapterSummaryResponse = await structured_llm.ainvoke(prompt)
            chapter_data = {
                "chapter_id": chapter_id,
                "heading": chapter_heading,
                "summary": response.summary,
                "child_section_ids": [s.get("section_id", "") for s in relevant],
            }
        except Exception as e:
            logger.error("Chapter aggregation failed for '%s': %s", chapter_heading, e)
            chapter_data = {
                "chapter_id": chapter_id,
                "heading": chapter_heading,
                "summary": f"[Chapter aggregation failed: {e}]",
                "child_section_ids": [],
            }

        repo.upsert_chapter_summary(doc_id, chapter_id, chapter_data)
        chapter_ids.append(chapter_id)
        logger.debug("Aggregated chapter '%s'", chapter_heading)

    logger.info("Aggregated %d chapters for doc %s", len(chapter_ids), doc_id)
    return {"chapter_ids": chapter_ids}


# ---------------------------------------------------------------------------
# Pass 3: Document master
# ---------------------------------------------------------------------------

async def aggregate_document_node(state: IngestionState) -> dict:
    """
    Read all chapter summaries, produce a single DocumentMasterSummary.
    """
    doc_id = state["doc_id"]

    # Skip if already done
    existing = repo.get_document_summary(doc_id)
    if existing:
        return {"has_master_summary": True}

    chapters = repo.list_chapter_summaries(doc_id, fields=["heading", "summary"])

    if not chapters:
        # Fallback: use section summaries directly
        sections = repo.list_section_summaries(doc_id, fields=["heading", "summary"])
        chapters = sections

    summaries_text = "\n\n---\n\n".join(
        f"Chapter: {c.get('heading', '')}\n{c.get('summary', '')}" for c in chapters
    )
    summaries_text = enforce_token_budget(summaries_text, settings.max_document_aggregate_tokens)

    prompt = DOCUMENT_AGGREGATE_PROMPT.format(chapter_summaries=summaries_text)

    llm = get_llm("heavy")
    structured_llm = llm.with_structured_output(DocumentSummaryResponse)

    try:
        response: DocumentSummaryResponse = await structured_llm.ainvoke(prompt)
        summary_data = {
            "summary": response.summary,
            "top_entities": [e.model_dump() for e in response.top_entities],
            "top_risks": response.top_risks,
            "top_decisions": response.top_decisions,
        }
    except Exception as e:
        logger.error("Document aggregation failed: %s", e)
        summary_data = {
            "summary": f"[Document aggregation failed: {e}]",
            "top_entities": [],
            "top_risks": [],
            "top_decisions": [],
        }

    repo.upsert_document_summary(doc_id, summary_data)
    logger.info("Document master summary written for %s", doc_id)
    return {"has_master_summary": True}
