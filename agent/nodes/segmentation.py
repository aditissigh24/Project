"""
Segmentation node — detect logical document boundaries, produce segment manifest.

Reads from Mongo: pages.cleaned_text
Writes to Mongo: segments collection
Returns to state: segment_ids
Also defines fan_out_local_analysis (Send dispatcher for parallel analysis).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from itertools import groupby

from langchain_core.messages import HumanMessage
from langgraph.types import Send

from agent.llm import get_llm
from agent.prompts import SEGMENT_BOUNDARY_PROMPT
from agent.state import IngestionState
from agent.tools.text_tools import count_tokens, enforce_token_budget
from config import settings
import db.repositories as repo

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 20    # pages per sliding window
_OVERLAP = 2         # overlap pages between consecutive windows

# Heading patterns for financial/annual reports
_HEADING_PATTERNS = [
    re.compile(r"^(?:CHAPTER|SECTION)\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\d+\.\s+[A-Z][A-Za-z\s]{4,}$", re.MULTILINE),
    re.compile(r"^\d+\.\d+\s+[A-Z][A-Za-z\s]{4,}$", re.MULTILINE),
    re.compile(r"^[A-Z][A-Z\s]{8,}$", re.MULTILINE),       # ALL CAPS headings
    # Common ICICI Bank / annual report headings
    re.compile(
        r"^(?:Chairman(?:'s)? Message|Managing Director|CEO|Board of Directors|"
        r"Directors'? Report|Corporate Governance|Risk Management|"
        r"Financial Statements|Balance Sheet|Profit and Loss|Cash Flow|"
        r"Notes to (?:the )?(?:Financial )?Statements|Auditor(?:'s)? Report|"
        r"Business Overview|Strategic Priorities|Performance Highlights|"
        r"Capital Adequacy|Asset Quality|Net Interest|Key Metrics)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def _detect_headings_regex(pages: list[dict]) -> list[dict]:
    """
    Fast heuristic pass to detect section boundaries using regex patterns.
    Returns list of {page_num, heading, type}.
    """
    boundaries = []
    for page in pages:
        text = page.get("cleaned_text") or page.get("raw_text", "")
        page_num = page["page_num"]
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for pattern in _HEADING_PATTERNS:
                if pattern.search(line):
                    heading = line[:120]  # cap heading length
                    section_type = "chapter" if re.match(r"^(?:CHAPTER|\d+\.\s+)", heading, re.IGNORECASE) else "section"
                    boundaries.append({"page_num": page_num, "heading": heading, "type": section_type})
                    break  # one boundary per page
    return boundaries


async def _detect_headings_llm(pages: list[dict], start_page: int, end_page: int) -> list[dict]:
    """LLM pass on ambiguous windows to find additional or corrected boundaries."""
    text_parts = []
    for p in pages:
        t = p.get("cleaned_text") or p.get("raw_text", "")
        text_parts.append(f"--- Page {p['page_num']} ---\n{t[:2000]}")

    combined = "\n\n".join(text_parts)
    combined = enforce_token_budget(combined, 15_000)

    prompt = SEGMENT_BOUNDARY_PROMPT.format(
        start_page=start_page, end_page=end_page, text=combined
    )
    llm = get_llm("light")
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    try:
        content = response.content.strip()
        # Extract JSON array from response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return []


def _build_segments(
    pages: list[dict],
    boundaries: list[dict],
    doc_id: str,
) -> list[dict]:
    """
    Given sorted page data and boundary markers, construct segment dicts.
    Each segment text is capped at max_segment_tokens.
    """
    if not boundaries:
        # Treat entire doc as one segment
        all_text = "\n\n".join(
            p.get("cleaned_text") or p.get("raw_text", "") for p in pages
        )
        return [_make_segment(doc_id, pages[0]["page_num"], pages[-1]["page_num"], "Document", "section", all_text, None, None)]

    # Sort and deduplicate boundaries by page_num
    boundaries = sorted({b["page_num"]: b for b in boundaries}.values(), key=lambda x: x["page_num"])

    segments = []
    for i, boundary in enumerate(boundaries):
        start_page = boundary["page_num"]
        end_page = boundaries[i + 1]["page_num"] - 1 if i + 1 < len(boundaries) else pages[-1]["page_num"]
        heading = boundary.get("heading", f"Section {i + 1}")
        seg_type = boundary.get("type", "section")

        section_pages = [p for p in pages if start_page <= p["page_num"] <= end_page]
        text = "\n\n".join(p.get("cleaned_text") or p.get("raw_text", "") for p in section_pages)
        text = enforce_token_budget(text, settings.max_segment_tokens)

        # Determine parent hierarchy
        parent_chapter = None
        parent_section = None
        if seg_type == "section":
            # Find most recent chapter boundary
            for j in range(i - 1, -1, -1):
                if boundaries[j].get("type") == "chapter":
                    parent_chapter = boundaries[j]["heading"]
                    break
        elif seg_type == "subsection":
            for j in range(i - 1, -1, -1):
                if boundaries[j].get("type") in ("section", "chapter"):
                    parent_section = boundaries[j]["heading"]
                    break

        seg = _make_segment(doc_id, start_page, end_page, heading, seg_type, text, parent_section, parent_chapter)
        segments.append(seg)

    return segments


def _make_segment(
    doc_id: str,
    start_page: int,
    end_page: int,
    heading: str,
    seg_type: str,
    text: str,
    parent_section: str | None,
    parent_chapter: str | None,
) -> dict:
    return {
        "segment_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "heading": heading,
        "type": seg_type,
        "page_range": [start_page, end_page],
        "token_count": count_tokens(text),
        "text": text,
        "parent_section": parent_section,
        "parent_chapter": parent_chapter,
    }


async def segment_node(state: IngestionState) -> dict:
    """
    LangGraph node: detect section boundaries, build segments, write to Mongo.
    Returns segment_ids only.
    """
    doc_id = state["doc_id"]
    pages = repo.get_pages(doc_id, fields=["page_num", "cleaned_text", "raw_text"])

    # Regex pass (fast)
    regex_boundaries = _detect_headings_regex(pages)

    # LLM pass in sliding windows for ambiguous sections
    llm_boundaries: list[dict] = []
    for i in range(0, len(pages), _WINDOW_SIZE - _OVERLAP):
        window = pages[i : i + _WINDOW_SIZE]
        if not window:
            break
        start = window[0]["page_num"]
        end = window[-1]["page_num"]
        try:
            found = await _detect_headings_llm(window, start, end)
            llm_boundaries.extend(found)
        except Exception as e:
            logger.warning("LLM boundary detection failed for pages %d-%d: %s", start, end, e)

    # Merge: prefer regex boundaries, add LLM ones that aren't already covered
    all_page_nums = {b["page_num"] for b in regex_boundaries}
    for b in llm_boundaries:
        if b.get("page_num") and b["page_num"] not in all_page_nums:
            regex_boundaries.append(b)

    segments = _build_segments(pages, regex_boundaries, doc_id)

    segment_ids = []
    for seg in segments:
        sid = repo.insert_segment(doc_id, seg)
        segment_ids.append(sid)

    logger.info("Segmented doc %s into %d segments", doc_id, len(segment_ids))
    return {"segment_ids": segment_ids}


def fan_out_local_analysis(state: IngestionState) -> list[Send]:
    """
    LangGraph conditional edge function: dispatch one Send per segment_id.
    Payloads are id-only — no text in the channel to keep memory bounded.
    """
    return [
        Send("analyze_segment", {"doc_id": state["doc_id"], "segment_id": sid, "run_id": state.get("run_id", "")})
        for sid in state["segment_ids"]
    ]
