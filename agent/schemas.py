"""
Pydantic schemas used across all agents.
These are the canonical shapes for data flowing in/out of LLM structured-output calls
and for data stored in MongoDB.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Page / OCR
# ---------------------------------------------------------------------------

class PageData(BaseModel):
    page_num: int
    raw_text: str
    ocr_used: bool
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

class Segment(BaseModel):
    segment_id: str
    doc_id: str
    type: Literal["chapter", "section", "subsection", "paragraph"] = "section"
    heading: str
    page_range: list[int] = Field(..., description="[start_page, end_page] inclusive")
    token_count: int
    text: str
    parent_section: str | None = None
    parent_chapter: str | None = None


# ---------------------------------------------------------------------------
# Local Analysis
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    name: str
    type: Literal["person", "organization", "location", "date", "metric", "other"]
    context: str = ""


class SegmentAnalysis(BaseModel):
    summary: str = Field(..., description="200-300 word dense summary of the section")
    key_entities: list[Entity] = Field(default_factory=list)
    key_claims: list[str] = Field(default_factory=list, description="Important factual assertions")
    decisions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list, description="Internal inconsistencies in this segment")
    topics: list[str] = Field(default_factory=list, description="2-5 topic tags")
    sentiment: Literal["neutral", "positive", "negative", "mixed"] = "neutral"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class SectionSummary(BaseModel):
    section_id: str
    heading: str
    summary: str
    key_claims: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    child_segment_ids: list[str] = Field(default_factory=list)


class ChapterSummary(BaseModel):
    chapter_id: str
    heading: str
    summary: str
    child_section_ids: list[str] = Field(default_factory=list)


class DocumentMasterSummary(BaseModel):
    summary: str = Field(..., description="~2000 word comprehensive document summary")
    top_entities: list[Entity] = Field(default_factory=list)
    top_risks: list[str] = Field(default_factory=list)
    top_decisions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------

class Contradiction(BaseModel):
    claim_a: str
    claim_b: str
    section_a: str
    section_b: str
    explanation: str
    severity: Literal["low", "medium", "high"] = "medium"


class ContradictionList(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

class QueryRoute(BaseModel):
    query_type: Literal[
        "summarize_section",
        "summarize_document",
        "compare",
        "extract",
        "find_contradictions",
        "open_question",
    ]
    target_section_ids: list[str] | None = Field(
        default=None,
        description="Relevant section ids if determinable from the query",
    )
    reasoning: str = Field(default="", description="Brief reasoning for the classification")


# ---------------------------------------------------------------------------
# Query answer
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    segment_id: str | None = None
    section_heading: str | None = None
    page_range: list[int] | None = None


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    query_type: str = ""


# ---------------------------------------------------------------------------
# Aggregation LLM response schemas
# ---------------------------------------------------------------------------

class SectionSummaryResponse(BaseModel):
    summary: str
    key_claims: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)


class ChapterSummaryResponse(BaseModel):
    summary: str


class DocumentSummaryResponse(BaseModel):
    summary: str
    top_entities: list[Entity] = Field(default_factory=list)
    top_risks: list[str] = Field(default_factory=list)
    top_decisions: list[str] = Field(default_factory=list)
