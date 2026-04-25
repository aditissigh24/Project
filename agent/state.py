"""
LangGraph state definitions.

Key principle: state holds ONLY ids, counts, flags, and final small outputs.
All heavy text artifacts (page text, cleaned text, segment text, full analyses)
are written to MongoDB by the producing node and read back by id in consuming nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.schemas import Answer


class IngestionState(TypedDict):
    # Inputs
    doc_id: str
    pdf_path: str
    run_id: str

    # Populated by ingest_node after writing pages -> Mongo
    page_count: int
    pages_ocr_used: int

    # Set True by clean_node after writing cleaned_text back onto pages in Mongo
    cleaned: bool

    # Set by segment_node after writing segments -> Mongo
    segment_ids: list[str]

    # Reducer: analyze_segment_node returns a single-item list; operator.add merges them
    analyzed_segment_ids: Annotated[list[str], operator.add]

    # Set by aggregation nodes after writing summaries -> Mongo
    section_ids: list[str]
    chapter_ids: list[str]
    has_master_summary: bool

    # Set by consistency_check_node
    contradiction_count: int

    # Append-only error log (also written to runs collection)
    errors: Annotated[list[str], operator.add]


class QueryState(TypedDict):
    # Inputs
    doc_id: str
    query: str
    run_id: str

    # Set by route_query_node
    query_type: Literal[
        "summarize_section",
        "summarize_document",
        "compare",
        "extract",
        "find_contradictions",
        "open_question",
    ] | None
    target_section_ids: list[str] | None

    # ReAct message history for global_reasoning_node
    messages: Annotated[list[AnyMessage], add_messages]

    # Final answer (small)
    answer: Answer | None
