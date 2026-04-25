"""
LangGraph graph definitions.

Two compiled graphs:
  - build_ingestion_graph(): PDF → MongoDB artifacts pipeline
  - build_query_graph(): user query → Answer pipeline

Both graphs are compiled once and reused across invocations.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.aggregation import (
    aggregate_chapters_node,
    aggregate_document_node,
    aggregate_sections_node,
)
from agent.nodes.cleaning import clean_node
from agent.nodes.consistency import consistency_check_node
from agent.nodes.finalize import finalize_node
from agent.nodes.global_reasoning import (
    direct_fetch_node,
    extract_data_node,
    fetch_contradictions_node,
    global_reasoning_node,
)
from agent.nodes.ingestion import ingest_node
from agent.nodes.local_analysis import analyze_segment_node
from agent.nodes.query_router import get_route, route_query_node
from agent.nodes.segmentation import fan_out_local_analysis, segment_node
from agent.state import IngestionState, QueryState


# ---------------------------------------------------------------------------
# Ingestion Graph
# ---------------------------------------------------------------------------

def build_ingestion_graph():
    """
    Build and compile the ingestion pipeline graph.

    Topology:
      START → ingest → clean → segment → fan_out → analyze_segment (×N, parallel)
            → aggregate_sections → aggregate_chapters → aggregate_document
            → consistency_check → finalize → END
    """
    builder = StateGraph(IngestionState)

    # Register nodes
    builder.add_node("ingest", ingest_node)
    builder.add_node("clean", clean_node)
    builder.add_node("segment", segment_node)
    builder.add_node("analyze_segment", analyze_segment_node)
    builder.add_node("aggregate_sections", aggregate_sections_node)
    builder.add_node("aggregate_chapters", aggregate_chapters_node)
    builder.add_node("aggregate_document", aggregate_document_node)
    builder.add_node("consistency_check", consistency_check_node)
    builder.add_node("finalize", finalize_node)

    # Linear edges up to segmentation
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "clean")
    builder.add_edge("clean", "segment")

    # Fan-out: segment → N parallel analyze_segment tasks via Send
    builder.add_conditional_edges("segment", fan_out_local_analysis, ["analyze_segment"])

    # All parallel analyze_segment tasks converge at aggregate_sections
    builder.add_edge("analyze_segment", "aggregate_sections")

    # Sequential aggregation pipeline
    builder.add_edge("aggregate_sections", "aggregate_chapters")
    builder.add_edge("aggregate_chapters", "aggregate_document")
    builder.add_edge("aggregate_document", "consistency_check")
    builder.add_edge("consistency_check", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Query Graph
# ---------------------------------------------------------------------------

def build_query_graph():
    """
    Build and compile the query pipeline graph.

    Topology:
      START → route_query → [branch based on query_type]
        → direct_fetch     → END   (summarize_*)
        → extract_data     → END   (extract)
        → fetch_contradictions → END  (find_contradictions)
        → global_reasoning → END   (compare, open_question)
    """
    builder = StateGraph(QueryState)

    # Register nodes
    builder.add_node("route_query", route_query_node)
    builder.add_node("direct_fetch", direct_fetch_node)
    builder.add_node("extract_data", extract_data_node)
    builder.add_node("fetch_contradictions_node", fetch_contradictions_node)
    builder.add_node("global_reasoning", global_reasoning_node)

    # Entry
    builder.add_edge(START, "route_query")

    # Routing: map query_type → node
    builder.add_conditional_edges(
        "route_query",
        get_route,
        {
            "direct_fetch": "direct_fetch",
            "extract_data": "extract_data",
            "fetch_contradictions_node": "fetch_contradictions_node",
            "global_reasoning": "global_reasoning",
        },
    )

    # All branches terminate at END
    builder.add_edge("direct_fetch", END)
    builder.add_edge("extract_data", END)
    builder.add_edge("fetch_contradictions_node", END)
    builder.add_edge("global_reasoning", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Module-level compiled instances
# Required by LangGraph Studio (referenced in langgraph.json)
# ---------------------------------------------------------------------------

ingestion_graph = build_ingestion_graph()
query_graph = build_query_graph()


# Convenience accessors (used by main.py CLI)
def get_ingestion_graph():
    return ingestion_graph


def get_query_graph():
    return query_graph
