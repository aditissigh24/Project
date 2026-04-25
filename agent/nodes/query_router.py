"""
Query Router node — classify user query and route to appropriate handler.

Uses gpt-4o-mini with structured output to avoid expensive heavy-model call.
Returns to state: query_type, target_section_ids.
"""

from __future__ import annotations

import logging

from agent.llm import get_llm
from agent.prompts import QUERY_ROUTE_PROMPT
from agent.schemas import QueryRoute
from agent.state import QueryState

logger = logging.getLogger(__name__)


async def route_query_node(state: QueryState) -> dict:
    """
    LangGraph node: classify the user query into a query_type and
    optionally identify target section ids/names.
    """
    query = state["query"]

    prompt = QUERY_ROUTE_PROMPT.format(query=query)
    llm = get_llm("light")
    structured_llm = llm.with_structured_output(QueryRoute)

    try:
        result: QueryRoute = await structured_llm.ainvoke(prompt)
        logger.info("Query routed to '%s' (reasoning: %s)", result.query_type, result.reasoning)
        return {
            "query_type": result.query_type,
            "target_section_ids": result.target_section_ids or [],
        }
    except Exception as e:
        logger.error("Query routing failed: %s — falling back to open_question", e)
        return {
            "query_type": "open_question",
            "target_section_ids": [],
        }


def get_route(state: QueryState) -> str:
    """
    LangGraph conditional edge: map query_type to the next node name.
    """
    qt = state.get("query_type", "open_question")
    if qt in ("summarize_section", "summarize_document"):
        return "direct_fetch"
    if qt == "extract":
        return "extract_data"
    if qt == "find_contradictions":
        return "fetch_contradictions_node"
    return "global_reasoning"  # compare, open_question
