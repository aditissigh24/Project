"""
Global Reasoning node — ReAct agent with ToolNode for complex queries.

Uses create_react_agent which internally manages the ToolNode loop.
The agent reads from MongoDB via @tool-decorated functions in mongo_tools.py.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agent.llm import get_llm
from agent.prompts import GLOBAL_REASONING_SYSTEM
from agent.schemas import Answer, Citation
from agent.state import QueryState
from agent.tools.mongo_tools import ALL_MONGO_TOOLS

logger = logging.getLogger(__name__)


def _build_react_agent(doc_id: str):
    """Build a fresh ReAct agent for the given doc_id context."""
    llm = get_llm("heavy")
    system_prompt = GLOBAL_REASONING_SYSTEM.format(doc_id=doc_id)

    agent = create_react_agent(
        model=llm,
        tools=ALL_MONGO_TOOLS,
        prompt=system_prompt,
    )
    return agent


async def global_reasoning_node(state: QueryState) -> dict:
    """
    LangGraph node: run the ReAct agent to answer complex queries.

    The ReAct loop:
    1. Receives user query + doc_id context
    2. Calls mongo_tools to retrieve summaries and analyses
    3. Synthesizes a final answer with citations
    """
    doc_id = state["doc_id"]
    query = state["query"]
    query_type = state.get("query_type", "open_question")
    target_section_ids = state.get("target_section_ids") or []

    # Build the user message with context hints
    user_content = query
    if target_section_ids:
        user_content += f"\n\n(Relevant sections to focus on: {', '.join(target_section_ids)})"

    agent = _build_react_agent(doc_id)

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_content)]},
        )
        # Extract the final AI message
        final_messages = result.get("messages", [])
        final_text = ""
        for msg in reversed(final_messages):
            if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        answer = Answer(
            answer=final_text,
            citations=[],  # Citations are embedded in the text by the LLM via system prompt
            confidence=0.85,
            query_type=query_type,
        )
    except Exception as e:
        logger.error("Global reasoning failed: %s", e)
        answer = Answer(
            answer=f"An error occurred while processing your query: {e}",
            citations=[],
            confidence=0.0,
            query_type=query_type,
        )

    return {"answer": answer}


async def direct_fetch_node(state: QueryState) -> dict:
    """
    LangGraph node: for summarize_* queries, fetch pre-computed summaries directly.
    No LLM call needed — the summaries are already in MongoDB.
    """
    import db.repositories as repo

    doc_id = state["doc_id"]
    query_type = state.get("query_type", "summarize_document")
    target_ids = state.get("target_section_ids") or []

    if query_type == "summarize_document":
        summary_doc = repo.get_document_summary(doc_id)
        if summary_doc:
            text = summary_doc.get("summary", "No summary available.")
        else:
            text = "Document summary not yet generated. Please run the ingestion pipeline first."
        answer = Answer(answer=text, citations=[], confidence=1.0, query_type=query_type)

    elif query_type == "summarize_section":
        if target_ids:
            parts = []
            for sid in target_ids:
                sec = repo.get_section_summary(doc_id, sid)
                if sec:
                    parts.append(f"**{sec.get('heading', sid)}**\n\n{sec.get('summary', '')}")
                else:
                    # Try to fuzzy-find by heading
                    all_secs = repo.list_section_summaries(doc_id)
                    for s in all_secs:
                        if sid.lower() in s.get("heading", "").lower():
                            parts.append(f"**{s.get('heading', sid)}**\n\n{s.get('summary', '')}")
                            break
            text = "\n\n---\n\n".join(parts) if parts else "Specified section(s) not found."
        else:
            # No specific section requested — fall back to document summary
            summary_doc = repo.get_document_summary(doc_id)
            text = summary_doc.get("summary", "No summary available.") if summary_doc else "No summary available."

        answer = Answer(answer=text, citations=[], confidence=1.0, query_type=query_type)

    else:
        answer = Answer(answer="Unexpected query type in direct_fetch_node.", citations=[], confidence=0.0)

    return {"answer": answer}


async def extract_data_node(state: QueryState) -> dict:
    """
    LangGraph node: for extract queries, aggregate entities/risks/decisions
    and return a structured response. One LLM call to format the result.
    """
    import json

    from langchain_core.messages import HumanMessage

    import db.repositories as repo

    doc_id = state["doc_id"]
    query = state["query"]

    entities = repo.get_all_entities(doc_id)
    risks = repo.get_all_risks(doc_id)
    decisions = repo.get_all_decisions(doc_id)

    structured_data = {
        "entities": entities[:100],   # cap to avoid huge responses
        "risks": risks[:50],
        "decisions": decisions[:50],
    }

    # Use LLM to format the response to match the specific user query
    prompt = f"""\
The user asked: {query}

Below is structured data extracted from the document:

{json.dumps(structured_data, indent=2)}

Produce a clear, well-organized response that directly addresses the user's question.
Include only the data that is relevant to the query. Format using markdown for readability.
"""
    llm = get_llm("heavy")
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content
    except Exception as e:
        text = f"Extraction failed: {e}"

    answer = Answer(answer=text, citations=[], confidence=0.9, query_type="extract")
    return {"answer": answer}


async def fetch_contradictions_node(state: QueryState) -> dict:
    """
    LangGraph node: for find_contradictions queries, fetch pre-computed contradictions
    and format them. One optional LLM call to focus on query-relevant ones.
    """
    import db.repositories as repo

    doc_id = state["doc_id"]
    query = state["query"]
    contradictions = repo.get_contradictions(doc_id)

    if not contradictions:
        answer = Answer(
            answer="No contradictions or inconsistencies were detected in this document.",
            citations=[],
            confidence=1.0,
            query_type="find_contradictions",
        )
        return {"answer": answer}

    # Format contradictions as markdown
    lines = ["# Detected Inconsistencies\n"]
    for i, c in enumerate(contradictions, 1):
        severity = c.get("severity", "medium").upper()
        lines.append(
            f"## {i}. [{severity}] Contradiction\n\n"
            f"**Claim A** (from `{c.get('section_a', 'unknown')}`):\n> {c.get('claim_a', '')}\n\n"
            f"**Claim B** (from `{c.get('section_b', 'unknown')}`):\n> {c.get('claim_b', '')}\n\n"
            f"**Explanation**: {c.get('explanation', '')}\n"
        )

    text = "\n".join(lines)

    # If the query is focused on a specific topic, add LLM filtering pass
    if len(contradictions) > 5 and query.strip().lower() not in ("find contradictions", "show contradictions"):
        from langchain_core.messages import HumanMessage
        filter_prompt = f"""\
The user specifically asked: {query}

Below are all detected contradictions. Select and highlight only those most relevant \
to the user's question, and explain the relevance:

{text}
"""
        llm = get_llm("light")
        try:
            response = await llm.ainvoke([HumanMessage(content=filter_prompt)])
            text = response.content
        except Exception:
            pass  # Use unfiltered text if LLM call fails

    answer = Answer(
        answer=text,
        citations=[
            Citation(section_heading=c.get("section_a"), page_range=None) for c in contradictions[:10]
        ],
        confidence=1.0,
        query_type="find_contradictions",
    )
    return {"answer": answer}
