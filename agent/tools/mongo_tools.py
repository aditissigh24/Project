"""
LangChain @tool-decorated functions for the GlobalReasoningAgent's ToolNode.
These are the ONLY tools the LLM can call; all data is fetched from MongoDB.
"""

from __future__ import annotations

from langchain_core.tools import tool

import db.repositories as repo


@tool
def fetch_master_summary(doc_id: str) -> dict:
    """
    Fetch the master document summary for the given document.
    Always call this first to get the big picture before drilling into sections.

    Args:
        doc_id: The document identifier.

    Returns:
        A dict with keys: summary, top_entities, top_risks, top_decisions.
        Returns an error dict if not found.
    """
    result = repo.get_document_summary(doc_id)
    if result is None:
        return {"error": f"No master summary found for document {doc_id}"}
    result.pop("_id", None)
    return result


@tool
def list_sections(doc_id: str) -> list[dict]:
    """
    List all sections of the document with their headings and token counts.
    Use this to discover which sections exist before fetching their summaries.

    Args:
        doc_id: The document identifier.

    Returns:
        A list of dicts with keys: section_id, heading, child_segment_ids.
    """
    sections = repo.list_section_summaries(doc_id, fields=["section_id", "heading", "child_segment_ids"])
    return sections or []


@tool
def fetch_section_summary(doc_id: str, section_id: str) -> dict:
    """
    Fetch the detailed summary for a specific section of the document.

    Args:
        doc_id: The document identifier.
        section_id: The section identifier (get these from list_sections).

    Returns:
        A dict with keys: heading, summary, key_claims, entities, risks, decisions.
        Returns an error dict if not found.
    """
    result = repo.get_section_summary(doc_id, section_id)
    if result is None:
        # Try to find by heading (partial match)
        all_sections = repo.list_section_summaries(doc_id)
        for s in all_sections:
            if section_id.lower() in s.get("heading", "").lower():
                result = s
                break
    if result is None:
        return {"error": f"Section '{section_id}' not found in document {doc_id}"}
    result.pop("_id", None)
    return result


@tool
def fetch_segment_analysis(doc_id: str, segment_id: str) -> dict:
    """
    Fetch the deep analysis of a specific segment (sub-section) for detailed information.
    Use this when section-level summaries are insufficient for answering the question.

    Args:
        doc_id: The document identifier.
        segment_id: The segment identifier.

    Returns:
        A dict with keys: summary, key_entities, key_claims, decisions, risks, topics, sentiment.
        Returns an error dict if not found.
    """
    result = repo.get_segment_analysis(doc_id, segment_id)
    if result is None:
        return {"error": f"Segment analysis for '{segment_id}' not found in document {doc_id}"}
    result.pop("_id", None)
    return result


@tool
def fetch_entities(doc_id: str, entity_types: list[str] | None = None) -> list[dict]:
    """
    Fetch all extracted entities from the document, optionally filtered by type.

    Args:
        doc_id: The document identifier.
        entity_types: Optional list of types to filter by.
                      Valid types: person, organization, location, date, metric, other.
                      If None, returns all entities.

    Returns:
        A list of dicts with keys: name, type, context.
    """
    return repo.get_all_entities(doc_id, types=entity_types)


@tool
def fetch_risks(doc_id: str) -> list[str]:
    """
    Fetch all identified risks from across the entire document.

    Args:
        doc_id: The document identifier.

    Returns:
        A list of risk description strings.
    """
    return repo.get_all_risks(doc_id)


@tool
def fetch_contradictions(doc_id: str) -> list[dict]:
    """
    Fetch all detected contradictions and inconsistencies in the document.

    Args:
        doc_id: The document identifier.

    Returns:
        A list of dicts with keys: claim_a, claim_b, section_a, section_b, explanation, severity.
    """
    results = repo.get_contradictions(doc_id)
    for r in results:
        r.pop("_id", None)
    return results


# Collect all tools for easy import
ALL_MONGO_TOOLS = [
    fetch_master_summary,
    list_sections,
    fetch_section_summary,
    fetch_segment_analysis,
    fetch_entities,
    fetch_risks,
    fetch_contradictions,
]
