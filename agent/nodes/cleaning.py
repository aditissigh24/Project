"""
Cleaning node — rule-based + optional LLM repair for low-confidence pages.

Reads from Mongo: pages collection (raw_text, confidence)
Writes to Mongo: pages.cleaned_text
Returns to state: cleaned=True
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.prompts import CLEAN_REPAIR_PROMPT
from agent.state import IngestionState
from agent.tools.text_tools import (
    count_tokens,
    enforce_token_budget,
    rule_based_clean,
    text_needs_llm_repair,
)
from config import settings
import db.repositories as repo

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20
_LOW_CONFIDENCE_THRESHOLD = 0.75


async def _repair_with_llm(noisy_text: str) -> str:
    """Call gpt-4o-mini to fix OCR noise. Budget-capped."""
    budgeted = enforce_token_budget(noisy_text, settings.max_clean_repair_tokens)
    prompt = CLEAN_REPAIR_PROMPT.format(noisy_text=budgeted)
    llm = get_llm("light")
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


async def clean_node(state: IngestionState) -> dict:
    """
    LangGraph node: stream pages from Mongo in batches, apply rule-based cleaning,
    run LLM repair on low-confidence pages, write cleaned_text back to Mongo.
    """
    doc_id = state["doc_id"]
    pages = repo.get_pages(doc_id, fields=["page_num", "raw_text", "confidence"])

    errors: list[str] = []

    for i in range(0, len(pages), _BATCH_SIZE):
        batch = pages[i : i + _BATCH_SIZE]
        for page in batch:
            page_num = page["page_num"]
            raw_text = page.get("raw_text", "")
            confidence = page.get("confidence", 1.0)

            try:
                cleaned = rule_based_clean(raw_text)

                if confidence < _LOW_CONFIDENCE_THRESHOLD and text_needs_llm_repair(cleaned):
                    logger.debug("LLM repair on page %d (conf=%.2f)", page_num, confidence)
                    cleaned = await _repair_with_llm(cleaned)

                repo.update_page_cleaned_text(doc_id, page_num, cleaned)

            except Exception as e:
                msg = f"Cleaning failed for page {page_num}: {e}"
                logger.error(msg)
                errors.append(msg)
                # Fallback: store rule-cleaned version
                try:
                    repo.update_page_cleaned_text(doc_id, page_num, rule_based_clean(raw_text))
                except Exception:
                    pass

    logger.info("Cleaned %d pages for doc %s", len(pages), doc_id)

    update: dict = {"cleaned": True}
    if errors:
        update["errors"] = errors
    return update
