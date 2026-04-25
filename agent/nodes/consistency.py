"""
Consistency Checker node — detect cross-document contradictions.

Reads from Mongo: segment_analyses.key_claims (lightweight field only)
Writes to Mongo: contradictions collection
Returns to state: contradiction_count

NOTE: embeddings are used ONLY for grouping claims by topic (not retrieval).
This preserves the no-RAG constraint.
"""

from __future__ import annotations

import logging

from agent.llm import get_llm
from agent.prompts import CONTRADICTION_CHECK_PROMPT
from agent.schemas import ContradictionList
from agent.state import IngestionState
from agent.tools.embedding_tools import cluster_claims_by_topic, embed_claims
from agent.tools.text_tools import enforce_token_budget
import db.repositories as repo

logger = logging.getLogger(__name__)

_MAX_CLAIMS_PER_CLUSTER = 30     # cap to avoid huge prompts
_MAX_TOKENS_PER_CHECK = 15_000


async def consistency_check_node(state: IngestionState) -> dict:
    """
    LangGraph node: collect all claims, embed + cluster by topic,
    then LLM-check each cluster for contradictions.
    """
    doc_id = state["doc_id"]

    # Fetch key_claims and segment metadata (lightweight)
    analyses = repo.get_all_analyses(doc_id, fields=["segment_id", "key_claims"])
    if not analyses:
        logger.info("No analyses found for consistency check on doc %s", doc_id)
        return {"contradiction_count": 0}

    # Build a flat list of claims with their source segment_id
    all_claims: list[str] = []
    claim_sources: list[str] = []   # parallel: segment_id for each claim

    for analysis in analyses:
        seg_id = analysis.get("segment_id", "unknown")
        for claim in analysis.get("key_claims", []):
            if claim and isinstance(claim, str) and claim.strip():
                all_claims.append(claim.strip())
                claim_sources.append(seg_id)

    if len(all_claims) < 2:
        logger.info("Too few claims (%d) for contradiction checking", len(all_claims))
        return {"contradiction_count": 0}

    logger.info("Embedding %d claims for consistency check", len(all_claims))

    try:
        embeddings = embed_claims(all_claims)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return {"contradiction_count": 0, "errors": [f"Embedding failed: {e}"]}

    clusters = cluster_claims_by_topic(all_claims, embeddings, similarity_threshold=0.78)
    logger.info("Found %d topic clusters", len(clusters))

    all_contradictions: list[dict] = []
    llm = get_llm("heavy")
    structured_llm = llm.with_structured_output(ContradictionList)

    for cluster_idx, claim_indices in enumerate(clusters):
        # Cap cluster size to avoid huge prompts
        if len(claim_indices) > _MAX_CLAIMS_PER_CLUSTER:
            claim_indices = claim_indices[:_MAX_CLAIMS_PER_CLUSTER]

        cluster_claims = [all_claims[i] for i in claim_indices]
        cluster_sources = [claim_sources[i] for i in claim_indices]

        # Build numbered claims list for the prompt
        claims_text = "\n".join(
            f"{j+1}. [{cluster_sources[j]}] {cluster_claims[j]}"
            for j in range(len(cluster_claims))
        )
        claims_text = enforce_token_budget(claims_text, _MAX_TOKENS_PER_CHECK)

        prompt = CONTRADICTION_CHECK_PROMPT.format(claims=claims_text)

        try:
            result: ContradictionList = await structured_llm.ainvoke(prompt)
            for c in result.contradictions:
                all_contradictions.append(c.model_dump())
        except Exception as e:
            logger.warning("Contradiction check failed for cluster %d: %s", cluster_idx, e)

    count = repo.insert_contradictions(doc_id, all_contradictions)
    logger.info("Found %d contradictions in doc %s", count, doc_id)
    return {"contradiction_count": count}
