"""
Embedding and clustering utilities for the Consistency Checker.
Embeddings are used ONLY for grouping claims by topic — not for retrieval.
This preserves the no-RAG constraint.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from agent.llm import get_embeddings


def embed_claims(claims: list[str]) -> np.ndarray:
    """
    Return a (N, D) float32 array of embeddings for each claim.
    Uses OpenAI text-embedding-3-small.
    """
    if not claims:
        return np.zeros((0, 1536), dtype=np.float32)
    embeddings_model = get_embeddings()
    vectors = embeddings_model.embed_documents(claims)
    return np.array(vectors, dtype=np.float32)


def cluster_claims_by_topic(
    claims: list[str],
    embeddings: np.ndarray,
    similarity_threshold: float = 0.78,
) -> list[list[int]]:
    """
    Group claim indices by topic using agglomerative clustering on cosine similarity.

    Returns a list of clusters, each cluster being a list of claim indices.
    Claims that are semantically similar (cosine similarity >= threshold) are grouped together.
    """
    n = len(claims)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Convert cosine similarity to distance for agglomerative clustering
    sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1.0 - sim_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

    distance_threshold = 1.0 - similarity_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    # Only return clusters with 2+ claims (single-claim clusters can't have contradictions)
    return [indices for indices in clusters.values() if len(indices) >= 2]
