"""
Tests for aggregation utilities: map-reduce prompt building, token enforcement,
and the SectionSummaryResponse schema validation.
"""

import pytest
from pydantic import ValidationError

from agent.schemas import (
    ChapterSummaryResponse,
    DocumentSummaryResponse,
    Entity,
    SectionSummaryResponse,
    SegmentAnalysis,
)
from agent.tools.text_tools import count_tokens, enforce_token_budget


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_segment_analysis_valid(self):
        data = {
            "summary": "A " * 150,
            "key_entities": [{"name": "ICICI Bank", "type": "organization", "context": "primary subject"}],
            "key_claims": ["Net profit increased by 12%"],
            "decisions": ["Expand retail banking"],
            "risks": ["Credit risk from NPA"],
            "contradictions": [],
            "topics": ["banking", "finance"],
            "sentiment": "positive",
        }
        analysis = SegmentAnalysis(**data)
        assert analysis.sentiment == "positive"
        assert len(analysis.key_entities) == 1

    def test_segment_analysis_defaults(self):
        analysis = SegmentAnalysis(summary="Brief summary.")
        assert analysis.key_entities == []
        assert analysis.sentiment == "neutral"

    def test_entity_types(self):
        valid_types = ["person", "organization", "location", "date", "metric", "other"]
        for t in valid_types:
            e = Entity(name="Test", type=t)
            assert e.type == t

    def test_section_summary_response(self):
        r = SectionSummaryResponse(
            summary="Section summary text.",
            key_claims=["Claim 1", "Claim 2"],
        )
        assert len(r.key_claims) == 2

    def test_chapter_summary_response(self):
        r = ChapterSummaryResponse(summary="Chapter synthesis.")
        assert "Chapter" in r.summary

    def test_document_summary_response(self):
        r = DocumentSummaryResponse(
            summary="Master document summary.",
            top_risks=["Market risk", "Liquidity risk"],
            top_decisions=["Digital expansion"],
        )
        assert len(r.top_risks) == 2


# ---------------------------------------------------------------------------
# Aggregation prompt building (unit test: token budget applied correctly)
# ---------------------------------------------------------------------------

class TestAggregationTokenBudget:
    def _make_segment_analyses(self, n: int, summary_length: int = 200) -> list[dict]:
        return [
            {
                "segment_id": f"seg-{i}",
                "summary": f"Summary of segment {i}. " + ("detail " * summary_length),
                "key_claims": [f"Claim {i}.a", f"Claim {i}.b"],
            }
            for i in range(n)
        ]

    def test_aggregation_input_within_budget(self):
        """Ensure that assembling N segment summaries and then budget-capping fits."""
        analyses = self._make_segment_analyses(100, summary_length=30)
        combined = "\n\n---\n\n".join(a["summary"] for a in analyses)
        budget = 20_000  # section aggregate budget
        budgeted = enforce_token_budget(combined, budget)
        assert count_tokens(budgeted) <= budget + 10  # small tolerance

    def test_small_input_not_truncated(self):
        analyses = self._make_segment_analyses(3, summary_length=5)
        combined = "\n\n---\n\n".join(a["summary"] for a in analyses)
        budget = 20_000
        budgeted = enforce_token_budget(combined, budget)
        assert budgeted == combined  # should be unchanged
