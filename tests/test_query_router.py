"""
Tests for the query router schema and get_route conditional edge.
These tests mock the LLM so no API key is needed.
"""

import pytest
from pydantic import ValidationError

from agent.schemas import Answer, Citation, QueryRoute


# ---------------------------------------------------------------------------
# QueryRoute schema
# ---------------------------------------------------------------------------

class TestQueryRoute:
    def test_valid_query_types(self):
        valid_types = [
            "summarize_section",
            "summarize_document",
            "compare",
            "extract",
            "find_contradictions",
            "open_question",
        ]
        for qt in valid_types:
            route = QueryRoute(query_type=qt)
            assert route.query_type == qt

    def test_invalid_query_type(self):
        with pytest.raises(ValidationError):
            QueryRoute(query_type="invalid_type")

    def test_target_section_ids_optional(self):
        route = QueryRoute(query_type="summarize_document")
        assert route.target_section_ids is None

    def test_target_section_ids_provided(self):
        route = QueryRoute(query_type="compare", target_section_ids=["risk-management", "financials"])
        assert len(route.target_section_ids) == 2


# ---------------------------------------------------------------------------
# get_route conditional edge
# ---------------------------------------------------------------------------

class TestGetRoute:
    def _state(self, query_type: str) -> dict:
        return {"query_type": query_type, "doc_id": "d1", "query": "test"}

    def test_summarize_document_routes_to_direct_fetch(self):
        from agent.nodes.query_router import get_route
        assert get_route(self._state("summarize_document")) == "direct_fetch"

    def test_summarize_section_routes_to_direct_fetch(self):
        from agent.nodes.query_router import get_route
        assert get_route(self._state("summarize_section")) == "direct_fetch"

    def test_extract_routes_to_extract_data(self):
        from agent.nodes.query_router import get_route
        assert get_route(self._state("extract")) == "extract_data"

    def test_find_contradictions_routes_correctly(self):
        from agent.nodes.query_router import get_route
        assert get_route(self._state("find_contradictions")) == "fetch_contradictions_node"

    def test_compare_routes_to_global_reasoning(self):
        from agent.nodes.query_router import get_route
        assert get_route(self._state("compare")) == "global_reasoning"

    def test_open_question_routes_to_global_reasoning(self):
        from agent.nodes.query_router import get_route
        assert get_route(self._state("open_question")) == "global_reasoning"

    def test_none_query_type_falls_back(self):
        from agent.nodes.query_router import get_route
        assert get_route(self._state(None)) == "global_reasoning"


# ---------------------------------------------------------------------------
# Answer schema
# ---------------------------------------------------------------------------

class TestAnswerSchema:
    def test_answer_valid(self):
        a = Answer(answer="Some answer text.", confidence=0.9, query_type="open_question")
        assert a.confidence == 0.9
        assert a.citations == []

    def test_answer_with_citations(self):
        a = Answer(
            answer="An answer with citations.",
            citations=[
                Citation(section_heading="Risk Management", page_range=[45, 67]),
                Citation(segment_id="seg-abc123"),
            ],
        )
        assert len(a.citations) == 2
        assert a.citations[0].section_heading == "Risk Management"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Answer(answer="test", confidence=1.5)

        with pytest.raises(ValidationError):
            Answer(answer="test", confidence=-0.1)
