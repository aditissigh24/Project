"""
Tests for segmentation utilities: heading regex detection, segment building,
token budget enforcement.
"""

import pytest

from agent.nodes.segmentation import (
    _detect_headings_regex,
    _build_segments,
    _make_segment,
)
from agent.tools.text_tools import count_tokens, enforce_token_budget, rule_based_clean


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

class TestHeadingDetection:
    def _make_pages(self, texts: list[str]) -> list[dict]:
        return [{"page_num": i + 1, "cleaned_text": t} for i, t in enumerate(texts)]

    def test_all_caps_heading(self):
        pages = self._make_pages(["RISK MANAGEMENT\nSome content here."])
        boundaries = _detect_headings_regex(pages)
        assert any("RISK MANAGEMENT" in b["heading"] for b in boundaries)

    def test_numbered_heading(self):
        pages = self._make_pages(["1. Financial Overview\nThis section covers..."])
        boundaries = _detect_headings_regex(pages)
        assert len(boundaries) >= 1

    def test_common_annual_report_heading(self):
        pages = self._make_pages(["Chairman's Message\nDear Shareholders,"])
        boundaries = _detect_headings_regex(pages)
        assert any("Chairman" in b["heading"] for b in boundaries)

    def test_balance_sheet_heading(self):
        pages = self._make_pages(["Balance Sheet\nAs at March 31, 2024"])
        boundaries = _detect_headings_regex(pages)
        assert any("Balance Sheet" in b["heading"] for b in boundaries)

    def test_no_heading_plain_text(self):
        pages = self._make_pages(["This is a regular paragraph with no heading pattern."])
        boundaries = _detect_headings_regex(pages)
        assert len(boundaries) == 0

    def test_multiple_pages(self):
        pages = self._make_pages([
            "Introduction\nSome intro text.",
            "Regular paragraph continuing from previous.",
            "FINANCIAL HIGHLIGHTS\nKey metrics follow.",
        ])
        boundaries = _detect_headings_regex(pages)
        assert len(boundaries) >= 2


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

class TestSegmentBuilding:
    def _make_pages(self, texts: list[str]) -> list[dict]:
        return [{"page_num": i + 1, "cleaned_text": t} for i, t in enumerate(texts)]

    def test_single_segment_no_boundaries(self):
        pages = self._make_pages(["Page 1 content", "Page 2 content"])
        segments = _build_segments(pages, [], "doc-001")
        assert len(segments) == 1
        assert segments[0]["heading"] == "Document"

    def test_multiple_segments_from_boundaries(self):
        pages = self._make_pages(["Intro text", "Chapter one content", "More content"])
        boundaries = [
            {"page_num": 1, "heading": "Introduction", "type": "section"},
            {"page_num": 2, "heading": "Chapter One", "type": "chapter"},
        ]
        segments = _build_segments(pages, boundaries, "doc-001")
        assert len(segments) == 2
        assert segments[0]["heading"] == "Introduction"
        assert segments[1]["heading"] == "Chapter One"

    def test_segment_page_range(self):
        pages = self._make_pages(["Intro", "Body content", "More body"])
        boundaries = [
            {"page_num": 1, "heading": "Intro", "type": "section"},
            {"page_num": 2, "heading": "Body", "type": "section"},
        ]
        segments = _build_segments(pages, boundaries, "doc-001")
        assert segments[0]["page_range"] == [1, 1]
        assert segments[1]["page_range"] == [2, 3]

    def test_segment_token_count(self):
        text = "This is a test sentence. " * 10
        pages = self._make_pages([text])
        boundaries = [{"page_num": 1, "heading": "Test", "type": "section"}]
        segments = _build_segments(pages, boundaries, "doc-001")
        assert segments[0]["token_count"] > 0

    def test_segment_id_unique(self):
        pages = self._make_pages(["A", "B", "C"])
        boundaries = [
            {"page_num": 1, "heading": "A", "type": "section"},
            {"page_num": 2, "heading": "B", "type": "section"},
            {"page_num": 3, "heading": "C", "type": "section"},
        ]
        segments = _build_segments(pages, boundaries, "doc-001")
        ids = [s["segment_id"] for s in segments]
        assert len(ids) == len(set(ids)), "Segment IDs must be unique"


# ---------------------------------------------------------------------------
# Token tools
# ---------------------------------------------------------------------------

class TestTokenTools:
    def test_count_tokens_basic(self):
        assert count_tokens("hello world") > 0

    def test_enforce_budget_no_truncation(self):
        text = "short text"
        result = enforce_token_budget(text, 1000)
        assert result == text

    def test_enforce_budget_truncates(self):
        text = " ".join(["word"] * 10_000)
        result = enforce_token_budget(text, 100)
        assert count_tokens(result) <= 110  # slight tolerance for boundary tokens

    def test_enforce_budget_preserves_start_end(self):
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)
        result = enforce_token_budget(text, 50)
        # Start tokens should be present
        assert "word0" in result
        # End tokens should be present
        assert "word499" in result


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

class TestRuleBasedCleaning:
    def test_unicode_normalization(self):
        # U+2019 right single quotation mark → normalized
        text = "It\u2019s a test"
        cleaned = rule_based_clean(text)
        assert cleaned is not None

    def test_hyphenation_fix(self):
        text = "This is a bro-\nken word example."
        cleaned = rule_based_clean(text)
        assert "bro-\nken" not in cleaned
        assert "broken" in cleaned

    def test_multi_space_collapse(self):
        text = "too    many    spaces"
        cleaned = rule_based_clean(text)
        assert "  " not in cleaned

    def test_non_printable_removal(self):
        text = "normal\x00text\x01with\x02junk"
        cleaned = rule_based_clean(text)
        assert "\x00" not in cleaned
        assert "normaltext" in cleaned.replace("with", "").replace("junk", "")
