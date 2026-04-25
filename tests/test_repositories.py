"""
Tests for MongoDB repository layer.
Uses mongomock or skips if a live MongoDB isn't available.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc_id() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# documents repository
# ---------------------------------------------------------------------------

class TestDocumentRepository:
    """Unit tests using mocked pymongo collection."""

    def _mock_collection(self):
        col = MagicMock()
        col.update_one = MagicMock()
        col.find_one = MagicMock(return_value=None)
        col.count_documents = MagicMock(return_value=0)
        return col

    def test_upsert_document_calls_update_one(self):
        from db import repositories as repo

        col = self._mock_collection()
        with patch("db.repositories.get_collection", return_value=col):
            repo.upsert_document("doc-1", "/path/to/file.pdf", page_count=50)
            col.update_one.assert_called_once()
            call_args = col.update_one.call_args
            assert call_args[0][0] == {"doc_id": "doc-1"}  # filter

    def test_set_document_status(self):
        from db import repositories as repo

        col = self._mock_collection()
        with patch("db.repositories.get_collection", return_value=col):
            repo.set_document_status("doc-1", "READY")
            col.update_one.assert_called_once()
            _, kwargs = col.update_one.call_args
            # Check upsert not set (status update shouldn't upsert)
            assert kwargs.get("upsert", False) is False

    def test_get_document_returns_none_when_missing(self):
        from db import repositories as repo

        col = self._mock_collection()
        col.find_one.return_value = None
        with patch("db.repositories.get_collection", return_value=col):
            result = repo.get_document("nonexistent")
            assert result is None


# ---------------------------------------------------------------------------
# pages repository
# ---------------------------------------------------------------------------

class TestPageRepository:
    def test_upsert_page(self):
        from db import repositories as repo

        col = MagicMock()
        with patch("db.repositories.get_collection", return_value=col):
            repo.upsert_page("doc-1", 1, "raw text here", ocr_used=False, confidence=1.0)
            col.update_one.assert_called_once()

    def test_update_page_cleaned_text(self):
        from db import repositories as repo

        col = MagicMock()
        with patch("db.repositories.get_collection", return_value=col):
            repo.update_page_cleaned_text("doc-1", 1, "cleaned text")
            col.update_one.assert_called_once()

    def test_get_pages_returns_list(self):
        from db import repositories as repo

        col = MagicMock()
        col.find.return_value.sort.return_value = [
            {"doc_id": "doc-1", "page_num": 1, "raw_text": "text", "cleaned_text": "text", "ocr_used": False, "confidence": 1.0},
        ]
        with patch("db.repositories.get_collection", return_value=col):
            pages = repo.get_pages("doc-1")
            assert isinstance(pages, list)
            assert pages[0]["page_num"] == 1


# ---------------------------------------------------------------------------
# segment_analyses repository
# ---------------------------------------------------------------------------

class TestSegmentAnalysisRepository:
    def test_upsert_segment_analysis(self):
        from db import repositories as repo

        col = MagicMock()
        with patch("db.repositories.get_collection", return_value=col):
            repo.upsert_segment_analysis("doc-1", "seg-1", {"summary": "test", "key_claims": []})
            col.update_one.assert_called_once()

    def test_get_segment_analysis_not_found(self):
        from db import repositories as repo

        col = MagicMock()
        col.find_one.return_value = None
        with patch("db.repositories.get_collection", return_value=col):
            result = repo.get_segment_analysis("doc-1", "seg-missing")
            assert result is None


# ---------------------------------------------------------------------------
# contradictions repository
# ---------------------------------------------------------------------------

class TestContradictionsRepository:
    def test_insert_contradictions_empty(self):
        from db import repositories as repo

        col = MagicMock()
        with patch("db.repositories.get_collection", return_value=col):
            count = repo.insert_contradictions("doc-1", [])
            assert count == 0
            col.insert_many.assert_not_called()

    def test_insert_contradictions_adds_doc_id(self):
        from db import repositories as repo

        col = MagicMock()
        contradictions = [
            {"claim_a": "A", "claim_b": "B", "section_a": "s1", "section_b": "s2",
             "explanation": "mismatch", "severity": "high"}
        ]
        with patch("db.repositories.get_collection", return_value=col):
            count = repo.insert_contradictions("doc-1", contradictions)
            assert count == 1
            # doc_id should be injected
            assert contradictions[0]["doc_id"] == "doc-1"

    def test_get_contradictions_empty(self):
        from db import repositories as repo

        col = MagicMock()
        col.find.return_value = []
        with patch("db.repositories.get_collection", return_value=col):
            result = repo.get_contradictions("doc-1")
            assert result == []
