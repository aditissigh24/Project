"""
Text processing utilities: cleaning, normalization, token management.
"""

from __future__ import annotations

import re
import unicodedata

import tiktoken

from config import settings

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def enforce_token_budget(text: str, budget: int) -> str:
    """
    Truncate text to fit within `budget` tokens.
    Preserves start and end, truncates from the middle to retain context boundaries.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= budget:
        return text
    half = budget // 2
    kept = tokens[:half] + tokens[-half:]
    return enc.decode(kept)


# ---------------------------------------------------------------------------
# Rule-based cleaning
# ---------------------------------------------------------------------------

# Common OCR character substitutions in financial documents
_OCR_SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"\brn\b", "m"),          # rn → m
    (r"\bI1\b", "ll"),          # I1 → ll
    (r"\b0(?=[A-Za-z])", "O"), # 0 before letter → O
    (r"(?<=[A-Za-z])0\b", "o"),# trailing 0 after letter → o
    (r"\|", "I"),               # pipe → I (common in tables)
]

_HYPHEN_BREAK_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def fix_hyphenation(text: str) -> str:
    """Rejoin words broken across lines by hyphens."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def remove_non_printable(text: str) -> str:
    return _NON_PRINTABLE_RE.sub("", text)


def apply_ocr_substitutions(text: str) -> str:
    for pattern, replacement in _OCR_SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    return text


def rule_based_clean(text: str) -> str:
    """Full deterministic cleaning pass."""
    text = normalize_unicode(text)
    text = remove_non_printable(text)
    text = fix_hyphenation(text)
    text = apply_ocr_substitutions(text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Confidence check
# ---------------------------------------------------------------------------

def _load_common_words() -> set[str]:
    """Minimal word list heuristic — just check high-frequency English tokens."""
    common = {
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "on",
        "are", "with", "as", "at", "by", "from", "or", "an", "this", "was",
        "be", "it", "not", "have", "has", "which", "its", "their", "been",
        "were", "will", "more", "also", "than", "may", "during", "year",
        "bank", "financial", "per", "cent", "crore", "lakh", "million",
    }
    return common


_COMMON_WORDS = _load_common_words()


def text_needs_llm_repair(text: str, threshold: float = 0.15) -> bool:
    """
    Return True if more than `threshold` fraction of tokens look like OCR noise.
    A token is considered noise if it's not in our common-word set AND
    contains a high proportion of non-alphanumeric characters.
    """
    words = text.split()
    if not words:
        return False
    noise_count = 0
    for w in words:
        w_lower = w.lower().strip(".,;:!?()'\"")
        if w_lower in _COMMON_WORDS:
            continue
        if len(w_lower) > 0:
            non_alpha = sum(not c.isalnum() for c in w_lower) / len(w_lower)
            if non_alpha > 0.5:
                noise_count += 1
    return (noise_count / len(words)) > threshold
