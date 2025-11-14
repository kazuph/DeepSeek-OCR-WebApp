"""Utilities for simplifying Japanese OCR text for LLM consumption.

This module focuses on three main operations that often help downstream
language models:

1. Convert katakana to hiragana (easier for some models to digest).
2. Convert all kanji phrases to their hiragana readings.
3. Replace kanji outside the Japanese "教育漢字" (elementary school list)
   with their hiragana readings, keeping the easier characters intact.

All functions are deterministic and covered by unit tests so that we can
assert the behaviour on captured OCR fixtures without re-running OCR.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable, Iterator, Tuple

import jaconv
from fugashi import Tagger


_DATA_DIR = Path(__file__).parent / "data"
_KYOU_KANJI_PATH = _DATA_DIR / "kyouiku_kanji.txt"
if not _KYOU_KANJI_PATH.exists():  # pragma: no cover - configuration error
    raise FileNotFoundError(f"Missing 教育漢字リスト: {_KYOU_KANJI_PATH}")
_KYOU_KANJI: set[str] = set(_KYOU_KANJI_PATH.read_text(encoding="utf-8"))

_WHITESPACE_PATTERN = re.compile(r"(\s+)")


@lru_cache(maxsize=1)
def _get_tagger() -> Tagger:
    """Initialise and cache the fugashi Tagger instance."""

    return Tagger()


def _token_readings(text: str) -> Iterable[Tuple[str, str]]:
    """Yield (surface, reading) tuples for the provided text chunk."""

    tagger = _get_tagger()
    for token in tagger(text):
        reading = getattr(token.feature, "kana", None) or getattr(token.feature, "reading", None)
        if reading:
            normalized = jaconv.kata2hira(reading)
        else:
            normalized = jaconv.kata2hira(token.surface)
        yield token.surface, normalized


def _iter_chunks(text: str) -> Iterator[Tuple[str, str]]:
    """Split text into (kind, value) pairs to preserve whitespace."""

    if not text:
        return

    last = 0
    for match in _WHITESPACE_PATTERN.finditer(text):
        start, end = match.span()
        if start > last:
            yield ("text", text[last:start])
        yield ("ws", match.group(0))
        last = end
    if last < len(text):
        yield ("text", text[last:])


def katakana_to_hiragana(text: str) -> str:
    """Convert every katakana syllable (full/half width) to hiragana."""

    return jaconv.kata2hira(text or "")


def text_to_hiragana(text: str) -> str:
    """Convert the entire string to hiragana using dictionary readings."""

    converted: list[str] = []
    for kind, value in _iter_chunks(text):
        if kind == "ws":
            converted.append(value)
            continue
        converted.extend(reading for _, reading in _token_readings(value))
    return "".join(converted)


def limit_to_kyouiku_kanji(text: str) -> str:
    """Replace kanji beyond the 小学生 list with their hiragana readings."""

    simplified: list[str] = []
    for kind, value in _iter_chunks(text):
        if kind == "ws":
            simplified.append(value)
            continue
        for surface, reading in _token_readings(value):
            if _needs_simplification(surface):
                simplified.append(reading)
            else:
                simplified.append(surface)
    return "".join(simplified)


def _needs_simplification(surface: str) -> bool:
    has_kanji = False
    for char in surface:
        if _is_cjk(char):
            has_kanji = True
            if char not in _KYOU_KANJI:
                return True
    return False


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )
