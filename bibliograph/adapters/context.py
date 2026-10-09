"""Utilities for assembling readable text windows from overlapping chunks."""

from __future__ import annotations

import re

_SECTION_PREFIX = re.compile(r"^\s*Section:\s*[^\n]+\s*(?:\n\s*)?", re.IGNORECASE)


def join_chunk_context(
    chunks: list[tuple[int, str]], *, center_ordinal: int, max_words: int
) -> str:
    """Join ordered, overlapping chunks without repeated text or section labels."""
    if not chunks or max_words < 1:
        return ""
    assembled: list[str] = []
    center_index = 0
    for ordinal, raw_text in sorted(chunks, key=lambda value: value[0]):
        text = _SECTION_PREFIX.sub("", raw_text).strip()
        words = text.split()
        if not words:
            continue
        if ordinal == center_ordinal:
            center_index = len(assembled)
        if assembled:
            previous = assembled[-1].split()
            overlap = _overlap_size(previous, words)
            words = words[overlap:]
        if words:
            assembled.append(" ".join(words))
    if not assembled:
        return ""

    all_words = " ".join(assembled).split()
    center_words = sum(len(part.split()) for part in assembled[:center_index])
    if len(all_words) > max_words:
        start = max(0, center_words - max_words // 2)
        start = min(start, len(all_words) - max_words)
        all_words = all_words[start : start + max_words]
    return " ".join(all_words)


def _overlap_size(previous: list[str], current: list[str]) -> int:
    """Find the longest suffix/prefix overlap, ignoring punctuation and case."""
    maximum = min(len(previous), len(current), 40)
    for size in range(maximum, 2, -1):
        left = [_normalise(word) for word in previous[-size:]]
        right = [_normalise(word) for word in current[:size]]
        if left == right:
            return size
    return 0


def _normalise(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word).casefold()
