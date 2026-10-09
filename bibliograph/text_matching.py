"""Text matching tolerant of whitespace and PDF line-wrap hyphenation."""

from __future__ import annotations


def find_text_span(text: str, quote: str) -> tuple[int, int] | None:
    """Find a quote in source text and return its original character span."""
    normalized_text, offsets = _normalize_with_offsets(text)
    normalized_quote, _ = _normalize_with_offsets(quote)
    if not normalized_quote:
        return None
    start = normalized_text.find(normalized_quote)
    if start < 0:
        return None
    end = start + len(normalized_quote) - 1
    return offsets[start][0], offsets[end][1]


def contains_text(text: str, quote: str) -> bool:
    return find_text_span(text, quote) is not None


def _normalize_with_offsets(value: str) -> tuple[str, list[tuple[int, int]]]:
    normalized: list[str] = []
    offsets: list[tuple[int, int]] = []
    index = 0
    while index < len(value):
        character = value[index]
        if (
            character == "-"
            and normalized
            and normalized[-1].isalnum()
            and index + 1 < len(value)
            and value[index + 1].isspace()
        ):
            next_word = index + 1
            while next_word < len(value) and value[next_word].isspace():
                next_word += 1
            if next_word < len(value) and value[next_word].isalnum():
                index = next_word
                continue
        if character.isspace():
            next_index = index + 1
            while next_index < len(value) and value[next_index].isspace():
                next_index += 1
            if normalized and normalized[-1] != " ":
                normalized.append(" ")
                offsets.append((index, next_index))
            index = next_index
            continue
        folded = character.casefold()
        for folded_character in folded:
            normalized.append(folded_character)
            offsets.append((index, index + 1))
        index += 1
    while normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return "".join(normalized).strip(), offsets
