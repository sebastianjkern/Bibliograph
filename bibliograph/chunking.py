import re
from collections.abc import Iterable

from .models import Paper, TextChunk


def split_text(text: str, max_words: int = 120, overlap_words: int = 20) -> list[str]:
    """Split text into smaller sentence/paragraph-aware overlapping chunks."""
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("max_words must be positive and overlap_words must be smaller")
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def emit() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(" ".join(current).strip())
            overlap = re.findall(r"\S+", " ".join(current))[-overlap_words:]
            current = [" ".join(overlap)] if overlap else []
            current_words = len(overlap)

    for sentence in sentences:
        sentence_words = re.findall(r"\S+", sentence)
        if len(sentence_words) > max_words:
            emit()
            for start in range(0, len(sentence_words), max_words - overlap_words):
                part = sentence_words[start : start + max_words]
                if part:
                    chunks.append(" ".join(part))
                if start + max_words >= len(sentence_words):
                    break
            current = []
            current_words = 0
        elif current and current_words + len(sentence_words) > max_words:
            emit()
            current.extend(sentence_words)
            current_words += len(sentence_words)
        else:
            current.extend(sentence_words)
            current_words += len(sentence_words)
    emit()
    return chunks


def chunk_pages(
    paper: Paper,
    pages: Iterable[tuple[int, str] | tuple[int, str, str | None]],
    max_words: int = 120,
    overlap_words: int = 20,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    index = 0
    for page_data in pages:
        page, text, *section_data = page_data
        section = section_data[0] if section_data else None
        for page_chunk in split_text(text, max_words, overlap_words):
            context = f"Section: {section}\n\n" if section else ""
            chunks.append(
                TextChunk(
                    chunk_id=f"{paper.zotero_key}:{page}:{index}",
                    paper=paper,
                    text=f"{context}{page_chunk}",
                    page=page,
                    section=section,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks
