import re
from collections.abc import Iterable

from .models import Paper, TextChunk


def split_text(text: str, max_words: int = 180, overlap_words: int = 30) -> list[str]:
    """Split text into overlapping word chunks while preserving readable boundaries."""
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("max_words must be positive and overlap_words must be smaller")
    words = re.findall(r"\S+", text)
    chunks: list[str] = []
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks


def chunk_pages(
    paper: Paper,
    pages: Iterable[tuple[int, str]],
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    index = 0
    for page, text in pages:
        for page_chunk in split_text(text, max_words, overlap_words):
            chunks.append(
                TextChunk(
                    chunk_id=f"{paper.zotero_key}:{page}:{index}",
                    paper=paper,
                    text=page_chunk,
                    page=page,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks
