"""Injected document indexing pipeline with visible batch progress."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from core.embedding import BatchEmbeddingStrategy

from ..domain import Chunk, Paper
from ..logging_utils import get_logger

logger = get_logger("indexing")

# Bump this explicit value whenever PDF cleanup or chunk boundaries change.
INDEXING_FINGERPRINT = "pdf-body-v1:sentence-chunks-v1:max-words=120:overlap-words=20:text"

ExtractPages = Callable[[str | Path], list[tuple[int, str, str | None]]]
ChunkDocument = Callable[[Paper, Iterable[tuple[int, str, str | None]]], list[Chunk]]
EmbedDocuments = BatchEmbeddingStrategy


def index_document(
    paper: Paper,
    source_key: str,
    version: str | int | None,
    path: str | Path,
    *,
    extract_pages: ExtractPages,
    chunk_document: ChunkDocument,
    embed_documents: EmbedDocuments,
    store,
    batch_size: int = 32,
    supported_kinds: Sequence[str] = ("text",),
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chunks = [
        replace(chunk, chunk_id=f"{source_key}:{chunk.ordinal}")
        for chunk in chunk_document(paper, extract_pages(path))
    ]
    unsupported = {chunk.content_kind for chunk in chunks} - set(supported_kinds)
    if unsupported:
        kinds = ", ".join(sorted(unsupported))
        raise ValueError(f"Embedding provider does not support content kinds: {kinds}")
    if not chunks:
        logger.warning("No text chunks extracted from %s", paper.title)
        # A changed document that no longer yields text must replace its prior
        # chunks rather than leaving stale evidence searchable.
        store.replace_document(paper, source_key, version, str(path), [])
        return {"source_key": source_key, "chunks": 0, "path": str(path)}

    batches: list[tuple[list[Chunk], list[list[float]]]] = []
    total = len(chunks)
    logger.info("Embedding %d chunks for %s in batches of %d", total, paper.title, batch_size)
    for start in range(0, total, batch_size):
        chunk_batch = chunks[start : start + batch_size]
        end = start + len(chunk_batch)
        logger.info("Embedding chunks %d-%d of %d for %s", start + 1, end, total, paper.title)
        vectors = embed_documents([chunk.text for chunk in chunk_batch])
        if len(vectors) != len(chunk_batch):
            raise ValueError("Embedding provider returned a different number of vectors")
        batches.append((chunk_batch, vectors))
        logger.info("Prepared %d/%d chunks for %s", end, total, paper.title)

    stored = store.replace_document(paper, source_key, version, str(path), batches)
    logger.info("Stored %d/%d chunks for %s", stored, total, paper.title)
    return {"source_key": source_key, "chunks": stored, "path": str(path)}
