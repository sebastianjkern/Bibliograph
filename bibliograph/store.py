import json
import math
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import CitationSource, Paper, TextChunk


class SQLiteIndex:
    """Small persistent vector index with complete source metadata."""

    def __init__(self, path: str | Path = "bibliograph.db"):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, zotero_key TEXT NOT NULL, title TEXT NOT NULL,
                authors TEXT NOT NULL, year TEXT, doi TEXT, collections TEXT NOT NULL,
                text TEXT NOT NULL, page INTEGER, section TEXT, chunk_index INTEGER NOT NULL,
                embedding TEXT NOT NULL
            )"""
        )
        self.connection.commit()

    def upsert(self, chunks: Iterable[TextChunk], embeddings: Iterable[list[float]]) -> int:
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.paper.zotero_key,
                    chunk.paper.title,
                    json.dumps(chunk.paper.authors),
                    chunk.paper.year,
                    chunk.paper.doi,
                    json.dumps(chunk.paper.collections),
                    chunk.text,
                    chunk.page,
                    chunk.section,
                    chunk.chunk_index,
                    json.dumps(embedding),
                )
            )
        self.connection.executemany(
            """INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                title=excluded.title, authors=excluded.authors, year=excluded.year,
                doi=excluded.doi, collections=excluded.collections, text=excluded.text,
                page=excluded.page, section=excluded.section, chunk_index=excluded.chunk_index,
                embedding=excluded.embedding""",
            rows,
        )
        self.connection.commit()
        return len(rows)

    def search(self, embedding: list[float], limit: int = 5) -> list[CitationSource]:
        scored: list[CitationSource] = []
        for row in self.connection.execute("SELECT * FROM chunks"):
            source_embedding = json.loads(row[11])
            score = _cosine(embedding, source_embedding)
            paper = Paper(
                row[1],
                row[2],
                tuple(json.loads(row[3])),
                row[4],
                row[5],
                tuple(json.loads(row[6])),
            )
            chunk = TextChunk(row[0], paper, row[7], row[8], row[9], row[10])
            scored.append(CitationSource(chunk, score))
        return sorted(scored, key=lambda source: source.score, reverse=True)[:limit]

    def close(self) -> None:
        self.connection.close()


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return (
        sum(a * b for a, b in zip(left, right, strict=True)) / denominator
        if denominator
        else 0.0
    )
