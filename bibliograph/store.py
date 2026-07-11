import json
import math
import re
import sqlite3
from collections.abc import Iterable
from hashlib import sha256
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
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS indexed_files (
                attachment_key TEXT PRIMARY KEY, source_version INTEGER, file_hash TEXT NOT NULL
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

    def search(
        self,
        embedding: list[float],
        limit: int = 5,
        query_text: str | None = None,
        semantic_weight: float = 0.75,
    ) -> list[CitationSource]:
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1")
        scored: list[CitationSource] = []
        for row in self.connection.execute("SELECT * FROM chunks"):
            source_embedding = json.loads(row[11])
            semantic_score = _cosine(embedding, source_embedding)
            lexical_score = _lexical_overlap(query_text, row[7]) if query_text else 0.0
            score = semantic_weight * semantic_score + (1 - semantic_weight) * lexical_score
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

    def needs_file_index(self, attachment_key: str, source_version: int | None, path: str) -> bool:
        file_hash = _file_hash(path)
        row = self.connection.execute(
            "SELECT source_version, file_hash FROM indexed_files WHERE attachment_key = ?",
            (attachment_key,),
        ).fetchone()
        return row is None or row != (source_version, file_hash)

    def mark_file_indexed(self, attachment_key: str, source_version: int | None, path: str) -> None:
        self.connection.execute(
            """INSERT INTO indexed_files VALUES (?, ?, ?)
            ON CONFLICT(attachment_key) DO UPDATE SET
                source_version=excluded.source_version, file_hash=excluded.file_hash""",
            (attachment_key, source_version, _file_hash(path)),
        )
        self.connection.commit()


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


def _lexical_overlap(query: str, document: str) -> float:
    query_terms = set(re.findall(r"[\w-]+", query.lower()))
    document_terms = set(re.findall(r"[\w-]+", document.lower()))
    return len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0


def _file_hash(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
