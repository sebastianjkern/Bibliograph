import json
import math
import re
import sqlite3
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

import sqlite_vec

from .logging_utils import get_logger
from .models import CitationSource, Paper, TextChunk

logger = get_logger("store")


class SQLiteIndex:
    """Persistent metadata store backed by sqlite-vec nearest-neighbor search."""

    def __init__(
        self,
        path: str | Path = "bibliograph.db",
        dimension: int | None = None,
        embedding_id: str | None = None,
    ):
        self.path = str(path)
        self.embedding_id = embedding_id
        self.reindexed = False
        self.connection = sqlite3.connect(self.path)
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)
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
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS vector_meta (dimension INTEGER NOT NULL)"
        )
        columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(vector_meta)")
        }
        if "embedding_id" not in columns:
            self.connection.execute("ALTER TABLE vector_meta ADD COLUMN embedding_id TEXT")
        self.connection.commit()
        self._vector_dimension = self._stored_dimension()
        stored_embedding_id = self._stored_embedding_id()
        if embedding_id is not None and stored_embedding_id != embedding_id:
            if self._has_index_data():
                logger.info(
                    "Embedding model changed from %s to %s; clearing index for reindexing",
                    stored_embedding_id or "unknown",
                    embedding_id,
                )
                self._reset()
            else:
                self._set_embedding_id(embedding_id)
        if dimension is not None:
            if dimension > 0:
                if self._vector_dimension not in (None, dimension) and embedding_id is not None:
                    self._reset()
                self._ensure_vector_table(dimension)

    def upsert(self, chunks: Iterable[TextChunk], embeddings: Iterable[list[float]]) -> int:
        rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            normalized = _normalize(embedding)
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
                    json.dumps(normalized),
                )
            )
        if rows:
            self._ensure_vector_table(len(json.loads(rows[0][-1])))
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
        if rows:
            for row in rows:
                self.connection.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (row[0],))
                self.connection.execute(
                    "INSERT INTO vec_chunks(embedding, chunk_id) VALUES (?, ?)",
                    (sqlite_vec.serialize_float32(json.loads(row[-1])), row[0]),
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
        if self._vector_dimension is None:
            row = self.connection.execute("SELECT embedding FROM chunks LIMIT 1").fetchone()
            if row is None:
                return []
            self._ensure_vector_table(len(json.loads(row[0])))
            self._backfill_vectors()
        candidate_limit = max(limit * 8, 50)
        vector_rows = self.connection.execute(
            """SELECT chunk_id, distance FROM vec_chunks
            WHERE embedding MATCH ? AND k = ? ORDER BY distance""",
            (sqlite_vec.serialize_float32(_normalize(embedding)), candidate_limit),
        ).fetchall()
        if not vector_rows:
            return []
        placeholders = ",".join("?" for _ in vector_rows)
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            [row[0] for row in vector_rows],
        ).fetchall()
        by_chunk_id = {row[0]: row for row in rows}
        scored: list[CitationSource] = []
        for chunk_id, _distance in vector_rows:
            row = by_chunk_id[chunk_id]
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

    def rebuild(self) -> None:
        """Clear all indexed content so the database can be rebuilt from scratch."""
        self._reset()

    def context_for(self, chunk: TextChunk, window: int = 1, max_words: int = 600) -> str:
        """Return nearby indexed chunks from the same paper for grounded quote selection."""
        if window < 0 or max_words <= 0:
            raise ValueError("window must be non-negative and max_words must be positive")
        rows = self.connection.execute(
            """SELECT text FROM chunks
            WHERE zotero_key = ? AND chunk_index BETWEEN ? AND ?
            ORDER BY chunk_index""",
            (chunk.paper.zotero_key, chunk.chunk_index - window, chunk.chunk_index + window),
        ).fetchall()
        words = " ".join(row[0] for row in rows).split()
        return " ".join(words[:max_words])

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

    def _stored_dimension(self) -> int | None:
        row = self.connection.execute("SELECT dimension FROM vector_meta LIMIT 1").fetchone()
        return row[0] if row else None

    def _stored_embedding_id(self) -> str | None:
        row = self.connection.execute("SELECT embedding_id FROM vector_meta LIMIT 1").fetchone()
        return row[0] if row else None

    def _has_index_data(self) -> bool:
        return any(
            self.connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
            for table in ("chunks", "indexed_files", "vector_meta")
        )

    def _set_embedding_id(self, embedding_id: str) -> None:
        row = self.connection.execute("SELECT 1 FROM vector_meta LIMIT 1").fetchone()
        if row:
            self.connection.execute("UPDATE vector_meta SET embedding_id = ?", (embedding_id,))
        self.connection.commit()

    def _reset(self) -> None:
        self.connection.execute("DROP TABLE IF EXISTS vec_chunks")
        self.connection.execute("DELETE FROM chunks")
        self.connection.execute("DELETE FROM indexed_files")
        self.connection.execute("DELETE FROM vector_meta")
        self.connection.commit()
        self._vector_dimension = None
        self.reindexed = True

    def _ensure_vector_table(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self._vector_dimension is not None and self._vector_dimension != dimension:
            raise ValueError(
                f"Embedding dimension {dimension} does not match index dimension "
                f"{self._vector_dimension}"
            )
        if self._vector_dimension is None:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks "
                f"USING vec0(embedding float[{dimension}], +chunk_id text)"
            )
            self.connection.execute(
                "INSERT INTO vector_meta(dimension, embedding_id) VALUES (?, ?)",
                (dimension, self.embedding_id),
            )
            self.connection.commit()
            self._vector_dimension = dimension

    def _backfill_vectors(self) -> None:
        rows = self.connection.execute("SELECT chunk_id, embedding FROM chunks").fetchall()
        for chunk_id, embedding in rows:
            self.connection.execute(
                "INSERT INTO vec_chunks(embedding, chunk_id) VALUES (?, ?)",
                (sqlite_vec.serialize_float32(_normalize(json.loads(embedding))), chunk_id),
            )
        self.connection.commit()


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


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
