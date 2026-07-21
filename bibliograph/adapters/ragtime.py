"""Ragtime-backed scientific-paper indexing and retrieval.

Bibliograph owns acquisition and citation presentation.  Ragtime owns the RAG
concerns: graph construction, embeddings, hybrid retrieval, graph expansion,
and ranking.  The small tables prefixed with ``bibliograph_`` contain only
sync bookkeeping and are deliberately outside the retrieval model.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ragtime import (
    IdentityQueryStrategy,
    PersonalizedPageRankExpansion,
    QueryPlan,
    Ragtime,
    SourceDocument,
    SQLiteGraphVectorStore,
    WeightedFusionRanking,
    scientific_paper_graph_profile,
)

from ..domain import Chunk, Paper, RetrievalScore, ScoredChunk

RAGTIME_SCHEMA_VERSION = 3
RAGTIME_FINGERPRINT = "ragtime:scientific-paper@1"
VECTOR_SPACE = "bibliograph-scientific-papers"
_PAGE_MARKER = re.compile(r"\s*\[Bibliograph page (?P<page>\d+)\]\s*$")


class RagtimeIndexError(RuntimeError):
    """Raised when an index cannot be opened safely by the Ragtime backend."""


class _EmbeddingAdapter:
    def __init__(self, model_name: str, embed: Callable[[Sequence[str]], list[list[float]]]):
        self.model_name = model_name
        self._embed = embed

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed, list(texts))


@dataclass(slots=True)
class _PlannedQueries:
    original: str
    alternatives: tuple[str, ...]

    async def plan(self, _query: str) -> QueryPlan:
        return QueryPlan(self.original, self.alternatives)


class RagtimeBackend:
    """Adapt Bibliograph providers and result shapes to Ragtime's public facade."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedding: dict[str, Any],
        mode: str = "read",
    ) -> None:
        if mode not in {"read", "write"}:
            raise ValueError("mode must be 'read' or 'write'")
        self.path = Path(path)
        self.mode = mode
        _assert_openable(self.path, create=mode == "write")
        self.store = SQLiteGraphVectorStore(self.path)
        try:
            self._document_embeddings = _EmbeddingAdapter(
                str(embedding["id"]), embedding["embed_documents"]
            )
            self._query_embeddings = _EmbeddingAdapter(
                str(embedding["id"]), embedding["embed_queries"]
            )
            self.embedding_id = str(embedding["id"])
            if mode == "write":
                self._initialize_bookkeeping()
            self._assert_compatible()
            self._ingestion = self._compose(
                self._document_embeddings,
                IdentityQueryStrategy(),
                owns_store=True,
            )
        except BaseException:
            self.store.close()
            raise

    def _compose(self, embeddings, query_strategy, *, owns_store: bool) -> Ragtime:
        return Ragtime(
            self.store,
            embeddings,
            query_strategy,
            graph_profiles=(scientific_paper_graph_profile(),),
            expansion=PersonalizedPageRankExpansion(),
            ranking=WeightedFusionRanking(),
            vector_space=VECTOR_SPACE,
            owns_store=owns_store,
        )

    def __enter__(self) -> RagtimeBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._ingestion.close()

    def index_pdf(
        self,
        paper: Paper,
        source_key: str,
        version: str | int | None,
        path: str | Path,
        *,
        extract_pages: Callable[[str | Path], list[tuple[int, str, str | None]]],
    ) -> dict[str, Any]:
        if self.mode != "write":
            raise RagtimeIndexError("a read-only backend cannot index documents")
        pages = extract_pages(path)
        content = _scientific_document_text(paper, pages)
        document = SourceDocument(
            content=content,
            uri=Path(path).resolve().as_uri(),
            modality="scientific-paper",
            document_id=source_key,
            repository_id="zotero",
            revision=None if version is None else str(version),
            metadata={
                "zotero_key": paper.zotero_key,
                "title": paper.title,
                "authors": list(paper.authors),
                "year": paper.year,
                "doi": paper.doi,
                "collections": list(paper.collections),
                "path": str(Path(path)),
            },
        )
        report = asyncio.run(self._ingestion.ingest(document))
        chunk_count = int(
            self.store.connection.execute(
                "SELECT COUNT(*) FROM nodes WHERE document_id = ? AND kind IN ('section', 'chunk')",
                (source_key,),
            ).fetchone()[0]
        )
        stat = Path(path).stat()
        with self.store.connection:
            self.store.connection.execute(
                """INSERT INTO bibliograph_sources(
                       source_key, document_id, version, path, mtime_ns, size, state, detail
                   ) VALUES (?, ?, ?, ?, ?, ?, 'indexed', NULL)
                   ON CONFLICT(source_key) DO UPDATE SET
                     document_id=excluded.document_id, version=excluded.version,
                     path=excluded.path, mtime_ns=excluded.mtime_ns, size=excluded.size,
                     state='indexed', detail=NULL, updated_at=CURRENT_TIMESTAMP""",
                (
                    source_key,
                    report.document_id,
                    None if version is None else str(version),
                    str(Path(path).resolve()),
                    stat.st_mtime_ns,
                    stat.st_size,
                ),
            )
        return {"source_key": source_key, "chunks": chunk_count, "path": str(path)}

    def retrieve(
        self,
        query: str,
        *,
        alternatives: Sequence[str] = (),
        limit: int = 10,
    ) -> tuple[list[ScoredChunk], dict[str, dict[str, float]]]:
        planner = _PlannedQueries(query, tuple(alternatives))
        retrieval = self._compose(self._query_embeddings, planner, owns_store=False)
        candidates = asyncio.run(
            retrieval.retrieve(
                query,
                limit=max(12, limit * 4),
                seed_limit=max(16, limit * 4),
                expansion_limit=max(96, limit * 8),
                modalities=("scientific-paper",),
            )
        )
        papers: dict[str, Paper] = {}
        hits: list[ScoredChunk] = []
        details: dict[str, dict[str, float]] = {}
        for result in candidates:
            node = result.node
            if node.kind not in {"section", "chunk"} or not node.document_id:
                continue
            paper = papers.get(node.document_id)
            if paper is None:
                paper = self._paper_for(node.document_id)
                papers[node.document_id] = paper
            title = str(node.metadata.get("title") or "")
            marker = _PAGE_MARKER.search(title)
            page = int(marker.group("page")) if marker else None
            section = _PAGE_MARKER.sub("", title).strip() or None
            ordinal = int(node.metadata.get("section_index", node.metadata.get("offset", 0)))
            score = RetrievalScore(
                float(result.score),
                semantic=float(result.vector_score),
                lexical=float(result.lexical_score),
            )
            chunk = Chunk(
                node.node_id,
                paper,
                node.content,
                page=page,
                section=section,
                ordinal=ordinal,
            )
            hits.append((chunk, score))
            details[chunk.chunk_id] = {
                "vector": float(result.vector_score),
                "lexical": float(result.lexical_score),
                "graph": float(result.graph_score),
                "retrieval": float(result.score),
            }
            if len(hits) >= limit:
                break
        return hits, details

    def needs_document(
        self,
        source_key: str,
        version: str | int | None,
        path: str | Path,
    ) -> bool:
        row = self.store.connection.execute(
            """SELECT version, path, mtime_ns, size, state
               FROM bibliograph_sources WHERE source_key = ?""",
            (source_key,),
        ).fetchone()
        if row is None or row["state"] != "indexed":
            return True
        file = Path(path)
        try:
            stat = file.stat()
        except OSError:
            return True
        expected_version = None if version is None else str(version)
        return not (
            row["version"] == expected_version
            and Path(str(row["path"])) == file.resolve()
            and int(row["mtime_ns"]) == stat.st_mtime_ns
            and int(row["size"]) == stat.st_size
        )

    def record_document_failure(
        self,
        paper: Paper,
        source_key: str,
        version: str | int | None,
        *,
        state: str,
        detail: str,
        path: str | Path | None = None,
    ) -> None:
        resolved = str(Path(path).resolve()) if path is not None else None
        with self.store.connection:
            self.store.connection.execute(
                """INSERT INTO bibliograph_sources(
                       source_key, version, path, state, detail
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(source_key) DO UPDATE SET
                     version=excluded.version, path=excluded.path, state=excluded.state,
                     detail=excluded.detail, updated_at=CURRENT_TIMESTAMP""",
                (source_key, None if version is None else str(version), resolved, state, detail),
            )

    def mark_sync_complete(self) -> None:
        with self.store.connection:
            self.store.connection.execute(
                """UPDATE bibliograph_index_meta
                   SET last_successful_sync=CURRENT_TIMESTAMP WHERE singleton=1"""
            )

    def stats(self) -> dict[str, Any]:
        return _stats(self.store.connection, self.path)

    def _paper_for(self, document_id: str) -> Paper:
        row = self.store.connection.execute(
            "SELECT metadata FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        if row is None:
            raise RagtimeIndexError(f"retrieved node has no source document: {document_id}")
        metadata = json.loads(str(row["metadata"]))
        return Paper(
            str(metadata.get("zotero_key") or document_id),
            str(metadata.get("title") or document_id),
            tuple(str(value) for value in metadata.get("authors", ())),
            None if metadata.get("year") is None else str(metadata["year"]),
            None if metadata.get("doi") is None else str(metadata["doi"]),
            tuple(str(value) for value in metadata.get("collections", ())),
        )

    def _initialize_bookkeeping(self) -> None:
        self.store.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS bibliograph_index_meta (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                schema_version INTEGER NOT NULL,
                backend TEXT NOT NULL,
                embedding_id TEXT NOT NULL,
                extraction_chunking_fingerprint TEXT NOT NULL,
                last_successful_sync TEXT
            );
            CREATE TABLE IF NOT EXISTS bibliograph_sources (
                source_key TEXT PRIMARY KEY,
                document_id TEXT,
                version TEXT,
                path TEXT,
                mtime_ns INTEGER,
                size INTEGER,
                state TEXT NOT NULL,
                detail TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.store.connection.execute(
            """INSERT OR IGNORE INTO bibliograph_index_meta(
                   singleton, schema_version, backend, embedding_id,
                   extraction_chunking_fingerprint
               ) VALUES (1, ?, 'ragtime', ?, ?)""",
            (RAGTIME_SCHEMA_VERSION, self.embedding_id, RAGTIME_FINGERPRINT),
        )
        self.store.connection.commit()

    def _assert_compatible(self) -> None:
        row = self.store.connection.execute(
            """SELECT backend, embedding_id FROM bibliograph_index_meta WHERE singleton=1"""
        ).fetchone()
        if row is None or row["backend"] != "ragtime":
            raise RagtimeIndexError(
                "This is not a Ragtime Bibliograph index. Run `bibliograph sync --rebuild`."
            )
        if str(row["embedding_id"]) != self.embedding_id:
            raise RagtimeIndexError(
                f"This index uses embedding '{row['embedding_id']}', but '{self.embedding_id}' "
                "is configured. Run `bibliograph sync --rebuild`."
            )


def is_ragtime_index(path: str | Path) -> bool:
    database = Path(path)
    if not database.is_file():
        return False
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database)
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='bibliograph_index_meta'"
        ).fetchone()
        if row is None:
            return False
        value = connection.execute(
            "SELECT backend FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        return value is not None and value[0] == "ragtime"
    except sqlite3.DatabaseError:
        return False
    finally:
        if connection is not None:
            connection.close()


def inspect_ragtime_index(path: str | Path) -> dict[str, Any]:
    database = Path(path)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return _stats(connection, database)
    finally:
        connection.close()


@contextmanager
def open_ragtime_staging(path: str | Path, *, embedding: dict[str, Any]):
    """Build a replacement index beside the live database and swap it after validation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.ragtime-staging-{uuid4().hex}")
    backend: RagtimeBackend | None = None
    try:
        backend = RagtimeBackend(staging, embedding=embedding, mode="write")
        yield backend
        summary = backend.stats()
        backend.close()
        backend = None
        if not all(summary[name] > 0 for name in ("documents", "chunks", "vectors")):
            raise RagtimeIndexError(
                "Refusing to replace the live index: the staged Ragtime index is incomplete."
            )
        os.replace(staging, target)
    finally:
        if backend is not None:
            backend.close()
        for candidate in (staging, Path(f"{staging}-wal"), Path(f"{staging}-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass


def _assert_openable(path: Path, *, create: bool) -> None:
    if not path.exists():
        if create:
            return
        raise RagtimeIndexError(
            f"Ragtime index not found at {path}. Run `bibliograph sync --rebuild`."
        )
    if not is_ragtime_index(path):
        raise RagtimeIndexError(
            "The existing database uses Bibliograph's legacy RAG schema. "
            "Run `bibliograph sync --rebuild` to migrate to Ragtime."
        )


def _scientific_document_text(
    paper: Paper, pages: Sequence[tuple[int, str, str | None]]
) -> str:
    parts = [f"# {paper.title}"]
    for page, text, section in pages:
        heading = section or "Body"
        parts.extend((f"## {heading} [Bibliograph page {page}]", text.strip()))
    return "\n\n".join(part for part in parts if part)


def _stats(connection: sqlite3.Connection, path: Path) -> dict[str, Any]:
    meta = connection.execute(
        "SELECT * FROM bibliograph_index_meta WHERE singleton=1"
    ).fetchone()
    vector = connection.execute(
        "SELECT model, dimensions FROM vector_spaces WHERE name = ?", (VECTOR_SPACE,)
    ).fetchone()
    def count(sql: str) -> int:
        return int(connection.execute(sql).fetchone()[0])

    return {
        "path": str(path),
        "schema_version": int(meta["schema_version"]),
        "papers": count("SELECT COUNT(*) FROM documents WHERE modality='scientific-paper'"),
        "documents": count("SELECT COUNT(*) FROM documents"),
        "chunks": count("SELECT COUNT(*) FROM nodes WHERE kind IN ('section', 'chunk')"),
        "vectors": count("SELECT COUNT(*) FROM node_vectors"),
        "embedding_id": str(meta["embedding_id"]),
        "dimension": int(vector["dimensions"]) if vector is not None else None,
        "extraction_chunking_fingerprint": str(meta["extraction_chunking_fingerprint"]),
        "last_successful_sync": meta["last_successful_sync"],
        "pending_failures": count(
            "SELECT COUNT(*) FROM bibliograph_sources WHERE state != 'indexed'"
        ),
        "rebuild_required": False,
    }
