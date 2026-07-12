"""Schema-v2 SQLite persistence for Bibliograph.

The store deliberately has a narrow API: commands compose it with extractors and
embedding functions, while this module owns the database shape and transactional
replacement of one source document.  Vector values live only in sqlite-vec's
virtual table; relational tables retain metadata and text needed for rendering.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import sqlite_vec

from ..domain import Chunk, Paper, ScoredChunk

if TYPE_CHECKING:
    from collections.abc import Generator


SCHEMA_VERSION = 2


class IndexCompatibilityError(RuntimeError):
    """Raised when an index cannot safely serve the configured embedding space."""


class SQLiteStore:
    """A schema-v2, sqlite-vec-backed Bibliograph index.

    ``mode="read"`` always opens the database read-only and never creates or
    migrates it.  ``mode="write"`` initializes only an empty database; it never
    mutates a legacy or unknown schema.  Callers must explicitly rebuild such a
    database into a staging file instead. ``indexing_fingerprint`` identifies
    the extraction/chunking implementation selected by the composition root.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "read",
        embedding_id: str | None = None,
        indexing_fingerprint: str | None = None,
    ) -> None:
        if mode not in {"read", "write"}:
            raise ValueError("mode must be 'read' or 'write'")

        self.path = Path(path)
        self.mode = mode
        self.embedding_id = embedding_id
        self.indexing_fingerprint = indexing_fingerprint
        self._closed = False

        if mode == "read":
            if not self.path.is_file():
                raise FileNotFoundError(f"Database does not exist: {self.path}")
            self.connection = sqlite3.connect(_readonly_uri(self.path), uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)

        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)

        self._schema_version = self._detect_schema_version()
        if self.mode == "write":
            if self._schema_version == 0 and not self._has_user_tables():
                self._initialize_schema()
                self._schema_version = SCHEMA_VERSION
            elif self._schema_version != SCHEMA_VERSION:
                self.close()
                raise IndexCompatibilityError(_rebuild_message(self.path, self._schema_version))

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def stats(self) -> dict[str, Any]:
        """Return status without changing the database, including legacy status."""
        self._require_open()
        if self._schema_version == SCHEMA_VERSION:
            metadata = self._metadata()
            return {
                "path": str(self.path),
                "mode": self.mode,
                "schema_version": SCHEMA_VERSION,
                "papers": self._count("papers"),
                "documents": self._count("documents"),
                "chunks": self._count("chunks"),
                "vectors": self._count_if_exists("vec_chunks"),
                "embedding_id": metadata["embedding_id"],
                "dimension": metadata["dimension"],
                "extraction_chunking_fingerprint": metadata[
                    "extraction_chunking_fingerprint"
                ],
                "last_successful_sync": metadata["last_successful_sync"],
                "pending_failures": self._pending_failure_count(),
                "rebuild_required": False,
            }

        if self._schema_version == 1:
            return {
                "path": str(self.path),
                "mode": self.mode,
                "schema_version": 1,
                "papers": self._legacy_paper_count(),
                "documents": self._count_if_exists("indexed_files"),
                "chunks": self._count_if_exists("chunks"),
                "vectors": self._count_if_exists("vec_chunks"),
                "embedding_id": self._legacy_embedding_id(),
                "dimension": self._legacy_dimension(),
                "extraction_chunking_fingerprint": None,
                "last_successful_sync": None,
                "pending_failures": 0,
                "rebuild_required": True,
            }

        return {
            "path": str(self.path),
            "mode": self.mode,
            "schema_version": self._schema_version,
            "papers": 0,
            "documents": 0,
            "chunks": 0,
            "vectors": 0,
            "embedding_id": None,
            "dimension": None,
            "extraction_chunking_fingerprint": None,
            "last_successful_sync": None,
            "pending_failures": 0,
            "rebuild_required": True,
        }

    def assert_compatible(
        self,
        embedding_id: str | None,
        indexing_fingerprint: str | None = None,
    ) -> None:
        """Reject legacy, unknown, or incompatible index fingerprints."""
        self._require_open()
        if self._schema_version != SCHEMA_VERSION:
            raise IndexCompatibilityError(_rebuild_message(self.path, self._schema_version))

        metadata = self._metadata()
        stored_id = metadata["embedding_id"]
        if self._count_if_exists("vec_chunks") and stored_id is None:
            raise IndexCompatibilityError(
                "This database has vectors without an embedding fingerprint. Run "
                "`bibliograph sync --rebuild` to create a compatible index."
            )
        if embedding_id is not None and stored_id is not None and stored_id != embedding_id:
            raise IndexCompatibilityError(
                f"This database was indexed with embedding '{stored_id}', but the configured "
                f"embedding is '{embedding_id}'. Run `bibliograph sync --rebuild` to create "
                "a compatible index."
            )
        if indexing_fingerprint is None:
            return
        stored_fingerprint = metadata["extraction_chunking_fingerprint"]
        if stored_fingerprint is None and self._count("documents"):
            raise IndexCompatibilityError(
                "This database has indexed documents without an extraction/chunking fingerprint. "
                "Run `bibliograph sync --rebuild` to create a compatible index."
            )
        if stored_fingerprint is not None and stored_fingerprint != indexing_fingerprint:
            raise IndexCompatibilityError(
                "This database was indexed with a different extraction/chunking fingerprint. "
                "Run `bibliograph sync --rebuild` to apply the current preprocessing."
            )

    def needs_document(
        self,
        source_key: str,
        version: str | int | None,
        path: str | Path,
    ) -> bool:
        """Return whether the document source/version/file has changed.

        A missing local file is always considered pending.  This prevents a
        transiently missing attachment from being treated as successfully indexed.
        """
        self._require_v2()
        file_hash = _file_hash(path)
        if file_hash is None:
            return True
        row = self.connection.execute(
            """SELECT source_version, file_path, file_hash, indexing_state FROM documents
            WHERE source_key = ?""",
            (source_key,),
        ).fetchone()
        return row is None or row["indexing_state"] not in {"indexed", "empty"} or (
            row["source_version"], row["file_path"], row["file_hash"]
        ) != (_version_value(version), str(path), file_hash)

    def replace_document(
        self,
        paper: Paper,
        source_key: str,
        version: str | int | None,
        path: str | Path,
        batches: Iterable[tuple[list[Chunk], list[list[float]]]],
    ) -> int:
        """Atomically replace one source document's chunks and vectors.

        All batches are validated before any persistent rows are removed.  A
        failed embedder batch therefore leaves the old document fully usable.
        """
        self._require_v2(writable=True)
        prepared, dimension = self._prepare_batches(batches)
        self._assert_write_compatible()
        path_text = str(path)
        file_hash = _file_hash(path)

        with self.connection:
            metadata = self._metadata()
            stored_dimension = metadata["dimension"]
            if dimension is not None:
                if stored_dimension is not None and stored_dimension != dimension:
                    raise IndexCompatibilityError(
                        f"This database uses {stored_dimension}-dimensional embeddings, but "
                        f"the replacement produced {dimension}-dimensional embeddings. Run "
                        "`bibliograph sync --rebuild`."
                    )
                if stored_dimension is None:
                    self._create_vector_table(dimension)
                    self.connection.execute(
                        "UPDATE index_meta SET dimension = ?, updated_at = CURRENT_TIMESTAMP "
                        "WHERE singleton = 1",
                        (dimension,),
                    )

            stored_id = metadata["embedding_id"]
            if stored_id is None and self.embedding_id is not None:
                if self._count_if_exists("vec_chunks"):
                    raise IndexCompatibilityError(
                        "The database has vectors without an embedding fingerprint. Run "
                        "`bibliograph sync --rebuild` before adding data."
                    )
                self.connection.execute(
                    "UPDATE index_meta SET embedding_id = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE singleton = 1",
                    (self.embedding_id,),
                )

            stored_fingerprint = metadata["extraction_chunking_fingerprint"]
            if stored_fingerprint is None and self.indexing_fingerprint is not None:
                if self._count("documents"):
                    raise IndexCompatibilityError(
                        "The database has documents without an extraction/chunking fingerprint. "
                        "Run `bibliograph sync --rebuild` before adding data."
                    )
                self.connection.execute(
                    "UPDATE index_meta SET extraction_chunking_fingerprint = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE singleton = 1",
                    (self.indexing_fingerprint,),
                )

            self._upsert_paper(paper)
            previous = self.connection.execute(
                "SELECT document_id, paper_key FROM documents WHERE source_key = ?", (source_key,)
            ).fetchone()
            if previous is None:
                cursor = self.connection.execute(
                    """INSERT INTO documents
                    (source_key, paper_key, source_version, file_path, file_hash,
                     indexing_state, failure_message, last_attempt_at, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                    (
                        source_key,
                        paper.zotero_key,
                        _version_value(version),
                        path_text,
                        file_hash,
                        "indexed" if prepared else "empty",
                    ),
                )
                document_id = int(cursor.lastrowid)
            else:
                document_id = int(previous["document_id"])
                self._delete_document_vectors(document_id)
                self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                self.connection.execute(
                    """UPDATE documents SET
                    paper_key = ?, source_version = ?, file_path = ?, file_hash = ?,
                    indexing_state = ?, failure_message = NULL,
                    last_attempt_at = CURRENT_TIMESTAMP, indexed_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?""",
                    (
                        paper.zotero_key,
                        _version_value(version),
                        path_text,
                        file_hash,
                        "indexed" if prepared else "empty",
                        document_id,
                    ),
                )

            inserted = 0
            for chunks, embeddings in prepared:
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    self.connection.execute(
                        """INSERT INTO chunks
                        (chunk_id, document_id, ordinal, page, section, content_kind, text)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk.chunk_id,
                            document_id,
                            chunk.ordinal,
                            chunk.page,
                            chunk.section,
                            chunk.content_kind,
                            chunk.text,
                        ),
                    )
                    self.connection.execute(
                        "INSERT INTO vec_chunks(embedding, chunk_id) VALUES (?, ?)",
                        (sqlite_vec.serialize_float32(embedding), chunk.chunk_id),
                    )
                    inserted += 1

            self.connection.execute(
                "DELETE FROM papers WHERE paper_key NOT IN "
                "(SELECT DISTINCT paper_key FROM documents)"
            )
        return inserted

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
        """Persist a retryable acquisition or indexing failure for ``status``.

        Existing chunks are intentionally retained: a temporary provider or
        source outage must not silently remove previously searchable evidence.
        A later successful replacement clears the state and message atomically.
        """
        self._require_v2(writable=True)
        if state not in {"missing", "failed"}:
            raise ValueError("Failure state must be 'missing' or 'failed'")
        self._assert_write_compatible()
        path_text = str(path) if path is not None else None
        file_hash = _file_hash(path) if path is not None else None
        with self.connection:
            self._upsert_paper(paper)
            previous = self.connection.execute(
                "SELECT document_id, file_path, file_hash FROM documents WHERE source_key = ?",
                (source_key,),
            ).fetchone()
            if previous is None:
                self.connection.execute(
                    """INSERT INTO documents
                    (source_key, paper_key, source_version, file_path, file_hash,
                     indexing_state, failure_message, last_attempt_at, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL)""",
                    (
                        source_key,
                        paper.zotero_key,
                        _version_value(version),
                        path_text,
                        file_hash,
                        state,
                        detail,
                    ),
                )
            else:
                self.connection.execute(
                    """UPDATE documents SET
                    paper_key = ?, source_version = ?, file_path = ?, file_hash = ?,
                    indexing_state = ?, failure_message = ?, last_attempt_at = CURRENT_TIMESTAMP
                    WHERE document_id = ?""",
                    (
                        paper.zotero_key,
                        _version_value(version),
                        path_text if path_text is not None else previous["file_path"],
                        file_hash if path_text is not None else previous["file_hash"],
                        state,
                        detail,
                        previous["document_id"],
                    ),
                )

    def mark_sync_complete(self) -> None:
        """Record completion only after the full source enumeration succeeds."""
        self._require_v2(writable=True)
        with self.connection:
            self.connection.execute(
                """UPDATE index_meta SET
                last_successful_sync = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE singleton = 1"""
            )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        query_text: str | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve nearest chunks, optionally nudged by lexical overlap."""
        self._require_v2()
        if limit <= 0:
            return []
        self.assert_compatible(self.embedding_id)
        metadata = self._metadata()
        dimension = metadata["dimension"]
        if dimension is None or not self._table_exists("vec_chunks"):
            return []
        if len(vector) != dimension:
            raise IndexCompatibilityError(
                f"Query embedding has dimension {len(vector)}, but index dimension is {dimension}. "
                "Run `bibliograph sync --rebuild` to create a compatible index."
            )

        candidate_limit = max(limit * 8, 50)
        rows = self.connection.execute(
            """SELECT chunk_id, distance FROM vec_chunks
            WHERE embedding MATCH ? AND k = ? ORDER BY distance""",
            (sqlite_vec.serialize_float32(_normalize(vector)), candidate_limit),
        ).fetchall()
        if not rows:
            return []

        chunk_ids = [row["chunk_id"] for row in rows]
        placeholders = ", ".join("?" for _ in chunk_ids)
        joined = self.connection.execute(
            f"""SELECT c.chunk_id, c.ordinal, c.page, c.section, c.content_kind, c.text,
                p.paper_key, p.title, p.authors_json, p.year, p.doi, p.collections_json
            FROM chunks AS c
            JOIN documents AS d ON d.document_id = c.document_id
            JOIN papers AS p ON p.paper_key = d.paper_key
            WHERE c.chunk_id IN ({placeholders})""",
            chunk_ids,
        ).fetchall()
        chunks = {row["chunk_id"]: self._chunk_from_row(row) for row in joined}

        scored: list[ScoredChunk] = []
        for row in rows:
            chunk = chunks.get(row["chunk_id"])
            if chunk is None:
                continue
            semantic = 1.0 / (1.0 + float(row["distance"]))
            lexical = _lexical_overlap(query_text, chunk.text) if query_text else 0.0
            scored.append((chunk, 0.75 * semantic + 0.25 * lexical))
        return sorted(scored, key=lambda hit: hit[1], reverse=True)[:limit]

    def context_for(self, chunk: Chunk, *, window: int = 1, max_words: int = 600) -> str:
        """Return nearby chunks from the same indexed document."""
        self._require_v2()
        if window < 0 or max_words <= 0:
            raise ValueError("window must be non-negative and max_words must be positive")
        row = self.connection.execute(
            "SELECT document_id, ordinal FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
        ).fetchone()
        if row is None:
            return ""
        rows = self.connection.execute(
            """SELECT text FROM chunks
            WHERE document_id = ? AND ordinal BETWEEN ? AND ?
            ORDER BY ordinal""",
            (row["document_id"], row["ordinal"] - window, row["ordinal"] + window),
        ).fetchall()
        words = " ".join(item["text"] for item in rows).split()
        return " ".join(words[:max_words])

    def _initialize_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE papers (
                    paper_key TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    year TEXT,
                    doi TEXT,
                    collections_json TEXT NOT NULL
                );
                CREATE TABLE documents (
                    document_id INTEGER PRIMARY KEY,
                    source_key TEXT NOT NULL UNIQUE,
                    paper_key TEXT NOT NULL REFERENCES papers(paper_key),
                    source_version TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    indexing_state TEXT NOT NULL DEFAULT 'pending',
                    failure_message TEXT,
                    last_attempt_at TEXT,
                    indexed_at TEXT
                );
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(document_id)
                        ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    page INTEGER,
                    section TEXT,
                    content_kind TEXT NOT NULL DEFAULT 'text',
                    text TEXT NOT NULL
                );
                CREATE INDEX chunks_document_ordinal ON chunks(document_id, ordinal);
                CREATE TABLE index_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL,
                    embedding_id TEXT,
                    dimension INTEGER,
                    extraction_chunking_fingerprint TEXT,
                    content_kind TEXT NOT NULL DEFAULT 'text',
                    last_successful_sync TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO index_meta
                    (singleton, schema_version, embedding_id, dimension,
                     extraction_chunking_fingerprint, content_kind, last_successful_sync,
                     created_at, updated_at)
                VALUES (1, 2, NULL, NULL, NULL, 'text', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
                PRAGMA user_version = 2;
                """
            )
            if self.embedding_id is not None:
                self.connection.execute(
                    "UPDATE index_meta SET embedding_id = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE singleton = 1",
                    (self.embedding_id,),
                )
            if self.indexing_fingerprint is not None:
                self.connection.execute(
                    "UPDATE index_meta SET extraction_chunking_fingerprint = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE singleton = 1",
                    (self.indexing_fingerprint,),
                )

    def _detect_schema_version(self) -> int:
        if self._table_exists("index_meta"):
            columns = self._table_columns("index_meta")
            v2_columns = {
                "singleton",
                "schema_version",
                "embedding_id",
                "dimension",
                "extraction_chunking_fingerprint",
                "last_successful_sync",
            }
            if v2_columns <= columns and {"papers", "documents", "chunks"} <= self._table_names():
                row = self.connection.execute(
                    "SELECT schema_version FROM index_meta WHERE singleton = 1"
                ).fetchone()
                return int(row[0]) if row is not None else 0
        if self._table_exists("vector_meta"):
            columns = self._table_columns("vector_meta")
            if "dimension" in columns and self._table_exists("chunks"):
                return 1
        return 0

    def _prepare_batches(
        self, batches: Iterable[tuple[list[Chunk], list[list[float]]]]
    ) -> tuple[list[tuple[list[Chunk], list[list[float]]]], int | None]:
        prepared: list[tuple[list[Chunk], list[list[float]]]] = []
        dimension: int | None = None
        seen_chunk_ids: set[str] = set()
        for chunks, embeddings in batches:
            if len(chunks) != len(embeddings):
                raise ValueError("Each chunk batch must have exactly one embedding per chunk")
            normalized_embeddings: list[list[float]] = []
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                if chunk.content_kind != "text":
                    raise ValueError(
                        f"Unsupported content kind {chunk.content_kind!r}; "
                        "this index stores text only"
                    )
                if not chunk.text.strip():
                    raise ValueError(f"Chunk {chunk.chunk_id!r} has no text")
                if chunk.chunk_id in seen_chunk_ids:
                    raise ValueError(f"Duplicate chunk id in replacement: {chunk.chunk_id}")
                seen_chunk_ids.add(chunk.chunk_id)
                if not embedding:
                    raise ValueError(f"Chunk {chunk.chunk_id!r} has an empty embedding")
                if dimension is None:
                    dimension = len(embedding)
                elif len(embedding) != dimension:
                    raise ValueError("All embeddings for one database must share a dimension")
                normalized_embeddings.append(_normalize(embedding))
            prepared.append((list(chunks), normalized_embeddings))
        return prepared, dimension

    def _assert_write_compatible(self) -> None:
        self.assert_compatible(self.embedding_id, self.indexing_fingerprint)

    def _create_vector_table(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self._table_exists("vec_chunks"):
            return
        self.connection.execute(
            "CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[" + str(dimension) + "]"
            ", +chunk_id text)"
        )

    def _upsert_paper(self, paper: Paper) -> None:
        self.connection.execute(
            """INSERT INTO papers
            (paper_key, title, authors_json, year, doi, collections_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(paper_key) DO UPDATE SET
                title = excluded.title,
                authors_json = excluded.authors_json,
                year = excluded.year,
                doi = excluded.doi,
                collections_json = excluded.collections_json""",
            (
                paper.zotero_key,
                paper.title,
                json.dumps(list(paper.authors)),
                paper.year,
                paper.doi,
                json.dumps(list(paper.collections)),
            ),
        )

    def _delete_document_vectors(self, document_id: int) -> None:
        if not self._table_exists("vec_chunks"):
            return
        self.connection.execute(
            """DELETE FROM vec_chunks WHERE chunk_id IN (
                SELECT chunk_id FROM chunks WHERE document_id = ?
            )""",
            (document_id,),
        )

    def _chunk_from_row(self, row: sqlite3.Row) -> Chunk:
        paper = Paper(
            zotero_key=row["paper_key"],
            title=row["title"],
            authors=tuple(json.loads(row["authors_json"])),
            year=row["year"],
            doi=row["doi"],
            collections=tuple(json.loads(row["collections_json"])),
        )
        return Chunk(
            chunk_id=row["chunk_id"],
            paper=paper,
            text=row["text"],
            page=row["page"],
            section=row["section"],
            ordinal=row["ordinal"],
            content_kind=row["content_kind"],
        )

    def _metadata(self) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM index_meta WHERE singleton = 1").fetchone()
        if row is None:
            raise IndexCompatibilityError(_rebuild_message(self.path, self._schema_version))
        return row

    def _require_v2(self, *, writable: bool = False) -> None:
        self._require_open()
        if self._schema_version != SCHEMA_VERSION:
            raise IndexCompatibilityError(_rebuild_message(self.path, self._schema_version))
        if writable and self.mode != "write":
            raise PermissionError("SQLiteStore is read-only; use mode='write' for index updates")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteStore is closed")

    def _table_names(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def _has_user_tables(self) -> bool:
        return bool(self._table_names())

    def _table_exists(self, table: str) -> bool:
        return table in self._table_names()

    def _table_columns(self, table: str) -> set[str]:
        if not self._table_exists(table):
            return set()
        return {str(row["name"]) for row in self.connection.execute(f"PRAGMA table_info({table})")}

    def _count(self, table: str) -> int:
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def _count_if_exists(self, table: str) -> int:
        return self._count(table) if self._table_exists(table) else 0

    def _pending_failure_count(self) -> int:
        if "indexing_state" not in self._table_columns("documents"):
            return 0
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM documents WHERE indexing_state IN ('missing', 'failed')"
            ).fetchone()[0]
        )

    def _legacy_embedding_id(self) -> str | None:
        if "embedding_id" not in self._table_columns("vector_meta"):
            return None
        row = self.connection.execute("SELECT embedding_id FROM vector_meta LIMIT 1").fetchone()
        return str(row[0]) if row is not None and row[0] is not None else None

    def _legacy_dimension(self) -> int | None:
        if not self._table_exists("vector_meta"):
            return None
        row = self.connection.execute("SELECT dimension FROM vector_meta LIMIT 1").fetchone()
        return int(row[0]) if row is not None and row[0] is not None else None

    def _legacy_paper_count(self) -> int:
        if not self._table_exists("chunks"):
            return 0
        columns = self._table_columns("chunks")
        if "zotero_key" not in columns:
            return 0
        return int(
            self.connection.execute("SELECT COUNT(DISTINCT zotero_key) FROM chunks").fetchone()[0]
        )


@contextmanager
def open_staging(
    path: str | Path,
    embedding_id: str | None,
    *,
    indexing_fingerprint: str | None = None,
) -> Generator[SQLiteStore, None, None]:
    """Build a sibling database and atomically replace ``path`` on success.

    A successful staging rebuild must contain at least one document, chunk, and
    vector.  Any exception (including validation failure) closes and removes the
    staging database, leaving the live database untouched.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_path = target.with_name(f".{target.name}.{uuid4().hex}.staging")
    store = SQLiteStore(
        staging_path,
        mode="write",
        embedding_id=embedding_id,
        indexing_fingerprint=indexing_fingerprint,
    )
    try:
        yield store
        summary = store.stats()
        if not all(summary[name] > 0 for name in ("documents", "chunks", "vectors")):
            raise ValueError(
                "Refusing to replace the live database with an empty rebuild; "
                "expected at least one document, chunk, and vector."
            )
        store.close()
        os.replace(staging_path, target)
        _remove_sidecars(target)
    except BaseException:
        store.close()
        _remove_sqlite_files(staging_path)
        raise


def _readonly_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def _rebuild_message(path: Path, schema_version: int) -> str:
    if schema_version == 1:
        detail = "This database uses legacy schema v1"
    elif schema_version == 0:
        detail = "This database has no recognised Bibliograph schema"
    else:
        detail = f"This database uses unsupported schema v{schema_version}"
    return f"{detail}. Run `bibliograph sync --rebuild` to rebuild {path}."


def _file_hash(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = sha256()
    with candidate.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_value(version: str | int | None) -> str | None:
    return None if version is None else str(version)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = set(re.findall(r"[\w-]+", query.lower()))
    text_terms = set(re.findall(r"[\w-]+", text.lower()))
    return len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def _remove_sidecars(path: Path) -> None:
    for candidate in (Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass
