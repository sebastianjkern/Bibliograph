"""Ikarus-backed paper graph, vector, and lexical indexing for Bibliograph."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from adapters.providers.filesystems.fsspec import LocalFilesystem
from adapters.providers.graph_stores import SQLiteGraphStore
from adapters.providers.lexical_indexes import SQLiteFTSIndex
from adapters.providers.vector_stores import sqlite_vec
from core.knowledge.graph import DocumentRecord, GraphEdge, GraphNode, PersonalizedPageRankExpansion
from core.knowledge.index_manifest import IndexManifest
from core.knowledge.ingest import DocumentGraph
from core.knowledge.ingest import ingest as ingest_documents
from core.knowledge.rag import retrieve as retrieve_rag
from core.knowledge.retrieval import QueryHypothesis as IkarusQueryHypothesis
from core.knowledge.retrieval import QueryPlan
from core.knowledge.vector_store import configure as configure_vector_store

from ..domain import Chunk, Paper, RetrievalScore, ScoredChunk
from .context import join_chunk_context
from ..knowledge import KnowledgeDocument, QueryRequest, RetrievalResult, RetrievalTrace
from ..pipeline.classification import PROFILE_VERSION
from ..pipeline.indexing import INDEXING_FINGERPRINT

INDEX_SCHEMA_VERSION = 1
GRAPH_PROFILE_FINGERPRINT = "bibliograph-paper-sections@1"
LEXICAL_SCHEMA_VERSION = 1


class IkarusIndexError(RuntimeError):
    """An Ikarus-backed Bibliograph index is absent or incompatible."""


def _configure_vectors(path: Path) -> None:
    if sqlite_vec._connection is not None:
        sqlite_vec._connection.close()
        sqlite_vec._connection = None
    sqlite_vec._DATABASE = str(path)
    configure_vector_store(
        sqlite_vec.add,
        sqlite_vec.delete,
        sqlite_vec.clear,
        sqlite_vec.query,
        sqlite_vec.get_metadata,
        sqlite_vec.delete_document,
    )


class IkarusBackend:
    """Adapt Bibliograph's PDF/citation domain to Ikarus's generic stores."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedding: Mapping[str, Any],
        mode: str = "read",
        ingestion_strategy: Callable[[str], dict[str, Any]] | None = None,
        classification_enabled: bool = False,
    ) -> None:
        if mode not in {"read", "write"}:
            raise ValueError("mode must be 'read' or 'write'")
        self.path = Path(path)
        self.mode = mode
        self.embedding = embedding
        self.ingestion_strategy = ingestion_strategy
        self.embedding_id = str(embedding["id"])
        probe = embedding["probe"]()
        self.embedding_dimension = int(probe["dimension"])
        self.manifest = IndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            embedding_id=self.embedding_id,
            embedding_dimension=self.embedding_dimension,
            ingestion_fingerprint=INDEXING_FINGERPRINT,
            graph_profile_fingerprint=(
                f"{GRAPH_PROFILE_FINGERPRINT}+{PROFILE_VERSION}"
                if ingestion_strategy is not None or classification_enabled
                else GRAPH_PROFILE_FINGERPRINT
            ),
            lexical_schema_version=LEXICAL_SCHEMA_VERSION,
        )
        self._assert_index_kind(create=mode == "write")
        if mode == "write" and not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        _configure_vectors(self.path)
        self.graph = SQLiteGraphStore(self.path)
        self.lexical = SQLiteFTSIndex(self.path)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.manifests = None
        try:
            if mode == "write":
                self._initialize_bookkeeping()
            from adapters.indexing.sqlite_manifest import SQLiteIndexManifestStore

            self.manifests = SQLiteIndexManifestStore(self.path)
            if mode == "write":
                stored = self.manifests.read()
                if stored is None:
                    self.manifests.write(self.manifest)
                else:
                    stored.require_compatible(self.manifest)
            else:
                self.manifests.require_compatible(self.manifest)
        except BaseException:
            self.close()
            raise

    def _assert_index_kind(self, *, create: bool) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            if create:
                return
            raise IkarusIndexError(
                f"Ikarus index not found at {self.path}. Run `bibliograph sync --rebuild`."
            )
        try:
            connection = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
            try:
                row = connection.execute(
                    "SELECT backend FROM bibliograph_index_meta WHERE singleton=1"
                ).fetchone()
            finally:
                connection.close()
        except sqlite3.DatabaseError:
            row = None
        if row is None or row[0] != "ikarus":
            raise IkarusIndexError(
                "This index is not Ikarus-backed. Run `bibliograph sync --rebuild` to migrate it."
            )

    def __enter__(self) -> IkarusBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "manifests", None) is not None:
            self.manifests.close()
            self.manifests = None
        if getattr(self, "connection", None) is not None:
            self.connection.close()
            self.connection = None
        if getattr(self, "lexical", None) is not None:
            self.lexical.close()
            self.lexical = None
        if getattr(self, "graph", None) is not None:
            self.graph.close()
            self.graph = None
        if sqlite_vec._connection is not None:
            sqlite_vec._connection.close()
            sqlite_vec._connection = None

    def index_document(
        self,
        document: KnowledgeDocument,
        *,
        extract_pages: Callable[[str | Path], list[tuple[int, str, str | None]]],
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Index through Ikarus's generic ingestion pipeline.

        Bibliograph supplies the PDF extraction and paper-domain graph profile;
        Ikarus owns chunk storage, embedding batches, replacement, lexical
        indexing, and cleanup.
        """
        self._require_write()
        if progress is not None:
            progress("Extracting PDF pages")
        pages = extract_pages(document.path)
        chunks = _chunk_paper(document.paper, pages)
        if progress is not None:
            progress(f"Prepared {len(chunks)} chunks")
        record = DocumentRecord(
            document.source_key,
            document.path.resolve().as_uri(),
            None if document.version is None else str(document.version),
            _paper_metadata(document.paper, document.source_key, document.path),
        )

        def load_document(_source: str, _filesystem) -> str:
            return "\n\n".join(chunk.text for chunk in chunks)

        def chunk_document(_text: str):
            for chunk in chunks:
                yield {
                    "text": chunk.text,
                    "metadata": {
                        "kind": "chunk",
                        "page": chunk.page,
                        "section": chunk.section,
                        "ordinal": chunk.ordinal,
                        "content_kind": chunk.content_kind,
                    },
                }

        ingest_strategy = self.ingestion_strategy
        if ingest_strategy is not None and progress is not None:
            classified = 0

            def report_classification(text: str):
                nonlocal classified
                classified += 1
                progress(f"Classifying evidence · {classified}/{len(chunks)} chunks")
                prepared = ingest_strategy(text)
                if classified == len(chunks):
                    progress(f"Embedding and indexing {len(chunks)} chunks")
                return prepared

            ingest_strategy = report_classification
        elif progress is not None:
            progress(f"Embedding and indexing {len(chunks)} chunks")
        ingested = ingest_documents(
            str(document.path),
            filesystem=LocalFilesystem(),
            loader=load_document,
            chunker=chunk_document,
            strategy=ingest_strategy,
            document=record,
            batch_embedder=self.embedding["embed_documents"],
            graph_store=self.graph,
            document_strategy=lambda _record, nodes: _paper_graph(record, nodes),
            lexical_index=self.lexical,
        )
        stat = document.path.stat()
        with self.connection:
            self.connection.execute(
                """INSERT INTO bibliograph_sources(
                       source_key, document_id, version, path, mtime_ns, size, state, detail
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                   ON CONFLICT(source_key) DO UPDATE SET
                     document_id=excluded.document_id, version=excluded.version,
                     path=excluded.path, mtime_ns=excluded.mtime_ns, size=excluded.size,
                     state=excluded.state, detail=NULL, updated_at=CURRENT_TIMESTAMP""",
                (
                    document.source_key,
                    document.source_key,
                    document.version,
                    str(document.path.resolve()),
                    stat.st_mtime_ns,
                    stat.st_size,
                    "indexed" if ingested else "empty",
                ),
            )
        return {
            "source_key": document.source_key,
            "chunks": len(ingested),
            "path": str(document.path),
        }

    def index_pdf(
        self,
        paper: Paper,
        source_key: str,
        version: str | int | None,
        path: str | Path,
        *,
        extract_pages: Callable[[str | Path], list[tuple[int, str, str | None]]],
    ) -> dict[str, Any]:
        self._require_write()
        return self.index_document(
            KnowledgeDocument(
                source_key=source_key,
                paper=paper,
                path=Path(path),
                version=None if version is None else str(version),
            ),
            extract_pages=extract_pages,
        )


    def retrieve(
        self,
        query: str,
        *,
        alternatives: Sequence[str] = (),
        limit: int = 10,
    ) -> tuple[list[ScoredChunk], dict[str, dict[str, float]]]:
        """Compatibility wrapper for callers using the old string interface."""
        result = self.retrieve_request(
            QueryRequest.from_alternatives(query, alternatives),
            limit=limit,
        )
        return list(result.hits), dict(result.score_details)

    def retrieve_request(
        self,
        request: QueryRequest,
        *,
        limit: int = 10,
        include_trace: bool = False,
    ) -> RetrievalResult:
        """Retrieve through the knowledge-engine boundary.

        The current adapter translates the neutral request to the installed
        Ikarus API.  Trace support is intentionally opt-in; it becomes
        available here when the Ikarus engine exposes provenance data.
        """
        self._require_open()
        query = request.text
        hypotheses = tuple(
            IkarusQueryHypothesis(
                hypothesis.text,
                hypothesis.kind,
                hypothesis.weight,
                hypothesis.entities,
                hypothesis.relations,
            )
            for hypothesis in request.hypotheses
        )
        plan = QueryPlan.from_hypotheses(query, hypotheses)
        trace_payload: dict[str, object] | None = {} if include_trace else None
        results = retrieve_rag(
            query,
            k=max(limit * 4, 12),
            embedder=lambda text: self.embedding["embed_queries"]([text])[0],
            vector_query=sqlite_vec.query,
            metadata_reader=sqlite_vec.get_metadata,
            lexical_index=self.lexical,
            graph_store=self.graph,
            graph_expansion=PersonalizedPageRankExpansion(),
            graph_max_depth=2,
            graph_max_nodes=max(100, limit * 12),
            graph_relations=("contains", "in_section", "next_chunk"),
            score_weights={"semantic": 0.55, "lexical": 0.2, "graph": 0.25},
            planner=lambda _query: plan,
            trace=trace_payload,
        )
        hits: list[ScoredChunk] = []
        details: dict[str, dict[str, float]] = {}
        for result in results:
            metadata = dict(result.metadata or {})
            if "kind" not in metadata:
                row = self.connection.execute(
                    "SELECT kind FROM graph_nodes WHERE identifier=?",
                    (result.identifier,),
                ).fetchone()
                if row is not None:
                    metadata["kind"] = row[0]
            if metadata.get("kind") != "chunk":
                continue
            paper = _paper_from_metadata(metadata, result.identifier)
            score_components = result.score_components or {}
            score = RetrievalScore(
                result.score,
                semantic=float(score_components.get("semantic", 0.0)),
                lexical=float(score_components.get("lexical", 0.0)),
            )
            chunk = Chunk(
                result.identifier,
                paper,
                result.text,
                page=_optional_int(metadata.get("page")),
                section=_optional_str(metadata.get("section")),
                ordinal=int(metadata.get("ordinal", 0)),
                content_kind=_optional_str(metadata.get("content_kind")) or "text",
                evidence_role=_optional_str(metadata.get("evidence_role")),
            )
            hits.append((chunk, score))
            details[chunk.chunk_id] = {
                "vector": float(score_components.get("semantic", 0.0)),
                "lexical": float(score_components.get("lexical", 0.0)),
                "graph": float(score_components.get("graph", 0.0)),
                "retrieval": float(result.score),
            }
            if len(hits) >= limit:
                break
        trace = None
        if trace_payload is not None:
            trace = RetrievalTrace(
                nodes=tuple(trace_payload.get("nodes", ())),
                edges=tuple(trace_payload.get("edges", ())),
                paths=tuple(trace_payload.get("paths", ())),
                metadata={
                    "queries": tuple(trace_payload.get("queries", ())),
                    "hypotheses": tuple(trace_payload.get("hypotheses", ())),
                    "seeds": tuple(trace_payload.get("seeds", ())),
                },
            )
        return RetrievalResult(
            hits=tuple(hits),
            score_details=details,
            trace=trace,
        )

    def needs_document(self, source_key: str, version: str | int | None, path: str | Path) -> bool:
        self._require_open()
        row = self.connection.execute(
            "SELECT version, path, mtime_ns, size, state FROM bibliograph_sources WHERE source_key=?",
            (source_key,),
        ).fetchone()
        if row is None or row["state"] not in {"indexed", "empty"}:
            return True
        try:
            stat = Path(path).stat()
        except OSError:
            return True
        return not (
            row["version"] == (None if version is None else str(version))
            and row["path"] == str(Path(path).resolve())
            and row["mtime_ns"] == stat.st_mtime_ns
            and row["size"] == stat.st_size
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
        self._require_open()
        with self.connection:
            self.connection.execute(
                """INSERT INTO bibliograph_sources(source_key, version, path, state, detail)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(source_key) DO UPDATE SET version=excluded.version,
                     path=excluded.path, state=excluded.state, detail=excluded.detail,
                     updated_at=CURRENT_TIMESTAMP""",
                (
                    source_key,
                    None if version is None else str(version),
                    str(Path(path).resolve()) if path is not None else None,
                    state,
                    detail,
                ),
            )

    def mark_sync_complete(self) -> None:
        self._require_open()
        with self.connection:
            self.connection.execute(
                "UPDATE bibliograph_index_meta SET last_successful_sync=CURRENT_TIMESTAMP "
                "WHERE singleton=1"
            )

    def context_for(self, chunk: Chunk, *, window: int = 1, max_words: int = 360) -> str:
        """Return a chunk with nearby same-section passages as explicit context."""
        self._require_open()
        if window < 0 or max_words < 1:
            raise ValueError("window must be non-negative and max_words must be positive")
        row = self.connection.execute(
            "SELECT document_id, metadata FROM graph_nodes WHERE identifier=? AND kind='chunk'",
            (chunk.chunk_id,),
        ).fetchone()
        if row is None:
            return chunk.text
        document_id, raw_metadata = row
        metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        metadata = metadata if isinstance(metadata, dict) else {}
        ordinal = int(metadata.get("ordinal", chunk.ordinal))
        section = metadata.get("section", chunk.section)
        rows = self.connection.execute(
            "SELECT identifier, content, metadata FROM graph_nodes "
            "WHERE document_id=? AND kind='chunk'",
            (document_id,),
        ).fetchall()
        nearby = []
        for _identifier, content, raw in rows:
            details = json.loads(raw) if isinstance(raw, str) else raw
            details = details if isinstance(details, dict) else {}
            neighbor_ordinal = int(details.get("ordinal", -1))
            neighbor_section = details.get("section")
            if abs(neighbor_ordinal - ordinal) > window:
                continue
            if section and neighbor_section != section:
                continue
            nearby.append((neighbor_ordinal, str(content or "")))
        text = join_chunk_context(
            nearby,
            center_ordinal=ordinal,
            max_words=max_words,
        )
        return text or chunk.text

    def stats(self) -> dict[str, Any]:
        self._require_open()
        count = lambda sql: int(self.connection.execute(sql).fetchone()[0])
        return {
            "path": str(self.path),
            "schema_version": INDEX_SCHEMA_VERSION,
            "papers": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='paper'"),
            "documents": count("SELECT COUNT(*) FROM bibliograph_sources WHERE state IN ('indexed','empty')"),
            "chunks": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='chunk'"),
            "vectors": count("SELECT COUNT(*) FROM vector_metadata"),
            "embedding_id": self.embedding_id,
            "dimension": self.embedding_dimension,
            "extraction_chunking_fingerprint": INDEXING_FINGERPRINT,
            "last_successful_sync": self._last_sync(),
            "pending_failures": count("SELECT COUNT(*) FROM bibliograph_sources WHERE state NOT IN ('indexed','empty')"),
            "rebuild_required": False,
        }

    def _last_sync(self):
        row = self.connection.execute(
            "SELECT last_successful_sync FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        return row[0] if row else None

    def _initialize_bookkeeping(self) -> None:
        self.connection.executescript(
            """CREATE TABLE IF NOT EXISTS bibliograph_index_meta (
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
               );"""
        )
        self.connection.execute(
            """INSERT OR IGNORE INTO bibliograph_index_meta(
                   singleton, schema_version, backend, embedding_id,
                   extraction_chunking_fingerprint
               ) VALUES (1, ?, 'ikarus', ?, ?)""",
            (INDEX_SCHEMA_VERSION, self.embedding_id, INDEXING_FINGERPRINT),
        )
        row = self.connection.execute(
            "SELECT backend, embedding_id FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        if row["backend"] != "ikarus" or row["embedding_id"] != self.embedding_id:
            raise IkarusIndexError("Index backend or embedding differs; run `bibliograph sync --rebuild`.")
        self.connection.commit()

    def _delete_document(self, source_key: str) -> None:
        sqlite_vec.delete_document(source_key)
        self.graph.delete_document(source_key)
        self.lexical.delete_document(source_key)

    def _require_write(self) -> None:
        self._require_open()
        if self.mode != "write":
            raise IkarusIndexError("a read-only backend cannot index documents")

    def _require_open(self) -> None:
        if self.connection is None:
            raise IkarusIndexError("backend is closed")


def _paper_metadata(paper: Paper, source_key: str, path: str | Path) -> dict[str, Any]:
    return {
        "zotero_key": paper.zotero_key,
        "title": paper.title,
        "authors": list(paper.authors),
        "year": paper.year,
        "doi": paper.doi,
        "collections": list(paper.collections),
        "source_key": source_key,
        "path": str(Path(path)),
        "modality": "scientific-paper",
    }


def _chunk_metadata(chunk: Chunk, record: DocumentRecord) -> dict[str, Any]:
    return {
        **dict(record.metadata or {}),
        "source": record.uri or record.identifier,
        "document_id": record.identifier,
        "revision": record.revision,
        "kind": "chunk",
        "text": chunk.text,
        "page": chunk.page,
        "section": chunk.section,
        "ordinal": chunk.ordinal,
        "content_kind": chunk.content_kind,
    }


def _chunk_node(chunk: Chunk, record: DocumentRecord) -> GraphNode:
    return GraphNode(
        chunk.chunk_id,
        "chunk",
        chunk.text,
        record.identifier,
        _chunk_metadata(chunk, record),
    )


def _paper_graph(record: DocumentRecord, chunks: Sequence[GraphNode]):
    metadata = dict(record.metadata or {})
    paper = GraphNode(
        f"paper:{record.identifier}",
        "paper",
        str(metadata.get("title") or record.identifier),
        record.identifier,
        {**metadata, "kind": "paper", "document_id": record.identifier},
    )
    nodes = [paper]
    edges = []
    sections: dict[str, GraphNode] = {}
    previous = None
    for chunk in chunks:
        section_name = chunk.metadata.get("section")
        if section_name:
            key = str(section_name).casefold()
            section = sections.get(key)
            if section is None:
                section = GraphNode(
                    f"section:{record.identifier}:{len(sections)}",
                    "section",
                    str(section_name),
                    record.identifier,
                    {"kind": "section", "document_id": record.identifier, "title": str(section_name)},
                )
                sections[key] = section
                nodes.append(section)
                edges.append(GraphEdge(paper.identifier, section.identifier, "contains"))
            edges.append(GraphEdge(chunk.identifier, section.identifier, "in_section"))
            edges.append(GraphEdge(section.identifier, chunk.identifier, "contains"))
        else:
            edges.append(GraphEdge(paper.identifier, chunk.identifier, "contains"))
        if previous is not None:
            edges.append(GraphEdge(previous, chunk.identifier, "next_chunk"))
        previous = chunk.identifier
    return DocumentGraph(tuple(nodes), tuple(edges))


def _paper_from_metadata(metadata, fallback: str) -> Paper:
    return Paper(
        str(metadata.get("zotero_key") or fallback),
        str(metadata.get("title") or fallback),
        tuple(str(author) for author in metadata.get("authors", ()) or ()),
        _optional_str(metadata.get("year")),
        _optional_str(metadata.get("doi")),
        tuple(str(value) for value in metadata.get("collections", ()) or ()),
    )


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _optional_str(value) -> str | None:
    return None if value is None else str(value)


def _chunk_paper(paper: Paper, pages: Sequence[tuple[int, str, str | None]]) -> list[Chunk]:
    from ..adapters.pdf import chunk_document

    return chunk_document(paper, pages)


def is_ikarus_index(path: str | Path) -> bool:
    database = Path(path)
    if not database.is_file():
        return False
    connection = None
    try:
        connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT backend FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        return row is not None and row[0] == "ikarus"
    except sqlite3.DatabaseError:
        return False
    finally:
        if connection is not None:
            connection.close()


def inspect_ikarus_index(path: str | Path) -> dict[str, Any]:
    database = Path(path)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        meta = connection.execute(
            "SELECT * FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        manifest = json.loads(connection.execute(
            "SELECT manifest FROM ikarus_index_manifest WHERE index_name='default'"
        ).fetchone()[0])
        count = lambda sql: int(connection.execute(sql).fetchone()[0])
        return {
            "path": str(database),
            "schema_version": int(meta["schema_version"]),
            "papers": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='paper'"),
            "documents": count("SELECT COUNT(*) FROM bibliograph_sources WHERE state IN ('indexed','empty')"),
            "chunks": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='chunk'"),
            "vectors": count("SELECT COUNT(*) FROM vector_metadata"),
            "embedding_id": manifest["embedding_id"],
            "dimension": int(manifest["embedding_dimension"]),
            "extraction_chunking_fingerprint": manifest["ingestion_fingerprint"],
            "last_successful_sync": meta["last_successful_sync"],
            "pending_failures": count("SELECT COUNT(*) FROM bibliograph_sources WHERE state NOT IN ('indexed','empty')"),
            "rebuild_required": False,
        }
    finally:
        connection.close()


@contextmanager
def open_ikarus_staging(
    path: str | Path,
    *,
    embedding: Mapping[str, Any],
    ingestion_strategy: Callable[[str], dict[str, Any]] | None = None,
    classification_enabled: bool = False,
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.ikarus-staging-", dir=target.parent)
    os.close(descriptor)
    staging = Path(name)
    backend = None
    try:
        backend = IkarusBackend(
            staging,
            embedding=embedding,
            mode="write",
            ingestion_strategy=ingestion_strategy,
            classification_enabled=classification_enabled,
        )
        yield backend
        summary = backend.stats()
        backend.close()
        backend = None
        if not all(summary[field] > 0 for field in ("documents", "chunks", "vectors")):
            raise IkarusIndexError("Refusing to replace the active index with an incomplete staged index.")
        os.replace(staging, target)
    finally:
        if backend is not None:
            backend.close()
        for candidate in (staging, Path(f"{staging}-wal"), Path(f"{staging}-shm")):
            candidate.unlink(missing_ok=True)
