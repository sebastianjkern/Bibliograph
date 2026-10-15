"""Ikarus-backed paper graph, vector, and lexical indexing for Bibliograph."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from inspect import Parameter, signature
from pathlib import Path
from threading import RLock
from typing import Any

from adapters.providers.filesystems.fsspec import LocalFilesystem
from adapters.providers.graph_stores import SQLiteGraphStore
from adapters.providers.lexical_indexes import SQLiteFTSIndex
from adapters.providers.vector_stores import sqlite_vec, vectorlite
from core.knowledge.graph import (
    DocumentRecord,
    PersonalizedPageRankExpansion,
)
from core.knowledge.index_manifest import IndexManifest
from core.knowledge.ingest import ingest as ingest_documents
from core.knowledge.rag import retrieve as retrieve_rag
from core.knowledge.retrieval import QueryHypothesis as IkarusQueryHypothesis
from core.knowledge.retrieval import QueryPlan
from core.knowledge.vector_store import configure as configure_vector_store

from ..domain import Chunk, Paper, RetrievalScore, ScoredChunk
from ..knowledge import KnowledgeDocument, QueryRequest, RetrievalResult, RetrievalTrace
from ..pipeline.classification import PROFILE_VERSION
from ..pipeline.indexing import INDEXING_FINGERPRINT
from ..pipeline.timing import TimingRecorder
from .context import join_chunk_context
from .ikarus_errors import IkarusIndexError
from .ikarus_index import (
    _can_seed_ikarus_index as _can_seed_ikarus_index,
)
from .ikarus_index import (
    inspect_ikarus_index as _inspect_ikarus_index,
)
from .ikarus_index import (
    is_ikarus_index as _is_ikarus_index,
)
from .ikarus_index import (
    open_ikarus_staging as _open_ikarus_staging,
)
from .ikarus_records import (
    _chunk_paper,
    _optional_int,
    _optional_str,
    _paper_from_metadata,
    _paper_graph,
    _paper_metadata,
)

_vector_store = (
    sqlite_vec
    if os.environ.get("IKARUS_VECTOR_STORE", "vectorlite").lower() == "sqlite_vec"
    else vectorlite
)
_VECTOR_METADATA_TABLE = "vector_metadata" if _vector_store is sqlite_vec else "vectorlite_metadata"

INDEX_SCHEMA_VERSION = 1


def _accepts_keyword(function, name: str) -> bool:
    try:
        parameters = signature(function).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


GRAPH_PROFILE_FINGERPRINT = "bibliograph-paper-sections@1"
LEXICAL_SCHEMA_VERSION = 1


class _ReadCachedGraphStore:
    """Bounded cache for immutable graph reads during a read-only search session."""

    def __init__(self, store) -> None:
        self._store = store
        self._get_node = lru_cache(maxsize=50_000)(store.get_node)
        self._get_edges = lru_cache(maxsize=50_000)(self._load_edges)

    def _load_edges(self, identifier: str, relations: tuple[str, ...] | None):
        return tuple(self._store.edges_from(identifier, relations))

    def get_node(self, identifier: str):
        return self._get_node(identifier)

    def edges_from(self, identifier: str, relations: Sequence[str] | None = None):
        relation_key = None if relations is None else tuple(relations)
        return list(self._get_edges(identifier, relation_key))

    def close(self) -> None:
        self._get_node.cache_clear()
        self._get_edges.cache_clear()
        self._store.close()

    def __getattr__(self, name: str):
        return getattr(self._store, name)


class _ProgressGraphStore:
    """Report bounded graph traversal activity without changing graph results."""

    def __init__(self, store, progress: Callable[[str], None]) -> None:
        self._store = store
        self._progress = progress
        self._reads = 0

    def _report_read(self) -> None:
        self._reads += 1
        if self._reads % 100 == 0:
            self._progress(f"Graph expansion · {self._reads} node/edge lookups")

    def get_node(self, identifier: str):
        self._report_read()
        return self._store.get_node(identifier)

    def edges_from(self, identifier: str, relations: Sequence[str] | None = None):
        self._report_read()
        return self._store.edges_from(identifier, relations)

    def __getattr__(self, name: str):
        return getattr(self._store, name)


class _ProgressGraphExpansion:
    """Add stage messages around the unchanged graph expansion implementation."""

    def __init__(self, progress: Callable[[str], None]) -> None:
        self._progress = progress
        self._expansion = PersonalizedPageRankExpansion()

    def expand(self, seeds, store, *, max_depth, max_nodes, relations=None):
        self._progress(f"Graph expansion started · depth {max_depth}, at most {max_nodes} nodes")
        hits = self._expansion.expand(
            seeds,
            store,
            max_depth=max_depth,
            max_nodes=max_nodes,
            relations=relations,
        )
        self._progress(f"Graph expansion complete · {len(hits)} nodes")
        return hits


class _TimedGraphExpansion:
    """Measure graph expansion without changing its inputs or results."""

    def __init__(self, expansion, timings: TimingRecorder) -> None:
        self._expansion = expansion
        self._timings = timings

    def expand(self, seeds, store, *, max_depth, max_nodes, relations=None):
        with self._timings.measure("graph_expansion"):
            return self._expansion.expand(
                seeds,
                store,
                max_depth=max_depth,
                max_nodes=max_nodes,
                relations=relations,
            )


def _configure_vectors(path: Path, index_namespace: str | None = None) -> None:
    if _vector_store is vectorlite:
        vectorlite.set_database(path, index_namespace)
    else:
        if _vector_store._connection is not None:
            _vector_store._connection.close()
            _vector_store._connection = None
        _vector_store._DATABASE = str(path)
    configure_vector_store(
        _vector_store.add,
        _vector_store.delete,
        _vector_store.clear,
        _vector_store.query,
        _vector_store.get_metadata,
        _vector_store.delete_document,
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
        vector_index_namespace: str | None = None,
    ) -> None:
        if mode not in {"read", "write"}:
            raise ValueError("mode must be 'read' or 'write'")
        self.path = Path(path)
        self._assessment_cache = None
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
        _configure_vectors(self.path, vector_index_namespace)
        graph = SQLiteGraphStore(self.path)
        self.graph = _ReadCachedGraphStore(graph) if mode == "read" else graph
        self.lexical = SQLiteFTSIndex(self.path)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection_lock = RLock()
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
        if _vector_store._connection is not None:
            _vector_store._connection.close()
            _vector_store._connection = None

    def workflow_cache(self):
        """Return persistent workflow caches beside this index, created on first use."""
        if self._assessment_cache is None:
            from ..pipeline.assessment_cache import SQLiteAssessmentCache

            if str(self.path) == ":memory:":
                return None
            cache_path = self.path.with_name(self.path.name + ".assessment-cache.sqlite3")
            try:
                self._assessment_cache = SQLiteAssessmentCache(cache_path)
            except (OSError, sqlite3.Error) as error:
                # Caching is an optimization; read-only or restricted deployments
                # should continue to work without it.
                from ..pipeline.assessment_cache import logger

                logger.warning("Evidence assessment cache unavailable: %s", error)
                return None
        return self._assessment_cache

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
        if progress is not None and _accepts_keyword(extract_pages, "progress"):
            pages = extract_pages(document.path, progress=progress)
        else:
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

    def repair_missing_vectors(self, *, batch_size: int = 32) -> int:
        """Re-embed graph chunks missing vectors inside the current staged database."""
        self._require_write()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows = self.connection.execute(
            f"SELECT nodes.identifier, nodes.document_id, nodes.content, nodes.metadata "
            "FROM graph_nodes AS nodes "
            f"LEFT JOIN {_VECTOR_METADATA_TABLE} AS vectors "
            "ON vectors.identifier=nodes.identifier "
            "WHERE nodes.kind='chunk' AND vectors.identifier IS NULL "
            "ORDER BY nodes.document_id, nodes.identifier"
        ).fetchall()
        embed_many = self.embedding.get("embed_documents")
        if rows and embed_many is None:
            raise IkarusIndexError(
                "The configured embedding provider cannot repair missing vectors."
            )
        repaired = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [str(row["content"]) for row in batch]
            vectors = embed_many(texts)
            if len(vectors) != len(batch):
                raise IkarusIndexError("Embedding provider returned an incomplete repair batch.")
            for row, vector in zip(batch, vectors, strict=True):
                if len(vector) != self.embedding_dimension:
                    raise IkarusIndexError("Embedding dimension changed while repairing the index.")
                metadata = json.loads(row["metadata"] or "{}")
                metadata.update({"document_id": row["document_id"], "text": row["content"]})
                _vector_store.add(row["identifier"], vector, metadata)
                repaired += 1
        return repaired

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

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed search queries together through the configured batch provider."""
        self._require_open()
        return self.embedding["embed_queries"](texts)

    def retrieve_request(
        self,
        request: QueryRequest,
        *,
        limit: int = 10,
        include_trace: bool = False,
        progress: Callable[[str], None] | None = None,
        query_vector: Sequence[float] | None = None,
        timings: TimingRecorder | None = None,
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

        def report(message: str) -> None:
            if progress is not None:
                progress(message)

        def embed_query(text: str) -> list[float]:
            if query_vector is not None:
                report("Using precomputed query embedding")
                return list(query_vector)
            report("Embedding query")
            if timings is None:
                vector = self.embedding["embed_queries"]([text])[0]
            else:
                with timings.measure("embedding"):
                    vector = self.embedding["embed_queries"]([text])[0]
            report("Query embedding complete")
            return vector

        def search_vectors(vector: list[float], k: int, filters):
            report(f"Vector search · top {k}")
            if timings is None:
                hits = _vector_store.query(vector, k, filters)
            else:
                with timings.measure("vector_search"):
                    hits = _vector_store.query(vector, k, filters)
            report(f"Vector search complete · {len(hits)} candidates")
            return hits

        def search_lexically(text: str, k: int, filters):
            report(f"Lexical search · top {k}")
            if timings is None:
                hits = self.lexical.search(text, k, filters)
            else:
                with timings.measure("lexical_search"):
                    hits = self.lexical.search(text, k, filters)
            report(f"Lexical search complete · {len(hits)} candidates")
            return hits

        graph_store = self.graph
        graph_expansion = PersonalizedPageRankExpansion()
        if progress is not None:
            graph_store = _ProgressGraphStore(self.graph, progress)
            graph_expansion = _ProgressGraphExpansion(progress)
        if timings is not None:
            graph_expansion = _TimedGraphExpansion(graph_expansion, timings)
        results = retrieve_rag(
            query,
            k=max(limit * 4, 12),
            embedder=embed_query
            if (progress is not None or timings is not None)
            else (lambda text: self.embedding["embed_queries"]([text])[0]),
            vector_query=(
                search_vectors
                if (progress is not None or timings is not None)
                else _vector_store.query
            ),
            metadata_reader=_vector_store.get_metadata,
            lexical_index=self.lexical,
            lexical_query=(
                search_lexically if (progress is not None or timings is not None) else None
            ),
            graph_store=graph_store,
            graph_expansion=graph_expansion,
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
                with self._connection_lock:
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
            "SELECT version, path, mtime_ns, size, state "
            "FROM bibliograph_sources WHERE source_key=?",
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

        def count(sql: str) -> int:
            return int(self.connection.execute(sql).fetchone()[0])

        return {
            "path": str(self.path),
            "schema_version": INDEX_SCHEMA_VERSION,
            "papers": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='paper'"),
            "documents": count(
                "SELECT COUNT(*) FROM bibliograph_sources WHERE state IN ('indexed','empty')"
            ),
            "chunks": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='chunk'"),
            "vectors": count(f"SELECT COUNT(*) FROM {_VECTOR_METADATA_TABLE}"),
            "missing_vectors": count(
                f"SELECT COUNT(*) FROM graph_nodes AS nodes "
                f"LEFT JOIN {_VECTOR_METADATA_TABLE} AS vectors "
                "ON vectors.identifier=nodes.identifier "
                "WHERE nodes.kind='chunk' AND vectors.identifier IS NULL"
            ),
            "embedding_id": self.embedding_id,
            "dimension": self.embedding_dimension,
            "extraction_chunking_fingerprint": INDEXING_FINGERPRINT,
            "last_successful_sync": self._last_sync(),
            "pending_failures": count(
                "SELECT COUNT(*) FROM bibliograph_sources WHERE state NOT IN ('indexed','empty')"
            ),
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
            raise IkarusIndexError(
                "Index backend or embedding differs; run `bibliograph sync --rebuild`."
            )
        self.connection.commit()

    def _delete_document(self, source_key: str) -> None:
        _vector_store.delete_document(source_key)
        self.graph.delete_document(source_key)
        self.lexical.delete_document(source_key)

    def _require_write(self) -> None:
        self._require_open()
        if self.mode != "write":
            raise IkarusIndexError("a read-only backend cannot index documents")

    def _require_open(self) -> None:
        if self.connection is None:
            raise IkarusIndexError("backend is closed")


def is_ikarus_index(path: str | Path) -> bool:
    return _is_ikarus_index(path)


def inspect_ikarus_index(path: str | Path) -> dict[str, Any]:
    return _inspect_ikarus_index(path, vector_metadata_table=_VECTOR_METADATA_TABLE)


def open_ikarus_staging(
    path: str | Path,
    *,
    embedding: Mapping[str, Any],
    ingestion_strategy: Callable[[str], dict[str, Any]] | None = None,
    classification_enabled: bool = False,
    seed_existing: bool = False,
):
    return _open_ikarus_staging(
        path,
        embedding=embedding,
        ingestion_strategy=ingestion_strategy,
        classification_enabled=classification_enabled,
        seed_existing=seed_existing,
        backend_factory=IkarusBackend,
        vector_store=_vector_store,
    )
