from __future__ import annotations

import sqlite3

import pytest

from bibliograph.adapters.sqlite import IndexCompatibilityError, SQLiteStore, open_staging
from bibliograph.domain import Chunk, Paper


def _paper(key: str = "P1") -> Paper:
    return Paper(key, "A Paper", ("Ada",), "2024", "10.1/example", ("COL",))


def _chunk(paper: Paper, index: int, text: str = "Evidence for a claim") -> Chunk:
    return Chunk(f"{paper.zotero_key}:{index}", paper, text, page=index + 1, ordinal=index)


def _replace(
    store: SQLiteStore,
    path,
    *,
    source_key: str = "ATTACHMENT",
    paper: Paper | None = None,
    chunks: list[Chunk] | None = None,
) -> int:
    paper = paper or _paper()
    chunks = chunks or [_chunk(paper, 0)]
    vectors = [[1.0, 0.0] for _ in chunks]
    return store.replace_document(paper, source_key, "1", path, [(chunks, vectors)])


def test_read_mode_never_mutates_an_existing_database(tmp_path) -> None:
    database = tmp_path / "index.db"
    document = tmp_path / "paper.pdf"
    document.write_text("PDF bytes")
    with SQLiteStore(database, mode="write", embedding_id="model-a") as store:
        _replace(store, document)
    before = database.read_bytes()

    with SQLiteStore(database, mode="read", embedding_id="model-a") as store:
        assert store.stats()["chunks"] == 1
        assert store.search([1.0, 0.0])[0][0].chunk_id == "P1:0"

    assert database.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        SQLiteStore(tmp_path / "absent.db", mode="read")


def test_embedding_mismatch_requires_explicit_rebuild(tmp_path) -> None:
    database = tmp_path / "index.db"
    document = tmp_path / "paper.pdf"
    document.write_text("PDF bytes")
    with SQLiteStore(database, mode="write", embedding_id="model-a") as store:
        _replace(store, document)

    with SQLiteStore(database, embedding_id="model-b") as store:
        with pytest.raises(IndexCompatibilityError, match="sync --rebuild"):
            store.assert_compatible("model-b")
        with pytest.raises(IndexCompatibilityError, match="model-a"):
            store.search([1.0, 0.0])


def test_replace_document_removes_stale_chunks_and_vectors(tmp_path) -> None:
    database = tmp_path / "index.db"
    document = tmp_path / "paper.pdf"
    document.write_text("PDF bytes")
    paper = _paper()
    with SQLiteStore(database, mode="write", embedding_id="model-a") as store:
        _replace(
            store,
            document,
            paper=paper,
            chunks=[_chunk(paper, 0, "old first"), _chunk(paper, 1, "old second")],
        )
        assert _replace(
            store,
            document,
            paper=paper,
            chunks=[_chunk(paper, 2, "new only")],
        ) == 1
        assert store.stats() | {"mode": "write"} == {
            "path": str(database),
            "mode": "write",
            "schema_version": 2,
            "papers": 1,
            "documents": 1,
            "chunks": 1,
            "vectors": 1,
            "embedding_id": "model-a",
            "dimension": 2,
            "extraction_chunking_fingerprint": None,
            "last_successful_sync": None,
            "pending_failures": 0,
            "rebuild_required": False,
        }
        hits = store.search([1.0, 0.0])
        assert [hit[0].chunk_id for hit in hits] == ["P1:2"]
        assert store.context_for(hits[0][0]) == "new only"


def test_schema_v1_has_readable_status_but_cannot_be_used(tmp_path) -> None:
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            zotero_key TEXT NOT NULL,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            year TEXT,
            doi TEXT,
            collections TEXT NOT NULL,
            text TEXT NOT NULL,
            page INTEGER,
            section TEXT,
            chunk_index INTEGER NOT NULL,
            embedding TEXT NOT NULL
        );
        CREATE TABLE indexed_files (
            attachment_key TEXT PRIMARY KEY,
            source_version INTEGER,
            file_hash TEXT NOT NULL
        );
        CREATE TABLE vector_meta (dimension INTEGER NOT NULL, embedding_id TEXT);
        INSERT INTO vector_meta VALUES (2, 'legacy-model');
        INSERT INTO chunks VALUES
            ('C1', 'P1', 'Legacy', '[]', NULL, NULL, '[]', 'old text', NULL, NULL, 0, '[]');
        """
    )
    connection.commit()
    connection.close()

    with SQLiteStore(database, mode="read") as store:
        status = store.stats()
        assert status["schema_version"] == 1
        assert status["chunks"] == 1
        assert status["rebuild_required"] is True
        with pytest.raises(IndexCompatibilityError, match="sync --rebuild"):
            store.assert_compatible("new-model")


def test_failed_staging_rebuild_preserves_live_database(tmp_path) -> None:
    database = tmp_path / "index.db"
    document = tmp_path / "paper.pdf"
    document.write_text("PDF bytes")
    with SQLiteStore(database, mode="write", embedding_id="old-model") as store:
        _replace(store, document)

    with pytest.raises(RuntimeError, match="abort rebuild"):
        with open_staging(database, "new-model") as staging:
            new_paper = _paper("P2")
            _replace(staging, document, source_key="NEW", paper=new_paper)
            raise RuntimeError("abort rebuild")

    with SQLiteStore(database, mode="read", embedding_id="old-model") as store:
        assert store.stats()["embedding_id"] == "old-model"
        assert [hit[0].chunk_id for hit in store.search([1.0, 0.0])] == ["P1:0"]


def test_fresh_v2_database_has_readable_zero_vector_status(tmp_path) -> None:
    database = tmp_path / "fresh.db"

    with SQLiteStore(
        database,
        mode="write",
        embedding_id="model-a",
        indexing_fingerprint="pdf-v1",
    ) as store:
        summary = store.stats()

    assert summary["documents"] == summary["chunks"] == summary["vectors"] == 0
    assert summary["embedding_id"] == "model-a"
    assert summary["extraction_chunking_fingerprint"] == "pdf-v1"


def test_unknown_embedding_fingerprint_with_vectors_requires_rebuild(tmp_path) -> None:
    database = tmp_path / "unknown.db"
    document = tmp_path / "paper.pdf"
    document.write_text("PDF bytes")
    with SQLiteStore(database, mode="write") as store:
        _replace(store, document)

    with SQLiteStore(database, mode="read", embedding_id="model-a") as store:
        with pytest.raises(IndexCompatibilityError, match="without an embedding fingerprint"):
            store.assert_compatible("model-a")


def test_extraction_fingerprint_mismatch_requires_rebuild(tmp_path) -> None:
    database = tmp_path / "index.db"
    document = tmp_path / "paper.pdf"
    document.write_text("PDF bytes")
    with SQLiteStore(
        database,
        mode="write",
        embedding_id="model-a",
        indexing_fingerprint="pdf-v1",
    ) as store:
        _replace(store, document)

    with SQLiteStore(database, mode="read", embedding_id="model-a") as store:
        with pytest.raises(IndexCompatibilityError, match="extraction/chunking fingerprint"):
            store.assert_compatible("model-a", "pdf-v2")


def test_failures_and_empty_staging_rebuild_are_reported_safely(tmp_path) -> None:
    database = tmp_path / "index.db"
    paper = _paper()
    with SQLiteStore(
        database,
        mode="write",
        embedding_id="model-a",
        indexing_fingerprint="pdf-v1",
    ) as store:
        store.record_document_failure(
            paper,
            "MISSING",
            1,
            state="missing",
            detail="cache: not found",
        )
        store.mark_sync_complete()
        summary = store.stats()

    assert summary["pending_failures"] == 1
    assert summary["last_successful_sync"] is not None
    with pytest.raises(ValueError, match="empty rebuild"):
        with open_staging(tmp_path / "replacement.db", "model-b"):
            pass


def test_non_text_content_is_rejected_until_a_matching_space_exists(tmp_path) -> None:
    database = tmp_path / "index.db"
    document = tmp_path / "paper.pdf"
    document.write_text("PDF bytes")
    paper = _paper()
    image_chunk = Chunk("P1:image", paper, "image description", content_kind="image")
    with SQLiteStore(database, mode="write", embedding_id="model-a") as store:
        with pytest.raises(ValueError, match="content kind"):
            store.replace_document(paper, "A1", 1, document, [([image_chunk], [[1.0, 0.0]])])
