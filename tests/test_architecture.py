from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bibliograph import bootstrap
from bibliograph.adapters.sqlite import SQLiteStore
from bibliograph.commands.status import status
from bibliograph.domain import Chunk, Paper
from bibliograph.pipeline import llm_tasks
from bibliograph.pipeline.indexing import INDEXING_FINGERPRINT, index_document


def _hash_settings(db: str) -> dict:
    return {
        "db": db,
        "pdf_dir": "pdfs",
        "collection": "Collection",
        "zotero_storage_dir": None,
        "zotero": {"library_id": "library", "library_type": "user", "api_key": "key"},
        "remote": {"email": None, "openalex_api_key": None},
        "embedding": {
            "provider": "hash",
            "dimension": 8,
            "model": "hash",
            "batch_size": 2,
            "document_prefix": "",
            "query_prefix": "",
        },
        "llm": {"provider": "ollama", "model": "unused", "mode": "off", "stages": []},
        "acquisition": {"order": []},
    }


def _seed_store(path: Path) -> tuple[Paper, Chunk]:
    paper = Paper("P1", "A paper", ("Ada",), "2024")
    chunk = Chunk("P1:1:0", paper, "Road quality changes trade outcomes.", page=1)
    with SQLiteStore(
        path,
        mode="write",
        embedding_id="hash:8",
        indexing_fingerprint=INDEXING_FINGERPRINT,
    ) as store:
        store.replace_document(paper, "A1", 1, __file__, [([chunk], [[1.0] + [0.0] * 7])])
    return paper, chunk


def test_indexing_pipeline_batches_before_transactional_replacement():
    paper = Paper("P1", "A paper")
    chunks = [
        Chunk(f"P1:1:{index}", paper, f"Evidence {index}", ordinal=index)
        for index in range(5)
    ]
    batches = []
    embeds = []

    class Store:
        def replace_document(self, paper, source_key, version, path, values):
            batches.extend(values)
            return sum(len(chunk_batch) for chunk_batch, _vectors in batches)

    result = index_document(
        paper,
        "A1",
        1,
        "paper.pdf",
        extract_pages=lambda _path: [(1, "text", None)],
        chunk_document=lambda _paper, _pages: chunks,
        embed_documents=lambda texts: embeds.append(list(texts)) or [[1.0] for _ in texts],
        store=Store(),
        batch_size=2,
    )

    assert [len(batch) for batch in embeds] == [2, 2, 1]
    assert [len(chunk_batch) for chunk_batch, _vectors in batches] == [2, 2, 1]
    assert result["chunks"] == 5


def test_indexing_scopes_chunk_ids_to_each_source_document():
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        document = Path(directory) / "paper.pdf"
        document.write_text("PDF bytes")
        paper = Paper("P1", "A paper")

        def make_chunks(_paper, _pages):
            return [Chunk("P1:1:0", paper, "Shared attachment evidence.", ordinal=0)]

        with SQLiteStore(
            path,
            mode="write",
            embedding_id="hash:2",
            indexing_fingerprint=INDEXING_FINGERPRINT,
        ) as store:
            for source_key in ("A1", "A2"):
                index_document(
                    paper,
                    source_key,
                    1,
                    document,
                    extract_pages=lambda _path: [(1, "text", None)],
                    chunk_document=make_chunks,
                    embed_documents=lambda _texts: [[1.0, 0.0]],
                    store=store,
                )
            hits = store.search([1.0, 0.0], limit=5)

    assert {chunk.chunk_id for chunk, _score in hits} == {"A1:0", "A2:0"}


def test_one_fake_chat_runtime_serves_all_llm_tasks():
    paper = Paper("P1", "A paper")
    chunk = Chunk("P1:1:0", paper, "Exact supporting sentence.", page=1)
    hit = (chunk, 0.4)

    def complete(messages, *, json_mode=False):
        prompt = messages[0]["content"]
        if "Rerank" in prompt:
            return '{"items":[{"candidate":0,"support":0.9}]}'
        if "Select exact" in prompt:
            return (
                '{"supports_claim":true,"quote":"Exact supporting sentence.",'
                '"rationale":"Direct support."}'
            )
        return "Direct support."

    assert llm_tasks.rerank("claim", [hit], complete=complete) == [(chunk, 0.9)]
    assert llm_tasks.select_evidence(
        "claim",
        hit,
        "Before. Exact supporting sentence. After.",
        complete=complete,
    ) == ("Exact supporting sentence.", "Direct support.")
    assert llm_tasks.explain("claim", chunk.text, complete=complete) == "Direct support."


def test_search_with_llm_off_never_constructs_a_chat_provider(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        _seed_store(path)
        settings = _hash_settings(str(path))
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_chat",
            lambda _settings: (_ for _ in ()).throw(AssertionError("chat must not be built")),
        )

        result = bootstrap.run_search(settings, "Road quality affects trade")

    assert result["items"]


def test_read_commands_leave_existing_database_bytes_unchanged_and_skip_zotero(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        draft = Path(directory) / "draft.tex"
        draft.write_text("Road quality changes trade outcomes.", encoding="utf-8")
        _seed_store(path)
        before = path.read_bytes()
        settings = _hash_settings(str(path))
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_zotero",
            lambda _settings: (_ for _ in ()).throw(AssertionError("Zotero must not be built")),
        )

        search = bootstrap.run_search(settings, "Road quality affects trade")
        check = bootstrap.run_check(settings, draft)

        assert path.read_bytes() == before

    assert search["items"]
    assert check["claims"]


def test_sync_never_constructs_a_chat_provider(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        settings = _hash_settings(str(path))
        monkeypatch.setattr("bibliograph.bootstrap.build_zotero", lambda _config: object())
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_chat",
            lambda _settings: (_ for _ in ()).throw(AssertionError("chat must not be built")),
        )
        monkeypatch.setattr(
            "bibliograph.bootstrap.sync_library",
            lambda *_args, **_kwargs: {"indexed": [], "collection": "Collection"},
        )

        result = bootstrap.run_sync(settings)

    assert result["collection"] == "Collection"


def test_optional_llm_initialization_failure_uses_configured_fallbacks(monkeypatch):
    settings = _hash_settings("unused.db")
    settings["llm"] = {
        "provider": "ollama",
        "model": "missing",
        "mode": "optional",
        "stages": ["rerank", "evidence", "rationale"],
    }
    paper = Paper("P1", "A paper")
    hit = (Chunk("P1:1:0", paper, "Supporting text."), 0.4)
    monkeypatch.setattr(
        "bibliograph.bootstrap.build_chat",
        lambda _settings: (_ for _ in ()).throw(RuntimeError("LLM unavailable")),
    )

    tools = bootstrap._llm_tools(settings)

    assert set(tools) == {"rerank", "select_evidence", "explain"}
    assert tools["rerank"]("claim", [hit]) == [hit]
    assert tools["select_evidence"]("claim", hit, hit[0].text) is None
    assert tools["explain"]("claim", hit[0].text) == (
        "Retrieved evidence overlaps with the draft passage; verify the source before citing."
    )


def test_required_llm_initialization_failure_propagates(monkeypatch):
    settings = _hash_settings("unused.db")
    settings["llm"] = {
        "provider": "ollama",
        "model": "missing",
        "mode": "required",
        "stages": ["rerank"],
    }
    monkeypatch.setattr(
        "bibliograph.bootstrap.build_chat",
        lambda _settings: (_ for _ in ()).throw(RuntimeError("LLM unavailable")),
    )

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        bootstrap._llm_tools(settings)


def test_empty_llm_stages_do_not_construct_a_chat_provider(monkeypatch):
    settings = _hash_settings("unused.db")
    settings["llm"] = {
        "provider": "ollama",
        "model": "unused",
        "mode": "optional",
        "stages": [],
    }
    monkeypatch.setattr(
        "bibliograph.bootstrap.build_chat",
        lambda _settings: (_ for _ in ()).throw(AssertionError("chat must not be built")),
    )

    assert bootstrap._llm_tools(settings) == {}


def test_status_without_probe_never_constructs_a_provider(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        _seed_store(path)
        settings = _hash_settings(str(path))
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_embedding",
            lambda _settings: (_ for _ in ()).throw(AssertionError("provider must not be built")),
        )
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_chat",
            lambda _settings: (_ for _ in ()).throw(AssertionError("provider must not be built")),
        )
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_zotero",
            lambda _settings: (_ for _ in ()).throw(AssertionError("Zotero must not be built")),
        )

        result = bootstrap.run_status(settings, probe=False)

    assert result["chunks"] == 1


def test_status_probe_keeps_database_status_and_reports_all_failed_capabilities(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        _seed_store(path)
        settings = _hash_settings(str(path))
        settings["llm"]["mode"] = "optional"
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_embedding",
            lambda _settings: (_ for _ in ()).throw(RuntimeError("embedding offline")),
        )
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_chat",
            lambda _settings: (_ for _ in ()).throw(RuntimeError("LLM offline")),
        )
        monkeypatch.setattr(
            "bibliograph.bootstrap.build_zotero",
            lambda _settings: (_ for _ in ()).throw(RuntimeError("Zotero offline")),
        )

        result = bootstrap.run_status(settings, probe=True)

    probes = {probe["capability"]: probe for probe in result["probes"]}
    assert result["chunks"] == 1
    assert probes == {
        "embedding": {
            "ok": False,
            "capability": "embedding",
            "error": "embedding offline",
        },
        "LLM": {"ok": False, "capability": "LLM", "error": "LLM offline"},
        "Zotero": {"ok": False, "capability": "Zotero", "error": "Zotero offline"},
    }


def test_failed_sync_rebuild_preserves_the_live_database(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        _seed_store(path)
        settings = _hash_settings(str(path))
        monkeypatch.setattr("bibliograph.bootstrap.build_zotero", lambda _config: object())
        monkeypatch.setattr(
            "bibliograph.bootstrap.sync_library",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("indexing failed")),
        )

        with pytest.raises(RuntimeError, match="indexing failed"):
            bootstrap.run_sync(settings, rebuild=True)

        with SQLiteStore(path, mode="read") as store:
            assert store.stats()["chunks"] == 1


def test_status_without_database_returns_a_non_mutating_empty_summary():
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "absent.db"
        result = status(path=path)

        assert not path.exists()
        assert result["schema_version"] == 0
