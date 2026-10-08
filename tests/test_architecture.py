from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep

import pytest

from bibliograph import bootstrap
from bibliograph.adapters.ikarus import IkarusBackend
from bibliograph.adapters.sqlite import SQLiteStore
from bibliograph.commands.status import status
from bibliograph.domain import Chunk, Paper
from bibliograph.pipeline import llm_tasks
from bibliograph.pipeline.indexing import INDEXING_FINGERPRINT, index_document
from bibliograph.providers.registry import build_embedding


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


def _seed_ikarus_store(path: Path) -> None:
    paper = Paper("P1", "A paper", ("Ada",), "2024")
    settings = _hash_settings(str(path))
    with IkarusBackend(path, mode="write", embedding=build_embedding(settings)) as backend:
        backend.index_pdf(
            paper,
            "A1",
            1,
            __file__,
            extract_pages=lambda _path: [
                (1, "Road quality changes trade outcomes.", "Results")
            ],
        )


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
            hits = store.search(
                [1.0, 0.0],
                limit=5,
                query_text="Shared attachment evidence",
            )

    assert {chunk.chunk_id for chunk, _score in hits} == {"A1:0", "A2:0"}
    assert all(score.semantic == 1.0 for _chunk, score in hits)
    assert all(score.lexical == 1.0 for _chunk, score in hits)


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


def test_rerank_scores_direct_entailment_above_topical_objective():
    paper = Paper("P1", "A paper", ("Author",), "2010")
    objective = Chunk(
        "P1:1:0",
        paper,
        "Our objective is to develop a gravity model for inter-city trade.",
        page=7,
        section="Introduction",
    )
    result = Chunk(
        "P1:2:0",
        paper,
        "We find that road improvements increased overland trade by 18 percent.",
        page=14,
        section="Results",
    )
    prompts = []

    def complete(messages, *, json_mode=False):
        prompts.append(messages[0]["content"])
        return (
            '{"items":['
            '{"candidate":0,"support":0.82,"relation":"neutral",'
            '"evidence_role":"objective","claim_specificity":0.7,'
            '"result_presence":0.0,"primary_source":1.0,"background_penalty":0.8,'
            '"secondary_citation_penalty":0.0,"unsupported_inference_penalty":0.8},'
            '{"candidate":1,"support":0.88,"relation":"supports",'
            '"evidence_role":"result","claim_specificity":0.95,'
            '"result_presence":1.0,"primary_source":1.0,"background_penalty":0.0,'
            '"secondary_citation_penalty":0.0,"unsupported_inference_penalty":0.0}'
            ']}'
        )

    ranked = llm_tasks.rerank(
        "Road network improvements increased overland trade in Sub-Saharan Africa.",
        [(objective, 0.9), (result, 0.7)],
        complete=complete,
    )

    assert [chunk.chunk_id for chunk, _score in ranked] == ["P1:2:0", "P1:1:0"]
    assert ranked[0][1] > 0.7
    assert ranked[1][1] <= 0.35
    assert "direct factual entailment" in prompts[0]
    assert "directly_entails_claim" in prompts[0]
    assert "Our objective is to develop a gravity model" in prompts[0]


def test_rerank_caps_publisher_metadata_even_when_llm_overrates_it():
    paper = Paper("P1", "A paper", ("Author",), "2010")
    metadata = Chunk(
        "P1:1:0",
        paper,
        "Published by Blackwell Publishing Ltd, 9600 Garsington Road, Oxford OX4 2DQ, UK.",
        page=2,
    )

    def complete(_messages, *, json_mode=False):
        return (
            '{"items":[{"candidate":0,"support":0.9,"relation":"supports",'
            '"evidence_role":"other","claim_specificity":0.9,'
            '"result_presence":0.8,"primary_source":1.0,'
            '"unsupported_inference_penalty":0.0}]}'
        )

    ranked = llm_tasks.rerank("Roads increase trade.", [(metadata, 0.8)], complete=complete)

    assert ranked == [(metadata, 0.05)]


def test_rerank_calculates_boolean_subscores_without_llm_numeric_score():
    paper = Paper("P1", "A paper", ("Author",), "2010")
    chunk = Chunk("P1:1:0", paper, "We find that road quality increases trade.")
    seen_prompt = []

    def complete(messages, *, json_mode=False):
        seen_prompt.append(messages[0]["content"])
        return (
            '{"items":[{"candidate":0,"relation":"supports",'
            '"evidence_role":"result","directly_entails_claim":true,'
            '"contains_claim_specific_result":true,"contains_explicit_finding":true,'
            '"requires_unsupported_inference":false,"source_type":"primary"}]}'
        )

    ranked = llm_tasks.rerank("Road quality increases trade.", [(chunk, 0.2)], complete=complete)

    assert ranked == [(chunk, 1.0)]
    assert "JSON booleans" in seen_prompt[0]
    assert '"support"' not in seen_prompt[0]


def test_select_evidence_rejects_publisher_address_quotes():
    paper = Paper("P1", "A paper", ("Author",), "2010")
    chunk = Chunk("P1:1:0", paper, "Publisher information.")

    def complete(_messages, *, json_mode=False):
        return (
            '{"supports_claim":true,"quote":"Published by Blackwell Publishing Ltd, '
            '9600 Garsington Road, Oxford OX4 2DQ, UK.",'
            '"rationale":"Direct support."}'
        )

    assert (
        llm_tasks.select_evidence(
            "Road infrastructure affects trade.",
            (chunk, 0.05),
            "Published by Blackwell Publishing Ltd, 9600 Garsington Road, Oxford OX4 2DQ, UK.",
            complete=complete,
        )
        is None
    )


def test_rerank_chunks_all_candidates_into_prompts_of_at_most_ten():
    paper = Paper("P1", "A paper")
    hits = [
        (Chunk(f"P1:{index}:0", paper, f"Evidence {index}."), 0.5)
        for index in range(11)
    ]
    calls = []

    def complete(messages, *, json_mode=False):
        candidates = messages[1]["content"].split("CANDIDATES:\n", 1)[1]
        calls.append(candidates.count("["))
        return '{"items": []}'

    result = llm_tasks.rerank("claim", hits, complete=complete)

    assert calls == [10, 1]
    assert result == hits


def test_rerank_batches_run_concurrently_and_keep_batch_order():
    paper = Paper("P1", "A paper")
    hits = [
        (Chunk(f"P1:{index}:0", paper, f"Evidence {index}."), 0.5)
        for index in range(11)
    ]
    lock = Lock()
    active = 0
    maximum_active = 0

    def complete(_messages, *, json_mode=False):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        sleep(0.02)
        with lock:
            active -= 1
        return '{"items": []}'

    result = llm_tasks.rerank("claim", hits, complete=complete)

    assert maximum_active > 1
    assert result == hits


def test_search_with_llm_off_never_constructs_a_chat_provider(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        _seed_ikarus_store(path)
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
        _seed_ikarus_store(path)
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


def test_query_expansion_parses_typed_alternatives():
    from bibliograph.pipeline.llm_tasks import expand_query

    result = expand_query(
        "Road quality affects market access.",
        complete=lambda _messages, json_mode=False: '{"items": ['
        '{"type": "paraphrase", "text": "Transport infrastructure affects market access"},'
        '{"type": "not-a-type", "text": "Ignore this"},'
        '{"type": "causal", "text": "Better roads reduce travel costs and improve access"}'
        "]}",
    )

    assert result == [
        "Transport infrastructure affects market access",
        "Better roads reduce travel costs and improve access",
    ]


def test_query_expansion_reports_each_partial_prompt():
    updates = []

    llm_tasks.expand_query(
        "Road quality affects market access.",
        complete=lambda _messages, json_mode=False: '{"items": []}',
        progress=updates.append,
    )

    assert updates == [
        "Expanding queries · 1/4 complete",
        "Expanding queries · 2/4 complete",
        "Expanding queries · 3/4 complete",
        "Expanding queries · 4/4 complete",
    ]


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
