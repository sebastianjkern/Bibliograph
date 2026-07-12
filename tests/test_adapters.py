from pathlib import Path
from types import ModuleType

import pytest

from bibliograph.adapters.acquisition import (
    AcquisitionError,
    acquire_pdf,
    cache_strategy,
    download_remote_pdf,
    require_pdf,
    zotero_storage_strategy,
)
from bibliograph.adapters.pdf import chunk_document, extract_pages, is_pdf_file, locate_local_pdf
from bibliograph.adapters.zotero import (
    build_zotero,
    discover_collection,
    import_document_pdf,
    resolve_collection,
    zotero_storage_dirs,
)
from bibliograph.domain import Paper


def test_pdf_adapter_finds_valid_cache_files_and_creates_durable_chunks(tmp_path):
    cache = tmp_path / "pdfs"
    cache.mkdir()
    path = cache / "A1.pdf"
    path.write_bytes(b"%PDF-test")
    paper = Paper("P1", "A useful study", authors=("Lovelace",), year="2024")

    assert is_pdf_file(path)
    assert locate_local_pdf(cache, "A1", paper_key="P1") == path

    chunks = chunk_document(
        paper,
        [(3, "A short finding with useful evidence.", "Results")],
        max_words=20,
        overlap_words=2,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "P1:3:0"
    assert chunks[0].ordinal == 0
    assert chunks[0].section == "Results"
    assert chunks[0].text == "Section: Results\n\nA short finding with useful evidence."


def test_extract_pages_removes_repeated_margins_and_references(monkeypatch, tmp_path):
    class Page:
        rect = type("Rect", (), {"height": 100})()

        def __init__(self, blocks):
            self.blocks = blocks

        def get_text(self, kind):
            assert kind == "blocks"
            return self.blocks

    class Document:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self.pages)

        def __len__(self):
            return len(self.pages)

    fake_fitz = ModuleType("fitz")
    fake_fitz.open = lambda path: Document(
        [
            Page(
                [
                    (0, 10, 100, 20, "Journal Header\n1", 0, 0),
                    (0, 30, 100, 40, "1 Introduction", 1, 0),
                    (0, 45, 100, 60, "Climate systems vary over time.", 2, 0),
                ]
            ),
            Page(
                [
                    (0, 10, 100, 20, "Journal Header\n2", 0, 0),
                    (0, 30, 100, 40, "2 Results", 1, 0),
                    (0, 45, 100, 60, "The result is robust.", 2, 0),
                    (0, 70, 100, 80, "References", 3, 0),
                    (0, 82, 100, 95, "Ignored reference entry.", 4, 0),
                ]
            ),
        ]
    )
    monkeypatch.setitem(__import__("sys").modules, "fitz", fake_fitz)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-test")

    assert extract_pages(pdf) == [
        (1, "Climate systems vary over time.", "1 Introduction"),
        (2, "The result is robust.", "2 Results"),
    ]


class _Zotero:
    def everything(self, request):
        return request

    def collections(self):
        return [
            {"data": {"key": "COL1", "name": "Methods"}},
            {"data": {"key": "COL2", "name": "Results"}},
        ]

    def collection_items(self, key):
        assert key == "COL1"
        return [
            {
                "data": {
                    "key": "P1",
                    "itemType": "journalArticle",
                    "title": "A Study",
                    "date": "2024-05-01",
                    "DOI": "10/example",
                    "creators": [{"lastName": "Lovelace"}],
                }
            },
            {"data": {"key": "D1", "itemType": "dataset", "title": "Skip me"}},
        ]

    def children(self, key):
        assert key == "P1"
        return [
            {
                "data": {
                    "key": "A1",
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                    "version": 4,
                }
            },
            {
                "data": {
                    "key": "A2",
                    "itemType": "attachment",
                    "contentType": "text/html",
                }
            },
        ]


def test_zotero_adapter_normalizes_documents_and_imports_local_storage(tmp_path):
    cache = tmp_path / "cache"
    storage = tmp_path / "storage"
    local = storage / "A1" / "paper.pdf"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"%PDF-local")
    client = _Zotero()

    assert resolve_collection(client, "Methods") == "COL1"
    assert resolve_collection(client, "COL1") == "COL1"
    documents = discover_collection(
        client,
        "COL1",
        pdf_dir=cache,
        storage_dirs=(storage,),
    )

    assert len(documents) == 1
    document = documents[0]
    assert document["source_key"] == "A1"
    assert document["version"] == 4
    assert document["paper"] == Paper(
        "P1", "A Study", authors=("Lovelace",), year="2024", doi="10/example", collections=("COL1",)
    )
    assert document["path_candidates"] == (str(local),)

    imported = import_document_pdf(document, pdf_dir=cache, storage_dirs=(storage,))
    assert imported == cache / "A1.pdf"
    assert imported.read_bytes() == b"%PDF-local"


def test_build_zotero_uses_explicit_configuration(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, library_id, *, library_type, api_key):
            created.append((library_id, library_type, api_key))

    fake_pyzotero = ModuleType("pyzotero")
    fake_pyzotero.Zotero = FakeClient
    monkeypatch.setitem(__import__("sys").modules, "pyzotero", fake_pyzotero)

    assert isinstance(
        build_zotero({"library_id": "42", "library_type": "group", "api_key": "secret"}),
        FakeClient,
    )
    assert created == [("42", "group", "secret")]


def test_zotero_storage_dirs_prefers_an_explicit_configured_location(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()

    assert zotero_storage_dirs(storage) == (storage.resolve(),)


def test_acquisition_chain_keeps_failure_details_and_uses_later_source(tmp_path):
    valid = tmp_path / "valid.pdf"
    valid.write_bytes(b"%PDF-test")

    def unavailable(document):
        return None

    def broken(document):
        raise OSError("service unavailable")

    def success(document):
        return {"path": str(valid), "source": "test-source", "downloaded": True}

    result = acquire_pdf({"source_key": "A1"}, strategies=(unavailable, broken, success))

    assert result["path"] == str(valid)
    assert result["source"] == "test-source"
    assert result["downloaded"] is True
    assert result["attempts"] == (
        {"source": "unavailable", "error": "not found"},
        {"source": "broken", "error": "service unavailable"},
    )


def test_acquisition_chain_preserves_structured_manual_hints(tmp_path):
    valid = tmp_path / "valid.pdf"
    valid.write_bytes(b"%PDF-test")
    hint = {
        "url": "https://doi.org/10/example",
        "source": "doi.org",
        "reason": "Use legitimate publisher access.",
    }

    def needs_manual_recovery(_document):
        raise AcquisitionError("remote unavailable", hints=(hint,))

    def success(_document):
        return valid

    result = acquire_pdf(
        {"source_key": "A1"}, strategies=(needs_manual_recovery, success)
    )

    assert result["path"] == str(valid)
    assert result["attempts"] == (
        {
            "source": "needs_manual_recovery",
            "error": "remote unavailable",
            "hints": (hint,),
        },
    )

    with pytest.raises(AcquisitionError) as error:
        require_pdf({"source_key": "A1"}, strategies=(needs_manual_recovery,))
    assert error.value.hints == (hint,)


def test_cache_and_zotero_storage_strategies_are_composable(tmp_path):
    cache = tmp_path / "cache"
    storage = tmp_path / "storage"
    local = storage / "A1" / "paper.pdf"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"%PDF-local")
    document = {"source_key": "A1", "attachment_key": "A1"}

    result = acquire_pdf(
        document,
        strategies=(cache_strategy(cache), zotero_storage_strategy(cache, (storage,))),
    )

    assert result["source"] == "zotero-storage"
    assert Path(result["path"]) == cache / "A1.pdf"
    assert result["attempts"] == ({"source": "cache", "error": "not found"},)


def test_invalid_collection_name_is_contextual():
    with pytest.raises(ValueError, match="Collection not found"):
        resolve_collection(_Zotero(), "unknown")


def test_remote_acquisition_returns_a_simple_result_mapping(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "bibliograph.adapters.acquisition._retrieve_with_doidownloader",
        lambda doi, email: (b"%PDF-test", f"https://repository.test/{doi}.pdf"),
    )

    result = download_remote_pdf("10/example", tmp_path, title="A paper", item_key="P1")

    assert result["downloaded"] is True
    assert result["source"] == "doidownloader"
    assert Path(result["path"]).read_bytes() == b"%PDF-test"


def test_remote_acquisition_exposes_manual_hints_after_all_strategies_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "bibliograph.adapters.acquisition._retrieve_with_doidownloader",
        lambda *_args: (_ for _ in ()).throw(OSError("offline")),
    )
    monkeypatch.setattr(
        "bibliograph.adapters.acquisition._retrieve_with_playwright",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("browser offline")),
    )

    with pytest.raises(AcquisitionError, match="DOIDownloader") as error:
        download_remote_pdf("10/example", tmp_path)

    assert error.value.hints[0]["url"] == "https://doi.org/10/example"


def test_remote_acquisition_passes_browser_options_by_dependency_injection(monkeypatch, tmp_path):
    monkeypatch.setenv("BIBLIOGRAPH_PLAYWRIGHT_PROFILE", "ignored-by-adapter")
    monkeypatch.setenv("BIBLIOGRAPH_PLAYWRIGHT_HEADLESS", "true")
    monkeypatch.setattr(
        "bibliograph.adapters.acquisition._retrieve_with_doidownloader",
        lambda *_args: (_ for _ in ()).throw(OSError("offline")),
    )
    captured = {}

    def browser(doi, *, profile, headless):
        captured.update({"doi": doi, "profile": profile, "headless": headless})
        return b"%PDF-test", "https://repository.test/paper.pdf"

    monkeypatch.setattr("bibliograph.adapters.acquisition._retrieve_with_playwright", browser)

    result = download_remote_pdf(
        "10/example",
        tmp_path,
        playwright_profile="configured-profile",
        playwright_headless=False,
    )

    assert result["source"] == "playwright"
    assert captured == {
        "doi": "10/example",
        "profile": "configured-profile",
        "headless": False,
    }
