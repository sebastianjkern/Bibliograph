from contextlib import nullcontext

from bibliograph.commands import sync as sync_command
from bibliograph.commands.indexing import index_available_document
from bibliograph.domain import Paper


def test_force_reindex_bypasses_unchanged_check():
    calls = []

    class Store:
        def needs_document(self, *_args):
            return False

        def index_document(self, document, *, extract_pages):
            calls.append(document.source_key)
            return {"chunks": 2}

    result = index_available_document(
        Store(),
        Paper("P1", "Road study"),
        "P1",
        "1",
        "paper.pdf",
        extract_pages=lambda _path: [],
        force_reindex=True,
    )

    assert calls == ["P1"]
    assert result["state"] == "indexed"
    assert result["chunks"] == 2


def test_sync_forwards_indexing_phases_to_live_progress(monkeypatch):
    messages = []
    paper = Paper("P1", "Road study")
    document = {"source_key": "P1", "paper": paper, "version": "1"}

    class FakeProgress:
        console = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def add_task(self, *_args, **_kwargs):
            return 1

        def advance(self, *_args, **_kwargs):
            return None

        def update(self, *_args, **_kwargs):
            return None

    class FakeStages:
        def __init__(self, _progress, _task_id):
            pass

        def update(self, message):
            messages.append(message)

        def complete(self, message):
            messages.append(message)

    class FakeConsole:
        def __init__(self, **_kwargs):
            pass

        def status(self, *_args, **_kwargs):
            return nullcontext()

    class FakeStore:
        def needs_document(self, *_args):
            return True

        def index_document(self, _document, *, extract_pages, progress=None):
            assert extract_pages is not None
            progress("Extracting PDF pages")
            progress("Classifying evidence · 1/1 chunks")
            progress("Embedding and indexing 1 chunks")
            return {"chunks": 1}

        def record_document_failure(self, *_args, **_kwargs):
            raise AssertionError("unexpected indexing failure")

        def mark_sync_complete(self):
            return None

        def stats(self):
            return {}

    monkeypatch.setattr(sync_command, "Console", FakeConsole)
    monkeypatch.setattr(sync_command, "Progress", lambda *_args, **_kwargs: FakeProgress())
    monkeypatch.setattr(sync_command, "StageProgress", FakeStages)
    monkeypatch.setattr(sync_command, "resolve_collection", lambda *_args: "COLL")
    monkeypatch.setattr(sync_command, "discover_collection", lambda *_args, **_kwargs: [document])
    monkeypatch.setattr(
        sync_command,
        "acquire_pdf",
        lambda *_args, **_kwargs: {"path": "paper.pdf", "downloaded": False},
    )

    sync_command.sync_library(
        {"collection": "Collection", "pdf_dir": "pdfs"},
        store=FakeStore(),
        zotero=object(),
        strategies=(),
        extract_pages=lambda _path: [],
    )

    assert any("Extracting PDF pages" in message for message in messages)
    assert any("Classifying evidence · 1/1 chunks" in message for message in messages)
    assert any("Embedding and indexing 1 chunks" in message for message in messages)
