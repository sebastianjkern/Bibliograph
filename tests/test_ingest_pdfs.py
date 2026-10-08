from pathlib import Path

import pytest

from bibliograph.commands.ingest_pdfs import ingest_pdfs


class FakeStore:
    def __init__(self):
        self.indexed = []
        self.unchanged = set()

    def needs_document(self, source_key, _version, _path):
        return source_key not in self.unchanged

    def index_pdf(self, paper, source_key, _version, path, *, extract_pages):
        self.indexed.append((paper, source_key, Path(path)))
        assert extract_pages is not None
        return {"chunks": 2}

    def stats(self):
        return {"documents": len(self.indexed)}


def test_ingest_pdfs_indexes_valid_files_and_skips_invalid(tmp_path):
    pdf = tmp_path / "A paper.pdf"
    pdf.write_bytes(b"%PDF-1.7 test")
    (tmp_path / "not really.pdf").write_text("not a PDF", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.pdf").write_bytes(b"%PDF-1.7 nested")
    store = FakeStore()

    result = ingest_pdfs(
        tmp_path, store=store, extract_pages=lambda _path: [], recursive=True
    )

    assert result["discovered"] == 3
    assert len(result["indexed"]) == 2
    assert len(result["invalid"]) == 1
    assert store.indexed[0][0].title == "A paper"
    assert store.indexed[0][0].zotero_key.startswith("LOCAL-")


def test_ingest_pdfs_can_be_non_recursive_and_is_idempotent(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7 test")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.pdf").write_bytes(b"%PDF-1.7 nested")
    store = FakeStore()

    first = ingest_pdfs(
        tmp_path, store=store, extract_pages=lambda _path: [], recursive=False
    )
    store.unchanged.add(first["indexed"][0]["source_key"])
    second = ingest_pdfs(
        tmp_path, store=store, extract_pages=lambda _path: [], recursive=False
    )

    assert first["discovered"] == 1
    assert len(second["unchanged"]) == 1
    assert len(store.indexed) == 1


def test_ingest_pdfs_requires_a_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        ingest_pdfs(tmp_path / "missing", store=FakeStore(), extract_pages=lambda _path: [])
