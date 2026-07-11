from pathlib import Path
from tempfile import TemporaryDirectory

from bibliograph.embeddings import HashEmbedder
from bibliograph.store import SQLiteIndex
from bibliograph.zotero_sync import (
    is_pdf_attachment,
    is_remote_downloadable_item,
    sync_collection,
    zotero_storage_dirs,
)


class _Zotero:
    def everything(self, request):
        return request

    def collection_items(self, key):
        return [{"data": {"key": "P1", "itemType": "journalArticle", "title": "Study"}}]

    def children(self, key):
        return [
            {
                "data": {
                    "key": "A1",
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                    "version": 3,
                }
            }
        ]


def test_sync_reports_missing_pdf_without_downloading():
    index = SQLiteIndex(":memory:")
    with TemporaryDirectory(dir=".") as directory:
        report = sync_collection(_Zotero(), "COL1", directory, index, HashEmbedder())
    assert report.missing_local_pdf == ["A1"]
    assert report.missing_papers[0].title == "Study"
    assert report.missing_papers[0].attachment_keys == ("A1",)
    assert report.indexed == []


def test_sync_indexes_local_pdf_once_and_then_marks_unchanged():
    index = SQLiteIndex(":memory:")
    calls = []

    def fake_indexer(index, embedder, paper, path):
        calls.append((paper.zotero_key, Path(path).name))
        return 1

    with TemporaryDirectory(dir=".") as directory:
        pdf = Path(directory) / "study-A1.pdf"
        pdf.write_bytes(b"%PDF-test")
        first = sync_collection(_Zotero(), "COL1", directory, index, HashEmbedder(), fake_indexer)
        second = sync_collection(_Zotero(), "COL1", directory, index, HashEmbedder(), fake_indexer)

    assert first.indexed == ["A1"]
    assert second.unchanged == ["A1"]
    assert calls == [("P1", "study-A1.pdf")]


def test_sync_treats_non_pdf_content_with_pdf_suffix_as_missing():
    index = SQLiteIndex(":memory:")
    with TemporaryDirectory(dir=".") as directory:
        Path(directory, "study-A1.pdf").write_bytes(b"<html>not a PDF</html>")

        report = sync_collection(_Zotero(), "COL1", directory, index, HashEmbedder())

    assert report.indexed == []
    assert report.missing_local_pdf == ["A1"]


def test_zotero_attachment_filter_uses_content_type():
    assert is_pdf_attachment({"itemType": "attachment", "contentType": "application/pdf"})
    assert not is_pdf_attachment(
        {"itemType": "attachment", "contentType": "text/html", "filename": "page.pdf"}
    )


def test_remote_download_filter_excludes_non_paper_item_types():
    assert is_remote_downloadable_item("journalArticle")
    assert not is_remote_downloadable_item("dataset")
    assert not is_remote_downloadable_item("webpage")


def test_sync_skips_unsupported_zotero_items_before_pdf_lookup():
    class NonPaperZotero:
        def everything(self, request):
            return request

        def collection_items(self, key):
            return [{"data": {"key": "D1", "itemType": "dataset", "DOI": "10/dataset"}}]

        def children(self, key):
            raise AssertionError("unsupported items must not be inspected for attachments")

    index = SQLiteIndex(":memory:")
    with TemporaryDirectory(dir=".") as directory:
        report = sync_collection(NonPaperZotero(), "COL1", directory, index, HashEmbedder())

    assert report.missing_papers == []
    assert report.missing_local_pdf == []


def test_sync_imports_pdf_from_local_zotero_storage_before_reporting_missing():
    index = SQLiteIndex(":memory:")
    calls = []

    def fake_indexer(index, embedder, paper, path):
        calls.append(Path(path).name)
        return 1

    with TemporaryDirectory(dir=".") as directory:
        storage = Path(directory) / "storage"
        (storage / "A1").mkdir(parents=True)
        (storage / "A1" / "paper.pdf").write_bytes(b"%PDF-from-zotero")
        cache = Path(directory) / "pdfs"

        report = sync_collection(
            _Zotero(),
            "COL1",
            cache,
            index,
            HashEmbedder(),
            fake_indexer,
            zotero_storage_dir=storage,
        )

        assert report.imported == ["A1"]
        assert report.indexed == ["A1"]
        assert report.missing_papers == []
        assert (cache / "A1.pdf").read_bytes() == b"%PDF-from-zotero"
        assert calls == ["A1.pdf"]


def test_zotero_storage_directory_can_be_discovered_from_environment(monkeypatch):
    with TemporaryDirectory(dir=".") as directory:
        storage = Path(directory) / "storage"
        storage.mkdir()
        monkeypatch.setenv("ZOTERO_STORAGE_DIR", str(storage))

        assert zotero_storage_dirs() == (storage.resolve(),)
