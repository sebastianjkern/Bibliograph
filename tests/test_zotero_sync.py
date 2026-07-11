from pathlib import Path
from tempfile import TemporaryDirectory

from bibliograph.embeddings import HashEmbedder
from bibliograph.store import SQLiteIndex
from bibliograph.zotero_sync import sync_collection


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
        pdf.write_bytes(b"pdf")
        first = sync_collection(_Zotero(), "COL1", directory, index, HashEmbedder(), fake_indexer)
        second = sync_collection(_Zotero(), "COL1", directory, index, HashEmbedder(), fake_indexer)

    assert first.indexed == ["A1"]
    assert second.unchanged == ["A1"]
    assert calls == [("P1", "study-A1.pdf")]
