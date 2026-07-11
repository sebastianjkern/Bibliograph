from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from connections.reload_embeddings import (
    download_pdf_for_item,
    download_pdfs_for_collection,
    resolve_item_key,
)


class _Zotero:
    def __init__(self):
        self.downloads = 0

    def item(self, key):
        if key == "PARENT":
            return {"data": {"key": "PARENT", "itemType": "journalArticle"}}
        return {
            "data": {
                "key": "ATTACH",
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": "paper.pdf",
            }
        }

    def children(self, key):
        return [
            {
                "data": {
                    "key": "ATTACH",
                    "itemType": "attachment",
                    "contentType": "application/pdf",
                }
            }
        ]

    def file(self, key):
        self.downloads += 1
        return b"%PDF-test"

    def items(self, q):
        return [
            {
                "data": {
                    "key": "PARENT",
                    "itemType": "journalArticle",
                    "title": "A Study",
                    "DOI": "10/example",
                }
            }
        ]

    def everything(self, request):
        return request


def test_download_pdf_for_item_skips_existing_attachment(monkeypatch):
    zotero = _Zotero()
    monkeypatch.setattr("connections.reload_embeddings.load_zotero_client", lambda: zotero)
    with TemporaryDirectory(dir=".") as directory:
        output = Path(directory)
        (output / "paper-ATTACH.pdf").write_bytes(b"%PDF-existing")

        result = download_pdf_for_item("PARENT", directory)

        assert result.downloaded is False
        assert zotero.downloads == 0


def test_download_pdf_for_item_downloads_one_missing_attachment(monkeypatch):
    zotero = _Zotero()
    monkeypatch.setattr("connections.reload_embeddings.load_zotero_client", lambda: zotero)
    with TemporaryDirectory(dir=".") as directory:
        result = download_pdf_for_item("PARENT", directory)

        assert result.downloaded is True
        assert Path(result.path).is_file()
        assert zotero.downloads == 1


def test_resolve_item_key_by_doi_or_title():
    zotero = _Zotero()
    assert resolve_item_key(zotero, doi="https://doi.org/10/example") == "PARENT"
    assert resolve_item_key(zotero, title="a study") == "PARENT"


def test_collection_downloads_are_disabled():
    with pytest.raises(RuntimeError, match="disabled"):
        download_pdfs_for_collection("COLLECTION")
