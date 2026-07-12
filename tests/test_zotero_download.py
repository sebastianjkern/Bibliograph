from pathlib import Path
from tempfile import TemporaryDirectory

from bibliograph.adapters.acquisition import (
    acquire_pdf,
    cache_strategy,
    zotero_api_strategy,
    zotero_storage_strategy,
)
from bibliograph.domain import Paper


class _Zotero:
    def __init__(self):
        self.downloads = 0

    def file(self, key):
        assert key == "ATTACH"
        self.downloads += 1
        return b"%PDF-test"


def _document():
    return {
        "paper": Paper("PARENT", "Study", doi="10/example"),
        "source_key": "ATTACH",
        "attachment_key": "ATTACH",
        "path_candidates": (),
    }


def test_acquisition_uses_cached_attachment_without_api_download():
    zotero = _Zotero()
    with TemporaryDirectory(dir=".") as directory:
        pdf_dir = Path(directory)
        (pdf_dir / "ATTACH.pdf").write_bytes(b"%PDF-existing")

        result = acquire_pdf(
            _document(),
            strategies=[cache_strategy(pdf_dir), zotero_api_strategy(zotero, pdf_dir)],
        )

    assert result["source"] == "cache"
    assert zotero.downloads == 0


def test_acquisition_downloads_one_missing_zotero_attachment():
    zotero = _Zotero()
    with TemporaryDirectory(dir=".") as directory:
        result = acquire_pdf(_document(), strategies=[zotero_api_strategy(zotero, directory)])

        assert result["downloaded"] is True
        assert Path(result["path"]).is_file()

    assert zotero.downloads == 1


def test_acquisition_imports_from_local_zotero_storage_before_api():
    zotero = _Zotero()
    with TemporaryDirectory(dir=".") as directory:
        root = Path(directory)
        storage = root / "storage"
        (storage / "ATTACH").mkdir(parents=True)
        (storage / "ATTACH" / "paper.pdf").write_bytes(b"%PDF-local")
        pdf_dir = root / "pdfs"

        result = acquire_pdf(
            _document(),
            strategies=[
                zotero_storage_strategy(pdf_dir, [storage]),
                zotero_api_strategy(zotero, pdf_dir),
            ],
        )

        assert Path(result["path"]).read_bytes() == b"%PDF-local"

    assert zotero.downloads == 0
