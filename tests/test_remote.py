from pathlib import Path
from tempfile import TemporaryDirectory

from bibliograph.remote import download_unpaywall_pdf


class _Headers:
    def get_content_type(self):
        return "application/pdf"


class _Response:
    headers = _Headers()

    def read(self):
        return b"%PDF-remote"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_unpaywall_download_uses_remote_pdf_and_skips_existing(monkeypatch):
    monkeypatch.setattr(
        "bibliograph.remote._get_json",
        lambda url, params: {"oa_locations": [{"url_for_pdf": "https://repo.example/paper.pdf"}]},
    )
    monkeypatch.setattr("bibliograph.remote.urlopen", lambda request, timeout: _Response())
    with TemporaryDirectory(dir=".") as directory:
        first = download_unpaywall_pdf(
            "10/example", directory, email="researcher@example.edu", title="Paper", item_key="P1"
        )
        second = download_unpaywall_pdf(
            "10/example", directory, email="researcher@example.edu", title="Paper", item_key="P1"
        )

        assert Path(first.path).read_bytes() == b"%PDF-remote"
        assert first.downloaded is True
        assert second.downloaded is False
