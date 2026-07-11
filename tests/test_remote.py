from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bibliograph.remote import (
    RemoteCandidate,
    ResolverChain,
    download_unpaywall_pdf,
)


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
        lambda url, params: {
            "is_oa": True,
            "oa_locations": [{"url_for_pdf": "https://repo.example/paper.pdf"}],
        },
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


def test_resolver_chain_rejects_non_https_candidates():
    class Resolver:
        name = "test"

        def resolve(self, doi):
            return [RemoteCandidate("http://example.com/paper.pdf", self.name, "test source")]

    with pytest.raises(ValueError, match="non-HTTPS"):
        ResolverChain([Resolver()]).resolve("10/example")


def test_resolver_chain_continues_after_a_failed_resolver():
    class FailedResolver:
        name = "failed"

        def resolve(self, doi):
            raise OSError("service unavailable")

    class WorkingResolver:
        name = "working"

        def resolve(self, doi):
            return [RemoteCandidate("https://repo.example/paper.pdf", self.name, "OA repository")]

    candidates = ResolverChain([FailedResolver(), WorkingResolver()]).resolve("10/example")
    assert candidates[0].resolver == "working"
