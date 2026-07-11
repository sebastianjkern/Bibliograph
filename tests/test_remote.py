from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace

import pytest

from bibliograph.remote import (
    PyDOIResolver,
    RemoteCandidate,
    RemoteDownloadError,
    ResolverChain,
    download_remote_pdf,
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


def test_remote_download_uses_doidownloader_by_default(monkeypatch):
    monkeypatch.setattr(
        "bibliograph.remote._retrieve_with_doidownloader",
        lambda doi, email: (b"%PDF-doidownloader", "https://publisher.example/paper.pdf"),
    )
    with TemporaryDirectory(dir=".") as directory:
        result = download_remote_pdf("10/example", directory, title="Paper", item_key="P1")

        assert Path(result.path).read_bytes() == b"%PDF-doidownloader"
        assert result.resolver == "doidownloader"
        assert result.source_url == "https://publisher.example/paper.pdf"


def test_doidownloader_adapter_reads_pdf_content_from_lookup_result(monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    package = ModuleType("doidownloader")
    package.DOIDownloader = lambda email_address=None: Client()
    implementation = ModuleType("doidownloader.doidownloader")

    async def retrieve_best_fulltext(doi, client):
        return ("https://publisher.example/paper.pdf", None, 200, b"%PDF-real", "pdf")

    implementation.retrieve_best_fulltext = retrieve_best_fulltext
    monkeypatch.setitem(__import__("sys").modules, "doidownloader", package)
    monkeypatch.setitem(__import__("sys").modules, "doidownloader.doidownloader", implementation)

    from bibliograph.remote import _retrieve_with_doidownloader

    assert _retrieve_with_doidownloader("10/example", None) == (
        b"%PDF-real",
        "https://publisher.example/paper.pdf",
    )


def test_missing_doidownloader_explains_python_version_requirement(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "doidownloader", None)

    from bibliograph.remote import _retrieve_with_doidownloader

    with pytest.raises(ImportError, match="Python 3.12"):
        _retrieve_with_doidownloader("10/example", None)


def test_remote_download_exposes_manual_pydoi_hints(monkeypatch):
    monkeypatch.setattr(
        "bibliograph.remote._retrieve_with_doidownloader",
        lambda doi, email: (_ for _ in ()).throw(FileNotFoundError("not found")),
    )
    monkeypatch.setattr(
        "bibliograph.remote._retrieve_with_playwright",
        lambda doi: (_ for _ in ()).throw(FileNotFoundError("browser not found")),
    )
    hint = SimpleNamespace(
        url="https://publisher.example/article",
        source="pydoi",
        reason="Download manually",
    )
    monkeypatch.setattr("bibliograph.remote.PyDOIResolver.manual_hints", lambda self, doi: [hint])

    with TemporaryDirectory(dir=".") as directory:
        with pytest.raises(RemoteDownloadError) as error:
            download_remote_pdf("10/example", directory)

    assert error.value.hints == (hint,)


def test_browser_pdf_candidates_prioritize_pdf_links():
    from bibliograph.remote import _browser_pdf_candidates

    candidates = _browser_pdf_candidates(
        "https://publisher.example/article/1",
        [
            {"href": "/article/1", "text": "Article"},
            {"href": "/article/1/download", "text": "Download PDF"},
            {"href": "/article/1/supplement.pdf", "text": "Supplement"},
            {"href": "http://publisher.example/article/1.pdf", "text": "PDF"},
        ],
    )

    assert candidates[:2] == [
        "https://publisher.example/article/1/download",
        "https://publisher.example/article/1/supplement.pdf",
    ]


def test_pydoi_resolver_returns_https_manual_landing_pages(monkeypatch):
    fake_pydoi = SimpleNamespace(
        get_url=lambda doi, allow_multi: [
            "https://publisher.example/article",
            "http://insecure.example/article",
        ]
    )
    monkeypatch.setitem(__import__("sys").modules, "pydoi", fake_pydoi)

    candidates = PyDOIResolver().resolve("10/Example")

    assert [candidate.url for candidate in candidates] == [
        "https://publisher.example/article"
    ]
