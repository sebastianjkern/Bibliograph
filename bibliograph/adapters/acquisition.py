"""Composable PDF-acquisition strategies.

The command composition supplies an ordered sequence of callables.  Each
strategy receives a normalized document mapping and may return a path or a
small result mapping.  This keeps cache, Zotero, and remote retrieval choices
out of the indexing pipeline.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from bibliograph.adapters.pdf import is_pdf_file, locate_local_pdf
from bibliograph.adapters.zotero import find_in_zotero_storage, import_zotero_pdf
from bibliograph.domain import Paper
from bibliograph.logging_utils import get_logger

logger = get_logger("acquisition")

PdfStrategy = Callable[[Mapping[str, Any]], str | Path | Mapping[str, Any] | None]


class AcquisitionError(FileNotFoundError):
    """A failed legal acquisition attempt with manual fallback hints."""

    def __init__(self, message: str, *, hints: Sequence[Mapping[str, str]] = ()) -> None:
        self.hints = tuple(dict(hint) for hint in hints)
        super().__init__(message)


def acquire_pdf(
    document: Mapping[str, Any], *, strategies: Iterable[PdfStrategy]
) -> dict[str, Any]:
    """Run PDF acquisition strategies in order and return a diagnostic result.

    A failed strategy is recorded and the chain continues.  The caller can
    present ``attempts`` to a user rather than silently treating a failed PDF
    lookup as an empty paper.  A successful result always points to a file with
    a valid PDF signature.
    """
    attempts: list[dict[str, Any]] = []
    for strategy in strategies:
        name = _strategy_name(strategy)
        try:
            raw_result = strategy(document)
        except AcquisitionError as error:
            logger.warning("PDF acquisition strategy %s failed: %s", name, error)
            attempt: dict[str, Any] = {"source": name, "error": str(error)}
            if error.hints:
                attempt["hints"] = error.hints
            attempts.append(attempt)
            continue
        except Exception as error:  # strategy boundaries are deliberately isolated
            logger.warning("PDF acquisition strategy %s failed: %s", name, error)
            attempts.append({"source": name, "error": str(error)})
            continue
        result = _normalise_result(raw_result, name)
        if result is None:
            attempts.append({"source": name, "error": "not found"})
            continue
        path = Path(result["path"])
        if not is_pdf_file(path):
            message = f"returned invalid PDF path: {path}"
            logger.warning("PDF acquisition strategy %s %s", name, message)
            attempts.append({"source": name, "error": message})
            continue
        result["path"] = str(path)
        result.setdefault("source", name)
        result.setdefault("downloaded", False)
        result["attempts"] = tuple(attempts)
        return result
    return {"path": None, "source": None, "downloaded": False, "attempts": tuple(attempts)}


def resolve_pdf(
    document: Mapping[str, Any], *, strategies: Iterable[PdfStrategy]
) -> dict[str, Any]:
    """Compatibility-friendly spelling for an ordered PDF acquisition attempt."""
    return acquire_pdf(document, strategies=strategies)


def require_pdf(document: Mapping[str, Any], *, strategies: Iterable[PdfStrategy]) -> Path:
    """Acquire a PDF or raise a contextual error carrying each failed source."""
    result = acquire_pdf(document, strategies=strategies)
    if result["path"]:
        return Path(result["path"])
    source_key = document.get("source_key") or _paper_value(document, "zotero_key") or "document"
    details = "; ".join(
        f"{attempt['source']}: {attempt['error']}" for attempt in result["attempts"]
    )
    hints = tuple(
        hint
        for attempt in result["attempts"]
        for hint in attempt.get("hints", ())
        if isinstance(hint, Mapping)
    )
    if hints:
        raise AcquisitionError(
            f"No PDF acquired for {source_key}. {details}".rstrip(), hints=hints
        )
    raise FileNotFoundError(f"No PDF acquired for {source_key}. {details}".rstrip())


def cache_strategy(pdf_dir: str | Path) -> PdfStrategy:
    """Return a strategy that finds an already cached PDF without copying it."""
    directory = Path(pdf_dir)

    def find(document: Mapping[str, Any]) -> dict[str, Any] | None:
        path = locate_local_pdf(
            directory,
            str(document.get("source_key") or ""),
            paper_key=_paper_value(document, "zotero_key"),
            doi=_paper_value(document, "doi"),
            candidates=tuple(str(path) for path in document.get("path_candidates", ())),
        )
        if path is None:
            return None
        return {"path": str(path), "source": "cache", "downloaded": False}

    find.__name__ = "cache"
    return find


def zotero_storage_strategy(
    pdf_dir: str | Path, storage_dirs: Iterable[str | Path]
) -> PdfStrategy:
    """Return a strategy that imports an attachment from local Zotero storage."""
    directory = Path(pdf_dir)
    storage = tuple(Path(path) for path in storage_dirs)

    def import_from_storage(document: Mapping[str, Any]) -> dict[str, Any] | None:
        attachment_key = document.get("attachment_key")
        if not attachment_key:
            return None
        source = find_in_zotero_storage(storage, str(attachment_key))
        if source is None:
            return None
        path = import_zotero_pdf(source, directory, str(attachment_key))
        return {"path": str(path), "source": "zotero-storage", "downloaded": True}

    import_from_storage.__name__ = "zotero-storage"
    return import_from_storage


def zotero_api_strategy(client: Any, pdf_dir: str | Path) -> PdfStrategy:
    """Return a strategy that requests one configured Zotero attachment file."""
    directory = Path(pdf_dir)

    def download_attachment(document: Mapping[str, Any]) -> dict[str, Any] | None:
        attachment_key = document.get("attachment_key")
        if not attachment_key:
            return None
        content = client.file(str(attachment_key))
        if not isinstance(content, bytes) or not content.startswith(b"%PDF-"):
            raise ValueError(f"Zotero attachment {attachment_key} did not return a PDF")
        path = directory / f"{attachment_key}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": str(path), "source": "zotero-api", "downloaded": True}

    download_attachment.__name__ = "zotero-api"
    return download_attachment


def remote_strategy(
    download: Callable[..., Any], output_dir: str | Path
) -> PdfStrategy:
    """Adapt an injected legal remote downloader to the common strategy shape.

    The adapter does not import a particular remote implementation.  A caller
    can pass the existing legal downloader, a test fake, or a future provider
    without changing indexing code.
    """
    directory = Path(output_dir)

    def download_remote(document: Mapping[str, Any]) -> Any:
        doi = _paper_value(document, "doi")
        if not doi:
            return None
        return download(
            doi,
            directory,
            title=_paper_value(document, "title"),
            item_key=str(document.get("source_key") or ""),
        )

    download_remote.__name__ = "remote"
    return download_remote


def download_remote_pdf(
    doi: str,
    output_dir: str | Path,
    *,
    email: str | None = None,
    openalex_api_key: str | None = None,
    playwright_profile: str | Path | None = None,
    playwright_headless: bool = True,
    title: str | None = None,
    item_key: str | None = None,
) -> dict[str, Any]:
    """Retrieve one legally accessible PDF through the configured fallbacks.

    The implementation is intentionally an adapter function rather than a
    resolver hierarchy.  A caller may still inject it into ``remote_strategy``
    or replace it entirely in a profile-specific composition root.
    """
    if not isinstance(playwright_headless, bool):
        raise ValueError("playwright_headless must be a boolean")
    normalized_doi = _normalize_doi(doi)
    key = item_key or _safe_key(normalized_doi)
    path = Path(output_dir) / f"{_safe_filename(title or normalized_doi)}-{key}.pdf"
    if is_pdf_file(path):
        return {"path": str(path), "downloaded": False, "source": "cache"}

    errors: list[str] = []
    try:
        content, source_url = _retrieve_with_doidownloader(normalized_doi, email)
        source = "doidownloader"
    except (ImportError, OSError, ValueError) as error:
        errors.append(f"DOIDownloader: {error}")
        try:
            content, source_url = _retrieve_with_playwright(
                normalized_doi,
                profile=playwright_profile,
                headless=playwright_headless,
            )
            source = "playwright"
        except (ImportError, OSError, ValueError) as browser_error:
            errors.append(f"browser: {browser_error}")
            hints = [
                {
                    "url": f"https://doi.org/{normalized_doi}",
                    "source": "doi.org",
                    "reason": "Open the DOI landing page and use legitimate publisher access.",
                }
            ]
            if openalex_api_key:
                hints.append(
                    {
                        "url": f"https://api.openalex.org/works/https://doi.org/{normalized_doi}",
                        "source": "openalex",
                        "reason": "Inspect the OpenAlex record for an open-access location.",
                    }
                )
            raise AcquisitionError("; ".join(errors), hints=hints) from browser_error

    if not content.startswith(b"%PDF-"):
        raise AcquisitionError(f"Remote source did not return a PDF for {normalized_doi}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": str(path),
        "downloaded": True,
        "source": source,
        "source_url": source_url,
    }


def _normalise_result(raw_result: Any, fallback_source: str) -> dict[str, Any] | None:
    if raw_result is None:
        return None
    if isinstance(raw_result, (str, Path)):
        return {"path": str(raw_result), "source": fallback_source}
    if isinstance(raw_result, Mapping):
        path = raw_result.get("path")
        if not path:
            return None
        return dict(raw_result)
    path = getattr(raw_result, "path", None)
    if not path:
        return None
    result = {"path": str(path), "source": getattr(raw_result, "resolver", fallback_source)}
    for key in ("downloaded", "source_url", "legal_basis"):
        value = getattr(raw_result, key, None)
        if value is not None:
            result[key] = value
    return result


def _strategy_name(strategy: PdfStrategy) -> str:
    return getattr(strategy, "__name__", strategy.__class__.__name__)


def _paper_value(document: Mapping[str, Any], name: str) -> str | None:
    paper = document.get("paper")
    if isinstance(paper, Paper):
        value = getattr(paper, name)
        return str(value) if value is not None else None
    value = document.get(name)
    return str(value) if value is not None else None


def _retrieve_with_doidownloader(doi: str, email: str | None) -> tuple[bytes, str]:
    try:
        from doidownloader import DOIDownloader
        from doidownloader.doidownloader import retrieve_best_fulltext
    except ModuleNotFoundError as error:
        if error.name == "doidownloader":
            raise ImportError(
                "Install the 'remote' optional dependency to use DOIDownloader"
            ) from error
        raise ImportError(f"DOIDownloader is missing a runtime dependency: {error}") from error
    except ImportError as error:
        raise ImportError(f"DOIDownloader could not be imported: {error}") from error

    async def retrieve() -> tuple[bytes, str]:
        async with DOIDownloader(email_address=email) as client:
            result = await retrieve_best_fulltext(doi, client)
        if not result or result[3] is None or result[4] != "pdf":
            raise FileNotFoundError(f"DOIDownloader found no PDF for {doi}")
        return result[3], str(result[0])

    return asyncio.run(retrieve())


def _retrieve_with_playwright(
    doi: str, *, profile: str | Path | None = None, headless: bool = True
) -> tuple[bytes, str]:
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as error:
        raise ImportError("Install the 'remote' extra and Chromium browser support") from error

    async def retrieve() -> tuple[bytes, str]:
        async with async_playwright() as playwright:
            browser = None
            if profile:
                context = await playwright.chromium.launch_persistent_context(
                    profile,
                    headless=headless,
                )
            else:
                browser = await playwright.chromium.launch(headless=headless)
                context = await browser.new_context()
            try:
                page = await context.new_page()
                try:
                    response = await page.goto(
                        f"https://doi.org/{doi}",
                        wait_until="domcontentloaded",
                        timeout=60_000,
                    )
                    if response is not None:
                        content = await response.body()
                        if content.startswith(b"%PDF-"):
                            return content, page.url
                    links = await page.locator(
                        "a[href], link[href], meta[name='citation_pdf_url']"
                    ).evaluate_all(
                        """elements => elements.map(element => ({
                            href: element.href || element.content || '',
                            text: element.innerText || element.getAttribute('aria-label') || ''
                        }))"""
                    )
                    for url in _browser_pdf_candidates(page.url, links):
                        try:
                            response = await context.request.get(url, timeout=60_000)
                            content = await response.body()
                        except (OSError, PlaywrightTimeoutError):
                            continue
                        if content.startswith(b"%PDF-"):
                            return content, url
                finally:
                    await page.close()
            finally:
                await context.close()
                if browser is not None:
                    await browser.close()
        raise FileNotFoundError(f"Browser found no PDF links for {doi}")

    return asyncio.run(retrieve())


def _browser_pdf_candidates(base_url: str, links: Sequence[Mapping[str, str]]) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for link in links:
        href = urljoin(base_url, str(link.get("href", "")).strip())
        if urlparse(href).scheme != "https":
            continue
        value = f"{href} {link.get('text', '')}".casefold()
        score = 5 if ".pdf" in value or "/pdf" in value or value.endswith(" pdf") else 0
        score += 2 if "download" in value else 0
        if score:
            ranked.append((score, href))
    return [url for _score, url in sorted(set(ranked), reverse=True)]


def _normalize_doi(value: str) -> str:
    return value.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:").strip()


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _safe_filename(value: str) -> str:
    cleaned = "".join(
        character for character in value if character.isalnum() or character in " ._-()"
    )
    return cleaned.strip() or "paper"
