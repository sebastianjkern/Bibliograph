import asyncio
import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

from .logging_utils import get_logger

logger = get_logger("remote")


@dataclass(frozen=True)
class RemoteCandidate:
    url: str
    resolver: str
    legal_basis: str
    license: str | None = None
    landing_page_url: str | None = None


class RemoteResolver(Protocol):
    name: str

    def resolve(self, doi: str) -> Sequence[RemoteCandidate]: ...


class UnpaywallResolver:
    name = "unpaywall"

    def __init__(self, email: str, api_base_url: str = "https://api.unpaywall.org/v2"):
        self.email = email
        self.api_base_url = api_base_url.rstrip("/")

    def resolve(self, doi: str) -> Sequence[RemoteCandidate]:
        logger.debug("Querying Unpaywall for %s", doi)
        record = _get_json(
            f"{self.api_base_url}/{quote(_normalize_doi(doi), safe='')}",
            {"email": self.email},
        )
        return [
            RemoteCandidate(
                location["url_for_pdf"],
                self.name,
                "Unpaywall open-access location",
                location.get("license"),
                location.get("url_for_landing_page"),
            )
            for location in record.get("oa_locations", [])
            if location.get("url_for_pdf") and location.get("is_oa", record.get("is_oa", False))
        ]


class OpenAlexResolver:
    name = "openalex"

    def __init__(self, api_key: str, api_base_url: str = "https://api.openalex.org/works"):
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")

    def resolve(self, doi: str) -> Sequence[RemoteCandidate]:
        logger.debug("Querying OpenAlex for %s", doi)
        record = _get_json(
            f"{self.api_base_url}/https://doi.org/{quote(_normalize_doi(doi), safe='')}",
            {"api_key": self.api_key},
        )
        locations = [record.get("best_oa_location", {})] + record.get("locations", [])
        return [
            RemoteCandidate(
                location["pdf_url"],
                self.name,
                "OpenAlex open-access location",
                location.get("license"),
                location.get("landing_page_url"),
            )
            for location in locations
            if location.get("pdf_url") and location.get("is_oa")
        ]


class ResolverChain:
    """Ordered, legal-source-only resolver chain."""

    def __init__(self, resolvers: Iterable[RemoteResolver]):
        self.resolvers = tuple(resolvers)

    def resolve(self, doi: str) -> list[RemoteCandidate]:
        candidates: list[RemoteCandidate] = []
        for resolver in self.resolvers:
            logger.debug("Trying remote resolver: %s", resolver.name)
            try:
                resolved = resolver.resolve(doi)
            except (OSError, ValueError):
                continue
            for candidate in resolved:
                _validate_candidate(candidate)
                logger.debug("Resolver %s returned %s", resolver.name, candidate.url)
                if candidate.url not in {item.url for item in candidates}:
                    candidates.append(candidate)
        return candidates


@dataclass(frozen=True)
class RemoteDownloadResult:
    path: str
    downloaded: bool
    source_url: str
    resolver: str
    legal_basis: str


@dataclass(frozen=True)
class ManualDownloadHint:
    """A legal URL a user can open to download a paper manually."""

    url: str
    source: str
    reason: str


class RemoteDownloadError(FileNotFoundError):
    """Raised when automatic retrieval fails but manual download hints exist."""

    def __init__(self, doi: str, hints: Sequence[ManualDownloadHint], reason: str):
        self.doi = doi
        self.hints = tuple(hints)
        super().__init__(reason)


def download_remote_pdf(
    doi: str,
    output_dir: str | Path,
    email: str | None = None,
    title: str | None = None,
    item_key: str | None = None,
    resolvers: Iterable[RemoteResolver] | None = None,
    openalex_api_key: str | None = None,
) -> RemoteDownloadResult:
    """Download one PDF using DOIDownloader, with legal manual fallback hints.

    Passing ``resolvers`` retains the lower-level resolver-chain API for custom
    integrations and tests. Without it, the ``doidownloader`` package is the
    default backend; it may retrieve publisher-hosted full text when the caller
    has legitimate access through their network or institution.
    """
    filename_key = item_key or _safe_key(_normalize_doi(doi))
    filepath = Path(output_dir) / f"{_safe_filename(title or doi)}-{filename_key}.pdf"
    if filepath.is_file():
        return RemoteDownloadResult(str(filepath), False, "", "cache", "Local file")

    if resolvers is None:
        try:
            content, source_url = _retrieve_with_doidownloader(doi, email)
        except (ImportError, OSError, ValueError) as error:
            logger.warning("DOIDownloader could not retrieve %s: %s", doi, error)
            try:
                content, source_url = _retrieve_with_playwright(doi)
            except (ImportError, OSError, ValueError) as browser_error:
                logger.warning("Playwright could not retrieve %s: %s", doi, browser_error)
                hints = PyDOIResolver().manual_hints(doi)
                raise RemoteDownloadError(
                    doi,
                    hints,
                    f"No PDF downloaded for {doi}: {error}; "
                    f"Playwright fallback: {browser_error}. "
                    "Open one of the manual download hints.",
                ) from browser_error
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(content)
            return RemoteDownloadResult(
                str(filepath), True, source_url, "playwright", "Accessible publisher page"
            )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(content)
        return RemoteDownloadResult(
            str(filepath), True, source_url, "doidownloader", "Accessible legal publisher/OA route"
        )

    chain = ResolverChain(
        resolvers
    )
    candidates = chain.resolve(doi)
    logger.info("Found %d legal remote PDF candidates for %s", len(candidates), doi)
    if not candidates:
        raise FileNotFoundError(f"No legal open-access PDF location found for DOI {doi}")

    for candidate in candidates:
        if filepath.is_file():
            return RemoteDownloadResult(
                str(filepath), False, candidate.url, candidate.resolver, candidate.legal_basis
            )
        try:
            logger.info("Downloading PDF via %s: %s", candidate.resolver, candidate.url)
            content = _get_pdf(candidate.url)
        except (OSError, ValueError):
            logger.warning("Remote PDF candidate failed: %s", candidate.url)
            continue
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(content)
        return RemoteDownloadResult(
            str(filepath), True, candidate.url, candidate.resolver, candidate.legal_basis
        )
    raise FileNotFoundError(f"Legal resolvers returned no downloadable PDF for DOI {doi}")


class PyDOIResolver:
    """Resolve DOI registry landing URLs for manual, user-driven downloads."""

    name = "pydoi"

    def resolve(self, doi: str) -> Sequence[RemoteCandidate]:
        try:
            import pydoi
        except ImportError as error:
            raise ImportError("Install the 'remote' extra to enable pyDOI hints") from error

        urls = pydoi.get_url(_normalize_doi(doi), allow_multi=True)
        if isinstance(urls, str):
            urls = [urls]
        return [
            RemoteCandidate(
                url,
                self.name,
                "DOI registry landing URL for manual access",
                landing_page_url=url,
            )
            for url in urls or []
            if isinstance(url, str) and url.startswith("https://")
        ]

    def manual_hints(self, doi: str) -> list[ManualDownloadHint]:
        hints = [
            ManualDownloadHint(
                f"https://doi.org/{_normalize_doi(doi)}",
                "doi.org",
                "Open the DOI landing page and use your publisher or institutional access.",
            )
        ]
        try:
            candidates = self.resolve(doi)
        except (ImportError, OSError, ValueError):
            return hints
        for candidate in candidates:
            if candidate.url not in {hint.url for hint in hints}:
                hints.append(
                    ManualDownloadHint(
                        candidate.url,
                        self.name,
                        "Resolved publisher landing page; download the PDF manually if available.",
                    )
                )
        return hints


def download_unpaywall_pdf(
    doi: str,
    output_dir: str | Path,
    email: str | None = None,
    title: str | None = None,
    item_key: str | None = None,
) -> RemoteDownloadResult:
    """Backward-compatible Unpaywall-only download helper."""
    if not email and not os.getenv("UNPAYWALL_EMAIL"):
        raise ValueError("An email address is required for the Unpaywall API")
    return download_remote_pdf(
        doi,
        output_dir,
        email=email,
        title=title,
        item_key=item_key,
        resolvers=[UnpaywallResolver(email or os.environ["UNPAYWALL_EMAIL"])],
    )


def _default_resolvers(email: str | None, openalex_api_key: str | None) -> list[RemoteResolver]:
    resolvers: list[RemoteResolver] = []
    if email:
        resolvers.append(UnpaywallResolver(email))
    if openalex_api_key:
        resolvers.append(OpenAlexResolver(openalex_api_key))
    return resolvers


def _retrieve_with_doidownloader(doi: str, email: str | None) -> tuple[bytes, str]:
    """Call DOIDownloader's legal full-text workflow without its result database."""
    try:
        from doidownloader import DOIDownloader
        from doidownloader.doidownloader import retrieve_best_fulltext
    except ModuleNotFoundError as error:
        if error.name == "doidownloader":
            raise ImportError(
                "doidownloader is not installed in the active Python environment. "
                "It requires Python 3.12 or newer; use "
                "'uv sync --python 3.12 --extra remote' and run with "
                "'uv run --python 3.12 bibliograph ...'."
            ) from error
        raise ImportError(f"doidownloader is missing a runtime dependency: {error}") from error
    except ImportError as error:
        raise ImportError(f"doidownloader could not be imported: {error}") from error

    async def retrieve() -> tuple[bytes, str]:
        async with DOIDownloader(email_address=email) as client:
            result = await retrieve_best_fulltext(_normalize_doi(doi), client)
        if not result or result[3] is None or result[4] != "pdf":
            raise FileNotFoundError(f"DOIDownloader found no PDF for {doi}")
        return result[3], str(result[0])

    return asyncio.run(retrieve())


def _retrieve_with_playwright(doi: str) -> tuple[bytes, str]:
    """Retrieve a PDF from a DOI-resolved publisher page using a browser session."""
    try:
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except ModuleNotFoundError as error:
        raise ImportError(
            "Playwright is not installed or its browser support is unavailable. "
            "Run 'uv run playwright install chromium'."
        ) from error

    async def retrieve() -> tuple[bytes, str]:
        profile = os.getenv("BIBLIOGRAPH_PLAYWRIGHT_PROFILE")
        headless = os.getenv("BIBLIOGRAPH_PLAYWRIGHT_HEADLESS", "true").casefold() not in {
            "0",
            "false",
            "no",
        }
        landing_urls = [candidate.url for candidate in PyDOIResolver().resolve(doi)]
        if not landing_urls:
            landing_urls = [f"https://doi.org/{_normalize_doi(doi)}"]

        async with async_playwright() as playwright:
            browser = None
            if profile:
                context = await playwright.chromium.launch_persistent_context(
                    profile, headless=headless
                )
            else:
                browser = await playwright.chromium.launch(headless=headless)
                context = await browser.new_context()
            try:
                for landing_url in landing_urls:
                    page = await context.new_page()
                    try:
                        response = await page.goto(
                            landing_url,
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
                        for pdf_url in _browser_pdf_candidates(page.url, links):
                            try:
                                pdf_response = await context.request.get(
                                    pdf_url, timeout=60_000
                                )
                                content = await pdf_response.body()
                            except (OSError, PlaywrightTimeoutError):
                                continue
                            if content.startswith(b"%PDF-"):
                                return content, pdf_url
                    except PlaywrightTimeoutError:
                        logger.debug("Browser navigation timed out: %s", landing_url)
                    finally:
                        await page.close()
            finally:
                await context.close()
                if browser is not None:
                    await browser.close()
        raise FileNotFoundError(f"Playwright found no PDF links for {doi}")

    return asyncio.run(retrieve())


def _browser_pdf_candidates(base_url: str, links: Sequence[dict[str, str]]) -> list[str]:
    """Rank HTTPS PDF/download links discovered on a publisher page."""
    candidates: list[tuple[int, str]] = []
    for link in links:
        href = urljoin(base_url, link.get("href", "").strip())
        parsed = urlparse(href)
        if parsed.scheme != "https":
            continue
        value = f"{href} {link.get('text', '')}".casefold()
        score = 0
        if ".pdf" in value or "/pdf" in value or " pdf" in value or value.endswith("pdf"):
            score += 5
        if "download" in value:
            score += 2
        if "full text" in value or value.rstrip().endswith(" pdf"):
            score += 1
        if score:
            candidates.append((score, href))
    return [url for _score, url in sorted(set(candidates), reverse=True)]


def _validate_candidate(candidate: RemoteCandidate) -> None:
    if not candidate.legal_basis:
        raise ValueError(f"Resolver {candidate.resolver} did not provide a legal basis")
    if not candidate.url.startswith("https://"):
        raise ValueError(f"Resolver {candidate.resolver} returned a non-HTTPS URL")


def _get_json(url: str, params: dict[str, str]) -> dict:
    query = "&".join(f"{quote(key)}={quote(value)}" for key, value in params.items())
    request = Request(f"{url}?{query}", headers={"User-Agent": "Bibliograph/0.1"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _get_pdf(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Bibliograph/0.1", "Accept": "application/pdf"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
        content_type = response.headers.get_content_type()
    if content_type != "application/pdf" and not content.startswith(b"%PDF"):
        raise ValueError(f"Remote source did not return a PDF: {url}")
    return content


def _normalize_doi(value: str) -> str:
    return value.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:").strip()


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _safe_filename(value: str) -> str:
    cleaned = "".join(
        character for character in value if character.isalnum() or character in " ._-()"
    )
    return cleaned.strip() or "paper"
