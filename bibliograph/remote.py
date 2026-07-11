import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen


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
            try:
                resolved = resolver.resolve(doi)
            except (OSError, ValueError):
                continue
            for candidate in resolved:
                _validate_candidate(candidate)
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


def download_remote_pdf(
    doi: str,
    output_dir: str | Path,
    email: str | None = None,
    title: str | None = None,
    item_key: str | None = None,
    resolvers: Iterable[RemoteResolver] | None = None,
    openalex_api_key: str | None = None,
) -> RemoteDownloadResult:
    """Download one PDF from an explicitly legal remote resolver chain."""
    chain = ResolverChain(
        resolvers
        or _default_resolvers(
            email or os.getenv("UNPAYWALL_EMAIL"),
            openalex_api_key or os.getenv("OPENALEX_API_KEY"),
        )
    )
    candidates = chain.resolve(doi)
    if not candidates:
        raise FileNotFoundError(f"No legal open-access PDF location found for DOI {doi}")

    filename_key = item_key or _safe_key(_normalize_doi(doi))
    filepath = Path(output_dir) / f"{_safe_filename(title or doi)}-{filename_key}.pdf"
    for candidate in candidates:
        if filepath.is_file():
            return RemoteDownloadResult(
                str(filepath), False, candidate.url, candidate.resolver, candidate.legal_basis
            )
        try:
            content = _get_pdf(candidate.url)
        except (OSError, ValueError):
            continue
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(content)
        return RemoteDownloadResult(
            str(filepath), True, candidate.url, candidate.resolver, candidate.legal_basis
        )
    raise FileNotFoundError(f"Legal resolvers returned no downloadable PDF for DOI {doi}")


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
