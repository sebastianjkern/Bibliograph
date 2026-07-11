import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class RemoteDownloadResult:
    path: str
    downloaded: bool
    source_url: str


def download_unpaywall_pdf(
    doi: str,
    output_dir: str | Path,
    email: str | None = None,
    title: str | None = None,
    item_key: str | None = None,
) -> RemoteDownloadResult:
    """Download one legally listed open-access PDF for a DOI."""
    email = email or os.getenv("UNPAYWALL_EMAIL")
    if not email:
        raise ValueError("An email address is required for the Unpaywall API")
    normalized_doi = _normalize_doi(doi)
    record = _get_json(
        f"https://api.unpaywall.org/v2/{quote(normalized_doi, safe='')}",
        {"email": email},
    )
    location = next(
        (item for item in record.get("oa_locations", []) if item.get("url_for_pdf")),
        None,
    )
    if location is None:
        raise FileNotFoundError(f"No open-access PDF location found for DOI {doi}")
    source_url = location["url_for_pdf"]
    request = Request(source_url, headers={"User-Agent": "Bibliograph/0.1"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
        content_type = response.headers.get_content_type()
    if content_type != "application/pdf" and not content.startswith(b"%PDF"):
        raise ValueError(f"Remote source did not return a PDF: {source_url}")

    filename_key = item_key or _safe_key(normalized_doi)
    filename = f"{_safe_filename(title or normalized_doi)}-{filename_key}.pdf"
    filepath = Path(output_dir) / filename
    if filepath.is_file():
        return RemoteDownloadResult(str(filepath), downloaded=False, source_url=source_url)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(content)
    return RemoteDownloadResult(str(filepath), downloaded=True, source_url=source_url)


def _get_json(url: str, params: dict[str, str]) -> dict:
    query = "&".join(f"{quote(key)}={quote(value)}" for key, value in params.items())
    request = Request(f"{url}?{query}", headers={"User-Agent": "Bibliograph/0.1"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _normalize_doi(value: str) -> str:
    return value.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:").strip()


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _safe_filename(value: str) -> str:
    cleaned = "".join(
        character for character in value if character.isalnum() or character in " ._-()"
    )
    return cleaned.strip() or "paper"
