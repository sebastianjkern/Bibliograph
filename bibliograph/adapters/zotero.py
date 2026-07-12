"""Zotero discovery and local-storage helpers.

This module intentionally accepts a configured client instead of reading
environment variables.  It normalizes Zotero's API payloads into plain
document dictionaries so indexing can be tested without a live service.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bibliograph.adapters.pdf import is_pdf_file, locate_local_pdf
from bibliograph.domain import Paper

PAPER_ITEM_TYPES = frozenset(
    {
        "book",
        "bookSection",
        "conferencePaper",
        "encyclopediaArticle",
        "journalArticle",
        "magazineArticle",
        "manuscript",
        "newspaperArticle",
        "preprint",
        "report",
        "thesis",
    }
)


def build_zotero(config: Mapping[str, Any] | object) -> Any:
    """Create a pyzotero client from explicit non-environment configuration.

    ``config`` may be a mapping or an object with ``library_id``,
    ``library_type``, and ``api_key`` attributes.  Importing pyzotero lazily
    means commands that only query an existing index do not need the package.
    """
    library_id = _setting(config, "library_id")
    library_type = _setting(config, "library_type", "user") or "user"
    api_key = _setting(config, "api_key")
    if not library_id or not api_key:
        raise ValueError("Zotero requires both library_id and api_key")
    try:
        from pyzotero import Zotero
    except ImportError as error:  # pragma: no cover - dependency availability
        raise RuntimeError("Install the 'zotero' optional dependency to synchronize") from error
    return Zotero(str(library_id), library_type=str(library_type), api_key=str(api_key))


def resolve_collection(client: Any, identifier: str) -> str:
    """Resolve a collection key or exact collection name to its stable key."""
    if not identifier.strip():
        raise ValueError("Collection identifier must not be empty")
    collections = client.everything(client.collections())
    exact_names: list[str] = []
    for collection in collections:
        data = _data(collection)
        key = data.get("key")
        if key == identifier:
            return str(key)
        if data.get("name") == identifier and key:
            exact_names.append(str(key))
    if len(exact_names) == 1:
        return exact_names[0]
    if len(exact_names) > 1:
        raise ValueError(f"Collection name is ambiguous: {identifier!r}")
    raise ValueError(
        f"Collection not found for {identifier!r}; use the collection key or exact name"
    )


def discover_collection(
    client: Any,
    collection_key: str,
    *,
    pdf_dir: str | Path | None = None,
    storage_dirs: Iterable[str | Path] = (),
) -> list[dict[str, Any]]:
    """Return normalized document records for paper-like Zotero collection items.

    A record has a durable ``paper``, an attachment-or-parent ``source_key``,
    version metadata, and known ``path_candidates``.  One parent item with two
    PDF attachments becomes two records because their source versions can
    evolve independently.
    """
    documents: list[dict[str, Any]] = []
    storage_dirs = tuple(storage_dirs)
    collection = str(collection_key)
    for item in client.everything(client.collection_items(collection)):
        data = _data(item)
        parent_key = data.get("key")
        item_type = data.get("itemType", "")
        if not parent_key or item_type == "attachment" or not is_paper_item(item_type):
            continue
        paper = paper_from_zotero_item(item, collections=(collection,))
        attachments = [
            child
            for child in client.children(parent_key)
            if is_pdf_attachment(_data(child))
        ]
        if not attachments:
            documents.append(
                _document_record(
                    paper=paper,
                    source_key=str(parent_key),
                    version=data.get("version"),
                    attachment_key=None,
                    parent_key=str(parent_key),
                    item_type=str(item_type),
                    pdf_dir=pdf_dir,
                    storage_dirs=storage_dirs,
                )
            )
            continue
        for attachment in attachments:
            attachment_data = _data(attachment)
            attachment_key = attachment_data.get("key")
            if not attachment_key:
                continue
            documents.append(
                _document_record(
                    paper=paper,
                    source_key=str(attachment_key),
                    version=attachment_data.get("version"),
                    attachment_key=str(attachment_key),
                    parent_key=str(parent_key),
                    item_type=str(item_type),
                    pdf_dir=pdf_dir,
                    storage_dirs=storage_dirs,
                )
            )
    return documents


def paper_from_zotero_item(
    item: Mapping[str, Any], *, collections: Iterable[str] = ()
) -> Paper:
    """Normalize one Zotero item into Bibliograph's durable paper identity."""
    data = _data(item)
    authors = tuple(
        str(creator.get("lastName") or creator.get("name") or creator.get("firstName"))
        for creator in data.get("creators", [])
        if creator.get("lastName") or creator.get("name") or creator.get("firstName")
    )
    date = str(data.get("date") or "")
    return Paper(
        zotero_key=str(data.get("key") or ""),
        title=str(data.get("title") or "Untitled"),
        authors=authors,
        year=date[:4] or None,
        doi=str(data["DOI"]) if data.get("DOI") else None,
        collections=tuple(str(value) for value in collections),
    )


def is_pdf_attachment(data: Mapping[str, Any]) -> bool:
    """Return whether Zotero attachment metadata represents a PDF."""
    return (
        data.get("itemType") == "attachment"
        and str(data.get("contentType") or "").casefold() == "application/pdf"
    )


def is_paper_item(item_type: str) -> bool:
    """Return whether the Zotero item can represent a citable paper/document."""
    return item_type in PAPER_ITEM_TYPES


def find_zotero_local_pdf(storage_dir: str | Path, attachment_key: str) -> Path | None:
    """Find the first valid PDF in one Zotero ``storage/<attachment-key>`` folder."""
    attachment_dir = Path(storage_dir) / attachment_key
    if not attachment_dir.is_dir():
        return None
    for candidate in sorted(attachment_dir.rglob("*")):
        if candidate.is_file() and is_pdf_file(candidate):
            return candidate
    return None


def zotero_storage_dirs(configured: str | Path | None = None) -> tuple[Path, ...]:
    """Return configured or conventional local Zotero storage locations.

    Configuration always wins; the fallback merely recognizes Zotero's common
    platform-specific installation paths.  It intentionally does not read a
    ``ZOTERO_STORAGE_DIR`` environment variable: settings resolution owns that
    concern and passes the selected value here.
    """
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    else:
        home = Path.home()
        candidates.extend(
            (
                home / "Zotero" / "storage",
                home / ".zotero" / "zotero" / "storage",
                home / "Library" / "Application Support" / "Zotero" / "storage",
            )
        )
        appdata = os.getenv("APPDATA")
        if appdata:
            candidates.extend(Path(appdata).glob("Zotero/Zotero/Profiles/*/storage"))
        candidates.extend(home.glob(".zotero/zotero/*/storage"))
        candidates.extend(home.glob("Library/Application Support/Zotero/Profiles/*/storage"))

    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and resolved not in existing:
            existing.append(resolved)
    return tuple(existing)


def find_in_zotero_storage(
    storage_dirs: Iterable[str | Path], attachment_key: str
) -> Path | None:
    """Find a local attachment PDF across configured Zotero storage directories."""
    for storage_dir in storage_dirs:
        path = find_zotero_local_pdf(storage_dir, attachment_key)
        if path is not None:
            return path
    return None


def import_zotero_pdf(
    source: str | Path, output_dir: str | Path, attachment_key: str
) -> Path:
    """Copy a Zotero-local PDF into Bibliograph's attachment-key cache."""
    source_path = Path(source)
    if not is_pdf_file(source_path):
        raise ValueError(f"Zotero source is not a valid PDF: {source_path}")
    destination = Path(output_dir) / f"{attachment_key}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return destination


def locate_document_pdf(
    document: Mapping[str, Any],
    *,
    pdf_dir: str | Path,
    storage_dirs: Iterable[str | Path] = (),
) -> Path | None:
    """Locate a valid cached or Zotero-local PDF without mutating either location."""
    for candidate in document.get("path_candidates", ()):
        candidate_path = Path(candidate)
        if is_pdf_file(candidate_path):
            return candidate_path
    paper = document.get("paper")
    source_key = str(document.get("source_key") or "")
    paper_key = paper.zotero_key if isinstance(paper, Paper) else document.get("parent_key")
    doi = paper.doi if isinstance(paper, Paper) else document.get("doi")
    cached = locate_local_pdf(
        pdf_dir,
        source_key,
        paper_key=str(paper_key) if paper_key else None,
        doi=str(doi) if doi else None,
    )
    if cached is not None:
        return cached
    attachment_key = document.get("attachment_key")
    if attachment_key:
        return find_in_zotero_storage(storage_dirs, str(attachment_key))
    return None


def import_document_pdf(
    document: Mapping[str, Any],
    *,
    pdf_dir: str | Path,
    storage_dirs: Iterable[str | Path] = (),
) -> Path | None:
    """Return a cached PDF, importing from Zotero storage when necessary."""
    local = locate_document_pdf(document, pdf_dir=pdf_dir)
    if local is not None and local.parent.resolve() == Path(pdf_dir).resolve():
        return local
    attachment_key = document.get("attachment_key")
    if not attachment_key:
        return local
    storage_path = find_in_zotero_storage(storage_dirs, str(attachment_key))
    if storage_path is None:
        return local
    return import_zotero_pdf(storage_path, pdf_dir, str(attachment_key))


def _document_record(
    *,
    paper: Paper,
    source_key: str,
    version: Any,
    attachment_key: str | None,
    parent_key: str,
    item_type: str,
    pdf_dir: str | Path | None,
    storage_dirs: Iterable[str | Path],
) -> dict[str, Any]:
    candidates: list[str] = []
    if pdf_dir is not None:
        directory = Path(pdf_dir)
        candidates.extend(
            str(path)
            for path in _cache_candidates(directory, source_key, parent_key, paper.doi)
            if is_pdf_file(path)
        )
    if attachment_key:
        local_storage = find_in_zotero_storage(storage_dirs, attachment_key)
        if local_storage is not None:
            candidates.append(str(local_storage))
    return {
        "paper": paper,
        "source_key": source_key,
        "version": version,
        "path_candidates": tuple(dict.fromkeys(candidates)),
        "attachment_key": attachment_key,
        "parent_key": parent_key,
        "item_type": item_type,
    }


def _cache_candidates(
    directory: Path, source_key: str, parent_key: str, doi: str | None
) -> tuple[Path, ...]:
    keys = [source_key, parent_key]
    if doi:
        keys.append(_safe_identifier(doi))
    candidates: list[Path] = []
    for key in dict.fromkeys(keys):
        candidates.append(directory / f"{key}.pdf")
        if directory.is_dir():
            candidates.extend(sorted(directory.glob(f"*-{key}.pdf")))
    return tuple(candidates)


def _data(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("data", item)
    return value if isinstance(value, Mapping) else {}


def _setting(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.lower()).strip("_")
