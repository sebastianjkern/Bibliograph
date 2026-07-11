import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embeddings import Embedder
from .ingest import index_pdf, paper_from_zotero_item
from .logging_utils import get_logger
from .store import SQLiteIndex

logger = get_logger("sync")


@dataclass
class SyncReport:
    indexed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    missing_local_pdf: list[str] = field(default_factory=list)
    missing_papers: list["MissingPaper"] = field(default_factory=list)


@dataclass(frozen=True)
class MissingPaper:
    item_key: str
    title: str
    doi: str | None
    attachment_keys: tuple[str, ...] = ()


def sync_collection(
    zotero: Any,
    collection_key: str,
    pdf_dir: str | Path,
    index: SQLiteIndex,
    embedder: Embedder,
    indexer: Callable[..., int] = index_pdf,
) -> SyncReport:
    """Incrementally index local PDFs belonging to one Zotero collection.

    This function never downloads files. Missing papers include titles and DOIs
    so callers can resolve them even when attachment keys are unavailable.
    """
    report = SyncReport()
    items = zotero.everything(zotero.collection_items(collection_key))
    logger.info("Found %d Zotero items in collection", len(items))
    for item in items:
        item_data = item.get("data", {})
        item_key = item_data.get("key")
        item_type = item_data.get("itemType", "")
        if not item_key or item_type == "attachment":
            continue
        attachments = _pdf_attachments(zotero, item_key)
        if not attachments:
            local_path = _find_local_pdf(pdf_dir, "", item_key, item_data.get("DOI"))
            if local_path is not None:
                version = item_data.get("version")
                if index.needs_file_index(item_key, version, str(local_path)):
                    paper = paper_from_zotero_item(item, [collection_key])
                    indexer(index, embedder, paper, local_path)
                    index.mark_file_indexed(item_key, version, str(local_path))
                    report.indexed.append(item_key)
                else:
                    report.unchanged.append(item_key)
                continue
            report.missing_papers.append(
                MissingPaper(item_key, item_data.get("title", "Untitled"), item_data.get("DOI"))
            )
            logger.info("Missing PDF: %s", item_data.get("title", "Untitled"))
            continue
        missing_keys: list[str] = []
        for attachment in attachments:
            attachment_data = attachment.get("data", {})
            attachment_key = attachment_data.get("key")
            if not attachment_key:
                missing_keys.append("")
                continue
            local_path = _find_local_pdf(pdf_dir, attachment_key, item_key, item_data.get("DOI"))
            if local_path is None:
                report.missing_local_pdf.append(attachment_key)
                missing_keys.append(attachment_key)
                logger.info("Missing local PDF: %s", item_data.get("title", "Untitled"))
                continue
            version = attachment_data.get("version")
            if not index.needs_file_index(attachment_key, version, str(local_path)):
                report.unchanged.append(attachment_key)
                continue
            paper = paper_from_zotero_item(item, [collection_key])
            indexer(index, embedder, paper, local_path)
            index.mark_file_indexed(attachment_key, version, str(local_path))
            report.indexed.append(attachment_key)
        if missing_keys:
            report.missing_papers.append(
                MissingPaper(
                    item_key,
                    item_data.get("title", "Untitled"),
                    item_data.get("DOI"),
                    tuple(key for key in missing_keys if key),
                )
            )
    return report


def _pdf_attachments(zotero: Any, parent_key: str) -> list[dict]:
    return [
        child
        for child in zotero.children(parent_key)
        if child.get("data", {}).get("itemType") == "attachment"
        and child.get("data", {}).get("contentType") == "application/pdf"
    ]


def _find_local_pdf(
    pdf_dir: str | Path,
    attachment_key: str,
    parent_key: str | None = None,
    doi: str | None = None,
) -> Path | None:
    directory = Path(pdf_dir)
    keys = [key for key in (attachment_key, parent_key) if key]
    if doi:
        keys.append(re.sub(r"[^A-Za-z0-9_-]+", "_", doi.lower()).strip("_"))
    for key in keys:
        exact = directory / f"{key}.pdf"
        if exact.is_file():
            return exact
        matches = sorted(directory.glob(f"*-{key}.pdf"))
        if matches:
            return matches[0]
    return None
