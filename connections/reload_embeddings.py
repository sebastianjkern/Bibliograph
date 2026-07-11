import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bibliograph.zotero_sync import (
    find_zotero_local_pdf,
    import_zotero_pdf,
    is_pdf_attachment,
    is_pdf_file,
    is_remote_downloadable_item,
    zotero_storage_dirs,
)


@dataclass(frozen=True)
class DownloadResult:
    path: str
    downloaded: bool
    attachment_key: str


def load_zotero_client() -> Any:
    """Load Zotero credentials from environment and return a Zotero client."""
    from dotenv import load_dotenv

    load_dotenv()

    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE")
    api_key = os.getenv("ZOTERO_API_KEY")

    if not library_id or not api_key:
        raise OSError(
            "ZOTERO_LIBRARY_ID and ZOTERO_API_KEY must be set in the environment."
        )

    from pyzotero import Zotero

    return Zotero(library_id, library_type=library_type, api_key=api_key)


def find_collection_key(zot: Any, collection_identifier: str) -> str:
    """Resolve a collection identifier to a Zotero collection key.

    The identifier can be either the collection key or the collection name.
    """
    if not collection_identifier:
        raise ValueError("Collection identifier must not be empty.")

    collections = zot.everything(zot.collections())
    for collection in collections:
        data = collection.get("data", {})
        if data.get("key") == collection_identifier:
            return data.get("key")
        if data.get("name") == collection_identifier:
            return data.get("key")

    raise ValueError(
        f"Collection not found for identifier '{collection_identifier}'. "
        "Use the collection key or exact collection name."
    )


def fetch_collection_items(zot: Any, collection_key: str) -> list[dict]:
    """Fetch all items contained in a Zotero collection."""
    return zot.everything(zot.collection_items(collection_key))


def find_pdf_attachment_ids(zot: Any, item_key: str, item_type: str) -> list[str]:
    """Return all PDF attachment keys for the given item."""
    pdf_keys: list[str] = []

    if item_type == "attachment":
        item = zot.item(item_key)
        data = item.get("data", {})
        if is_pdf_attachment(data):
            pdf_keys.append(item_key)
        return pdf_keys

    for child in zot.children(item_key):
        child_data = child.get("data", {})
        if is_pdf_attachment(child_data):
            pdf_keys.append(child_data.get("key"))

    return pdf_keys


def resolve_item_key(
    zot: Any,
    item_key: str | None = None,
    doi: str | None = None,
    title: str | None = None,
) -> str:
    """Resolve a user-friendly DOI or title to one Zotero parent key."""
    identifiers = [value for value in (item_key, doi, title) if value]
    if len(identifiers) != 1:
        raise ValueError("Provide exactly one of item_key, doi, or title")
    if item_key:
        return item_key

    query = doi or title
    candidates = [
        item.get("data", {})
        for item in zot.everything(zot.items(q=query))
        if item.get("data", {}).get("itemType") != "attachment"
    ]
    if doi:
        wanted_doi = _normalize_doi(doi)
        candidates = [
            item for item in candidates if _normalize_doi(item.get("DOI", "")) == wanted_doi
        ]
    else:
        wanted_title = title.strip().casefold()
        candidates = [
            item
            for item in candidates
            if item.get("title", "").strip().casefold() == wanted_title
        ]
    if len(candidates) == 1:
        return candidates[0].get("key", "")
    if not candidates:
        raise ValueError(f"No Zotero item found for {doi or title!r}")
    descriptions = ", ".join(item.get("key", "unknown") for item in candidates[:5])
    raise ValueError(f"Multiple Zotero items matched; use an exact DOI or item key: {descriptions}")


def _normalize_doi(value: str) -> str:
    return value.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:").strip()


def safe_filename(name: str) -> str:
    """Generate a filesystem-safe filename from a string."""
    cleaned = "".join(c for c in name if c.isalnum() or c in " ._-()").strip()
    return cleaned or "attachment"


def download_pdf_attachment(
    zot: Any, attachment_key: str, output_dir: str
) -> str:
    """Download a single PDF attachment and return its saved path."""
    pdf_bytes = zot.file(attachment_key)
    item = zot.item(attachment_key)
    item_data = item.get("data", {})

    title = item_data.get("filename") or item_data.get("title") or attachment_key
    filename = f"{safe_filename(title)}-{attachment_key}.pdf"
    filepath = Path(output_dir) / filename

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as fp:
        fp.write(pdf_bytes)

    return str(filepath)


def find_local_pdf(output_dir: str, attachment_key: str) -> Path | None:
    """Find a previously downloaded PDF for one Zotero attachment key."""
    directory = Path(output_dir)
    exact = directory / f"{attachment_key}.pdf"
    if is_pdf_file(exact):
        return exact
    matches = sorted(directory.glob(f"*-{attachment_key}.pdf"))
    return next((match for match in matches if is_pdf_file(match)), None)


def download_pdf_for_item(
    item_key: str | None = None,
    output_dir: str = "pdfs",
    attachment_key: str | None = None,
    doi: str | None = None,
    title: str | None = None,
    zotero_storage_dir: str | None = None,
) -> DownloadResult:
    """Download one PDF attachment for one Zotero item, only if it is not local.

    ``item_key`` may be a parent item key or an attachment key. When a parent has
    multiple PDF attachments, pass ``attachment_key`` to select one explicitly;
    otherwise only the first PDF attachment is considered.
    """
    zot = load_zotero_client()
    item_key = resolve_item_key(zot, item_key, doi, title)
    item = zot.item(item_key)
    item_data = item.get("data", {})
    if not is_remote_downloadable_item(item_data.get("itemType", "")):
        raise ValueError(
            f"Zotero item {item_key} has unsupported type {item_data.get('itemType', '')!r}; "
            "PDF downloads are limited to paper-like items"
        )
    attachment_keys = find_pdf_attachment_ids(
        zot, item_key, item_data.get("itemType", "")
    )
    if attachment_key:
        if attachment_key not in attachment_keys:
            raise ValueError(f"Attachment {attachment_key} is not a PDF for item {item_key}")
        selected_key = attachment_key
    elif attachment_keys:
        selected_key = attachment_keys[0]
    else:
        raise ValueError(f"No PDF attachment found for Zotero item {item_key}")

    existing = find_local_pdf(output_dir, selected_key)
    if existing:
        return DownloadResult(str(existing), downloaded=False, attachment_key=selected_key)
    for storage_dir in zotero_storage_dirs(zotero_storage_dir):
        zotero_path = find_zotero_local_pdf(storage_dir, selected_key)
        if zotero_path is not None:
            imported = import_zotero_pdf(zotero_path, output_dir, selected_key)
            return DownloadResult(str(imported), downloaded=True, attachment_key=selected_key)
    path = download_pdf_attachment(zot, selected_key, output_dir)
    return DownloadResult(path, downloaded=True, attachment_key=selected_key)


def download_pdfs_for_collection(
    collection_identifier: str, output_dir: str = "pdfs"
) -> list[str]:
    """Prevent accidental collection-wide downloads."""
    raise RuntimeError(
        "Collection-wide PDF downloads are disabled; use the single-item downloader"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download one missing PDF from Zotero.")
    parser.add_argument("item_key", nargs="?", help="Optional Zotero parent item or attachment key")
    identifier = parser.add_mutually_exclusive_group()
    identifier.add_argument("--doi", help="Resolve the Zotero item by DOI")
    identifier.add_argument("--title", help="Resolve the Zotero item by exact title")
    parser.add_argument("--attachment-key", help="Select one PDF when there are several")
    parser.add_argument(
        "--output-dir",
        default="pdfs",
        help="Directory where downloaded PDFs will be stored.",
    )
    args = parser.parse_args()

    result = download_pdf_for_item(
        args.item_key,
        args.output_dir,
        args.attachment_key,
        args.doi,
        args.title,
    )
    status = "Downloaded" if result.downloaded else "Already present"
    print(f"{status}: {result.path}")
