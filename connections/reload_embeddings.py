import os
from pathlib import Path

from dotenv import load_dotenv
from pyzotero import Zotero


def load_zotero_client() -> Zotero:
    """Load Zotero credentials from environment and return a Zotero client."""
    load_dotenv()

    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE")
    api_key = os.getenv("ZOTERO_API_KEY")

    if not library_id or not api_key:
        raise OSError(
            "ZOTERO_LIBRARY_ID and ZOTERO_API_KEY must be set in the environment."
        )

    return Zotero(library_id, library_type=library_type, api_key=api_key)


def find_collection_key(zot: Zotero, collection_identifier: str) -> str:
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


def fetch_collection_items(zot: Zotero, collection_key: str) -> list[dict]:
    """Fetch all items contained in a Zotero collection."""
    return zot.everything(zot.collection_items(collection_key))


def find_pdf_attachment_ids(zot: Zotero, item_key: str, item_type: str) -> list[str]:
    """Return all PDF attachment keys for the given item."""
    pdf_keys: list[str] = []

    if item_type == "attachment":
        item = zot.item(item_key)
        data = item.get("data", {})
        if data.get("contentType") == "application/pdf":
            pdf_keys.append(item_key)
        return pdf_keys

    for child in zot.children(item_key):
        child_data = child.get("data", {})
        if (
            child_data.get("itemType") == "attachment"
            and child_data.get("contentType") == "application/pdf"
        ):
            pdf_keys.append(child_data.get("key"))

    return pdf_keys


def safe_filename(name: str) -> str:
    """Generate a filesystem-safe filename from a string."""
    cleaned = "".join(c for c in name if c.isalnum() or c in " ._-()").strip()
    return cleaned or "attachment"


def download_pdf_attachment(
    zot: Zotero, attachment_key: str, output_dir: str
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


def download_pdfs_for_collection(
    collection_identifier: str, output_dir: str = "pdfs"
) -> list[str]:
    """Retrieve all available PDF attachments for items in the specified Zotero collection.

    Args:
        collection_identifier: Zotero collection key or name.
        output_dir: Directory where PDF files will be saved.

    Returns:
        List of saved PDF file paths.
    """
    zot = load_zotero_client()
    collection_key = find_collection_key(zot, collection_identifier)
    items = fetch_collection_items(zot, collection_key)

    saved_paths: list[str] = []
    downloaded_keys = set()

    for item in items:
        data = item.get("data", {})
        item_key = data.get("key")
        item_type = data.get("itemType", "")

        if not item_key:
            continue

        pdf_keys = find_pdf_attachment_ids(zot, item_key, item_type)
        for pdf_key in pdf_keys:
            if pdf_key in downloaded_keys:
                continue
            try:
                path = download_pdf_attachment(zot, pdf_key, output_dir)
                saved_paths.append(path)
                downloaded_keys.add(pdf_key)
            except Exception as exc:
                print(f"Failed to download PDF {pdf_key}: {exc}")

    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download all PDFs for items in a Zotero collection."
    )
    parser.add_argument(
        "collection",
        help="Zotero collection key or name to retrieve PDFs from.",
    )
    parser.add_argument(
        "--output-dir",
        default="pdfs",
        help="Directory where downloaded PDFs will be stored.",
    )
    args = parser.parse_args()

    downloaded = download_pdfs_for_collection(args.collection, args.output_dir)
    print(f"Downloaded {len(downloaded)} PDF(s) to '{args.output_dir}'.")
