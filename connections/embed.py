import os
from pyzotero import Zotero  # Import the Zotero file handling library 

def load_pdf_from_zotero(record_id: str) -> bytes:
    """
    Load a PDF attachment from Zotero using its attachment item ID.
    """

    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE")
    api_key = os.getenv("ZOTERO_API_KEY")

    zot = Zotero(library_id, library_type, api_key)

    # IMPORTANT: record_id must be the ATTACHMENT ID, not the parent item
    return zot.file(record_id)

def get_pdf_attachment_id(zot, parent_id: str) -> str:
    children = zot.children(parent_id)

    for item in children:
        data = item.get("data", {})
        if data.get("itemType") == "attachment" and data.get("contentType") == "application/pdf":
            return item.get("key")

    raise ValueError("No PDF attachment found.")

def load_pdf_from_parent(parent_id: str) -> bytes:
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE")
    api_key = os.getenv("ZOTERO_API_KEY")

    zot = Zotero(library_id, library_type, api_key)

    attachment_id = get_pdf_attachment_id(zot, parent_id)
    return zot.file(attachment_id)