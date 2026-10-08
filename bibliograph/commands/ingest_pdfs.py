"""Index PDFs found in a local directory without requiring Zotero."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..adapters.pdf import is_pdf_file
from ..domain import Paper


def ingest_pdfs(
    directory: str | Path,
    *,
    store,
    extract_pages,
    recursive: bool = True,
) -> dict[str, Any]:
    """Index valid PDFs using their filename as fallback bibliographic metadata."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    candidates = root.rglob("*") if recursive else root.glob("*")
    pdfs = sorted(
        (path for path in candidates if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: path.as_posix().casefold(),
    )
    summary: dict[str, Any] = {
        "directory": str(root),
        "discovered": len(pdfs),
        "indexed": [],
        "unchanged": [],
        "invalid": [],
    }
    for path in pdfs:
        if not is_pdf_file(path):
            summary["invalid"].append(str(path))
            continue
        stat = path.stat()
        source_key = "local:" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
        paper = Paper(f"LOCAL-{source_key.split(':', 1)[1]}", path.stem)
        version = f"{stat.st_mtime_ns}:{stat.st_size}"
        if not store.needs_document(source_key, version, path):
            summary["unchanged"].append(str(path))
            continue
        result = store.index_pdf(
            paper,
            source_key,
            version,
            path,
            extract_pages=extract_pages,
        )
        summary["indexed"].append(
            {"path": str(path), "source_key": source_key, "chunks": result["chunks"]}
        )
    summary["index"] = store.stats()
    return summary
