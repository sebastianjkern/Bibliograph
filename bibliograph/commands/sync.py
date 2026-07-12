"""The sole mutating Bibliograph workflow."""

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..adapters.acquisition import acquire_pdf
from ..adapters.zotero import discover_collection, resolve_collection
from ..pipeline.indexing import index_document


def sync_library(
    settings: Mapping[str, Any],
    *,
    store,
    embedding: Mapping[str, Any],
    zotero: Any,
    strategies: Iterable[Callable],
    extract_pages: Callable,
    chunk_document: Callable,
    storage_dirs: Iterable[str | Path] = (),
    collection: str | None = None,
) -> dict[str, Any]:
    """Synchronize one Zotero collection into an injected writable store."""
    identifier = collection or str(settings["collection"])
    collection_key = resolve_collection(zotero, identifier)
    pdf_dir = settings["pdf_dir"]
    documents = discover_collection(
        zotero,
        collection_key,
        pdf_dir=pdf_dir,
        storage_dirs=storage_dirs,
    )
    summary: dict[str, Any] = {
        "collection": collection_key,
        "discovered": len(documents),
        "indexed": [],
        "unchanged": [],
        "empty": [],
        "missing": [],
        "downloaded": [],
    }
    for document in documents:
        source_key = str(document["source_key"])
        acquired = acquire_pdf(document, strategies=strategies)
        path = acquired.get("path")
        if not path:
            detail = _attempt_detail(acquired.get("attempts", ()))
            store.record_document_failure(
                document["paper"],
                source_key,
                document.get("version"),
                state="missing",
                detail=detail,
            )
            summary["missing"].append(
                {
                    "source_key": source_key,
                    "title": document["paper"].title,
                    "attempts": acquired.get("attempts", ()),
                }
            )
            continue
        if acquired.get("downloaded"):
            summary["downloaded"].append(source_key)
        if not store.needs_document(source_key, document.get("version"), path):
            summary["unchanged"].append(source_key)
            continue
        try:
            result = index_document(
                document["paper"],
                source_key,
                document.get("version"),
                path,
                extract_pages=extract_pages,
                chunk_document=chunk_document,
                embed_documents=embedding["embed_documents"],
                store=store,
                batch_size=int(settings["embedding"]["batch_size"]),
                supported_kinds=embedding.get("kinds", ("text",)),
            )
        except Exception as error:
            store.record_document_failure(
                document["paper"],
                source_key,
                document.get("version"),
                state="failed",
                detail=str(error),
                path=path,
            )
            raise
        if result["chunks"]:
            summary["indexed"].append({"source_key": source_key, "chunks": result["chunks"]})
        else:
            summary["empty"].append(source_key)
    store.mark_sync_complete()
    summary["index"] = store.stats()
    return summary


def _attempt_detail(attempts: Iterable[Mapping[str, Any]]) -> str:
    values = [
        f"{attempt.get('source', 'source')}: {attempt.get('error', 'not found')}"
        for attempt in attempts
    ]
    return "; ".join(values) if values else "No acquisition source returned a PDF"
