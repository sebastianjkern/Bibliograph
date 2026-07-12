"""The sole mutating Bibliograph workflow."""

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

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
    show_progress: bool = True,
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
    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=Console(stderr=True),
            transient=True,
        )
        if show_progress and documents
        else nullcontext()
    )
    with progress_cm as progress:
        task_id = progress.add_task("Synchronizing documents", total=len(documents)) if show_progress and documents else None
        for document in documents:
            source_key = str(document["source_key"])
            title = document["paper"].title
            if task_id is not None:
                progress.update(task_id, description=f"Acquiring {title}")
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
                        "title": title,
                        "attempts": acquired.get("attempts", ()),
                    }
                )
                if task_id is not None:
                    progress.advance(task_id)
                continue
            if acquired.get("downloaded"):
                summary["downloaded"].append(source_key)
            if not store.needs_document(source_key, document.get("version"), path):
                summary["unchanged"].append(source_key)
                if task_id is not None:
                    progress.update(task_id, description=f"Skipping {title}")
                    progress.advance(task_id)
                continue
            try:
                if task_id is not None:
                    progress.update(task_id, description=f"Indexing {title}")
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
            if task_id is not None:
                progress.advance(task_id)
    store.mark_sync_complete()
    summary["index"] = store.stats()
    return summary


def _attempt_detail(attempts: Iterable[Mapping[str, Any]]) -> str:
    values = [
        f"{attempt.get('source', 'source')}: {attempt.get('error', 'not found')}"
        for attempt in attempts
    ]
    return "; ".join(values) if values else "No acquisition source returned a PDF"
