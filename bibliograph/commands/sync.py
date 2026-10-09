"""The sole mutating Bibliograph workflow."""

from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..adapters.acquisition import acquire_pdf
from ..adapters.zotero import discover_collection, resolve_collection
from ..knowledge import KnowledgeDocument
from .progress import StageProgress


def sync_library(
    settings: Mapping[str, Any],
    *,
    store,
    zotero: Any,
    strategies: Iterable[Callable],
    extract_pages: Callable,
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
        task_id = None
        if progress is not None and show_progress and documents:
            task_id = progress.add_task("Synchronizing documents", total=len(documents))
        stages = (
            StageProgress(progress, task_id)
            if progress is not None and task_id is not None
            else None
        )
        for position, document in enumerate(documents, start=1):
            source_key = str(document["source_key"])
            title = document["paper"].title
            prefix = f"[{position}/{len(documents)}]"
            if stages is not None:
                stages.update(f"{prefix} Acquire PDF · {title}")
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
                if stages is not None:
                    stages.update(f"{prefix} Missing PDF · {title}")
                if task_id is not None and progress is not None:
                    progress.advance(task_id)
                continue
            if acquired.get("downloaded"):
                summary["downloaded"].append(source_key)
            if not store.needs_document(source_key, document.get("version"), path):
                summary["unchanged"].append(source_key)
                if stages is not None:
                    stages.update(f"{prefix} Unchanged · {title}")
                if task_id is not None and progress is not None:
                    progress.advance(task_id)
                continue
            try:
                if stages is not None:
                    stages.update(f"{prefix} Extract and index · {title}")
                index_document = getattr(store, "index_document", None)
                if index_document is not None:
                    result = index_document(
                        KnowledgeDocument(
                            source_key=source_key,
                            paper=document["paper"],
                            path=Path(path),
                            version=(
                                None
                                if document.get("version") is None
                                else str(document["version"])
                            ),
                            metadata=document.get("metadata", {}),
                        ),
                        extract_pages=extract_pages,
                    )
                else:
                    # Compatibility for legacy stores during the knowledge-engine migration.
                    result = store.index_pdf(
                        document["paper"],
                        source_key,
                        document.get("version"),
                        path,
                        extract_pages=extract_pages,
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
            if stages is not None:
                stages.update(f"{prefix} Indexed · {title}")
            if task_id is not None and progress is not None:
                progress.advance(task_id)
        if stages is not None:
            stages.complete("✓ Synchronization complete")
    store.mark_sync_complete()
    summary["index"] = store.stats()
    return summary


def _attempt_detail(attempts: Iterable[Mapping[str, Any]]) -> str:
    values = [
        f"{attempt.get('source', 'source')}: {attempt.get('error', 'not found')}"
        for attempt in attempts
    ]
    return "; ".join(values) if values else "No acquisition source returned a PDF"
