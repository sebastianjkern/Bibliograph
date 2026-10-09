"""Index PDFs found in a local directory without requiring Zotero."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
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

from ..adapters.pdf import is_pdf_file
from ..domain import Paper
from .indexing import index_available_document
from .progress import StageProgress


def ingest_pdfs(
    directory: str | Path,
    *,
    store,
    extract_pages,
    recursive: bool = True,
    show_progress: bool = True,
    force_reindex: bool = False,
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
        if show_progress and pdfs
        else nullcontext()
    )
    with progress_cm as progress:
        task_id = (
            progress.add_task("Ingesting PDFs", total=len(pdfs))
            if progress is not None and show_progress and pdfs
            else None
        )
        stages = (
            StageProgress(progress, task_id)
            if progress is not None and task_id is not None
            else None
        )
        for position, path in enumerate(pdfs, start=1):
            prefix = f"[{position}/{len(pdfs)}]"
            if stages is not None:
                stages.update(f"{prefix} Checking PDF · {path.name}")
            if not is_pdf_file(path):
                summary["invalid"].append(str(path))
                if stages is not None:
                    stages.update(f"{prefix} Invalid PDF · {path.name}")
                if task_id is not None and progress is not None:
                    progress.advance(task_id)
                continue
            stat = path.stat()
            source_key = "local:" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
            paper = Paper(f"LOCAL-{source_key.split(':', 1)[1]}", path.stem)
            version = f"{stat.st_mtime_ns}:{stat.st_size}"

            def report_progress(message: str, *, prefix=prefix, title=path.name) -> None:
                if stages is not None:
                    stages.update(f"{prefix} {message} · {title}")

            result = index_available_document(
                store,
                paper,
                source_key,
                version,
                path,
                extract_pages=extract_pages,
                progress=report_progress if stages is not None else None,
                force_reindex=force_reindex,
            )
            if result["state"] == "unchanged":
                summary["unchanged"].append(str(path))
                if stages is not None:
                    stages.update(f"{prefix} Unchanged · {path.name}")
            else:
                summary["indexed"].append(
                    {"path": str(path), "source_key": source_key, "chunks": result["chunks"]}
                )
                if stages is not None:
                    stages.update(f"{prefix} Indexed · {path.name}")
            if task_id is not None and progress is not None:
                progress.advance(task_id)
        if stages is not None:
            stages.complete("✓ PDF ingestion complete")
    summary["index"] = store.stats()
    return summary
