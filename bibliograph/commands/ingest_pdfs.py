"""Index PDFs found in a local directory without requiring Zotero."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from adapters.providers.filesystems.fsspec import LocalFilesystem
from rich.progress import (
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..domain import Paper
from ..logging_utils import stderr_console
from ..pipeline.ingestion_workflow import build_ingestion_graph
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
    verbose: bool = False,
) -> dict[str, Any]:
    """Discover PDFs through Ikarus fsspec, then process them through a graph."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    filesystem = LocalFilesystem()
    progress_cm = (
        Progress(
            SpinnerColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.description}"),
            console=stderr_console(),
            transient=True,
        )
        if show_progress or verbose
        else nullcontext()
    )

    def process_files(pdf_paths: list[str]) -> dict[str, Any]:
        pdfs = [Path(path).resolve() for path in pdf_paths]
        summary: dict[str, Any] = {
            "directory": str(root),
            "discovered": len(pdfs),
            "indexed": [],
            "unchanged": [],
            "invalid": [],
        }
        with progress_cm as progress:
            task_id = (
                progress.add_task("Ingesting PDFs", total=len(pdfs))
                if progress is not None and (show_progress or verbose) and pdfs
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
                try:
                    with filesystem.open(str(path), "rb") as stream:
                        valid_pdf = stream.read(5) == b"%PDF-"
                except OSError:
                    valid_pdf = False
                if not valid_pdf:
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

    workflow = build_ingestion_graph(filesystem, process_files)
    result = workflow.invoke({"root": str(root), "recursive": recursive})
    return result["summary"]
