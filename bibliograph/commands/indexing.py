"""Shared per-document indexing behavior for all PDF sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import signature
from pathlib import Path
from typing import Any

from ..domain import Paper
from ..knowledge import KnowledgeDocument


def index_available_document(
    store,
    paper: Paper,
    source_key: str,
    version: str | int | None,
    path: str | Path,
    *,
    extract_pages: Callable,
    metadata: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Skip unchanged files and index changed PDFs through the same backend path."""
    if not force_reindex and not store.needs_document(source_key, version, path):
        return {"state": "unchanged", "source_key": source_key, "chunks": 0}

    try:
        index_document = getattr(store, "index_document", None)
        if index_document is not None:
            kwargs: dict[str, Any] = {"extract_pages": extract_pages}
            if progress is not None and _accepts_keyword(index_document, "progress"):
                kwargs["progress"] = progress
            result = index_document(
                KnowledgeDocument(
                    source_key=source_key,
                    paper=paper,
                    path=Path(path),
                    version=None if version is None else str(version),
                    metadata=metadata or {},
                ),
                **kwargs,
            )
        else:
            result = store.index_pdf(
                paper,
                source_key,
                version,
                path,
                extract_pages=extract_pages,
            )
    except Exception as error:
        record_failure = getattr(store, "record_document_failure", None)
        if record_failure is not None:
            record_failure(
                paper,
                source_key,
                version,
                state="failed",
                detail=str(error),
                path=path,
            )
        raise

    return {
        **result,
        "state": "indexed" if result.get("chunks") else "empty",
        "source_key": source_key,
    }


def _accepts_keyword(function: Callable, name: str) -> bool:
    try:
        parameters = signature(function).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    )
