"""Read-only single-claim search workflow."""

from collections.abc import Mapping
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

from ..pipeline.retrieval import search_claim_with_ragtime
from ..render import render_search


def search(
    claim: str,
    *,
    backend,
    llm_tools: Mapping[str, Any],
    limit: int = 5,
    min_score: float = 0.0,
    show_progress: bool = True,
    enrich: bool = True,
) -> dict[str, Any]:
    progress_cm = _progress_context(show_progress)
    with progress_cm as progress:
        def update_progress(description: str) -> None:
            if progress is None:
                return
            completed = _stage_progress(description)
            progress.update(
                task_id,
                description=description,
                completed=completed,
                refresh=True,
            )

        task_id = (
            progress.add_task("Preparing search", total=5)
            if progress is not None
            else None
        )
        result = search_claim_with_ragtime(
            {"text": claim, "line_start": None, "citation_keys": (), "source_format": None},
            backend=backend,
            limit=limit,
            min_score=min_score,
            expand=llm_tools.get("expand"),
            rerank=llm_tools.get("rerank"),
            progress=update_progress,
            select_evidence=llm_tools.get("select_evidence"),
            explain=llm_tools.get("explain"),
            enrich=enrich,
        )
        items = result["items"]
        if task_id is not None:
            progress.update(
                task_id,
                completed=5,
                description="Search complete",
                refresh=True,
            )
    return {"result": result, "items": items, "markdown": render_search(items)}


def _stage_progress(description: str) -> float:
    ratio = next((part for part in description.split() if "/" in part), None)
    fraction = 0.0
    if ratio is not None:
        try:
            numerator, denominator = ratio.split("/", 1)
            fraction = int(numerator) / max(1, int(denominator))
        except ValueError:
            fraction = 0.0
    if description.startswith("Expanding queries"):
        return fraction
    if description.startswith("Embedding"):
        return 2.0
    if description.startswith("Retrieving evidence"):
        return 2.0 + max(0.0, fraction - (1.0 / max(1, int(ratio.split("/")[1]))))
    if description.startswith("Reranking candidates ·"):
        return 3.0 + fraction
    if description.startswith("Reranking"):
        return 3.0
    if description.startswith("Enriching finalists"):
        return 4.0 + fraction
    return 0.0


def _progress_context(show_progress: bool):
    if not show_progress:
        from contextlib import nullcontext

        return nullcontext(None)
    return Progress(
        SpinnerColumn(style="cyan"),
        BarColumn(bar_width=28),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("[bold cyan]{task.description}"),
        console=Console(stderr=True),
        transient=False,
    )
