"""Read-only, format-aware draft checking workflow."""

from collections.abc import Mapping
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

from ..pipeline.drafts import parse_draft_file
from ..pipeline.retrieval import search_claim
from ..render import render_check


def check(
    draft: str | Path,
    *,
    backend,
    llm_tools: Mapping[str, Any],
    limit: int = 5,
    min_score: float = 0.0,
    show_progress: bool = True,
    enrich: bool = True,
) -> dict[str, Any]:
    claims = parse_draft_file(draft)
    progress_cm = _progress_context(show_progress)
    with progress_cm as progress:
        task_id = (
            progress.add_task("Checking draft", total=len(claims))
            if progress is not None and claims
            else None
        )
        results = []
        for claim in claims:
            def update_progress(description: str) -> None:
                if progress is not None:
                    progress.update(task_id, description=description, refresh=True)

            result = search_claim(
                claim,
                backend=backend,
                limit=limit,
                min_score=min_score,
                expand=llm_tools.get("expand"),
                rerank=llm_tools.get("rerank"),
                progress=update_progress,
                select_evidence=llm_tools.get("select_evidence"),
                explain=llm_tools.get("explain"),
                enrich=enrich,
                top_only=True,
            )
            results.append(result)
            if task_id is not None:
                progress.advance(task_id)
                progress.refresh()
        items = [item for result in results for item in result["items"]]
        if task_id is not None:
            progress.update(task_id, description="Check complete", refresh=True)
    return {"claims": claims, "results": results, "items": items, "markdown": render_check(items)}


def _progress_context(show_progress: bool):
    if not show_progress:
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
