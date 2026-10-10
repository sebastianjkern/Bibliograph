"""Read-only, format-aware draft checking workflow."""

import atexit
import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from ..pipeline.retrieval import DEFAULT_SEARCH_LIMIT, search_claim
from ..pipeline.timing import format_workflow_timings
from ..render import render_check
from .progress import StageProgress

_WORKER_BACKEND = None
_WORKER_TOOLS: dict[str, Any] = {}


def _close_check_worker() -> None:
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.close()


def _initialize_check_worker(settings, no_llm, disabled_stages) -> None:
    global _WORKER_BACKEND, _WORKER_TOOLS

    from adapters.providers.model_runtimes.registry import build_embedding

    from ..adapters.ikarus import IkarusBackend
    from ..bootstrap import _llm_tools

    embedding = build_embedding(settings)
    _WORKER_BACKEND = IkarusBackend(
        settings["db"],
        mode="read",
        embedding=embedding,
        classification_enabled=settings.get("classification", {}).get("enabled", False),
    )
    _WORKER_TOOLS = _llm_tools(
        settings,
        disabled=no_llm,
        disabled_stages=disabled_stages,
    )
    atexit.register(_close_check_worker)


def _check_claim_in_worker(claim, limit, min_score, enrich):
    if _WORKER_BACKEND is None:
        raise RuntimeError("Draft-check worker was not initialized")
    return search_claim(
        claim,
        backend=_WORKER_BACKEND,
        limit=limit,
        min_score=min_score,
        expand=_WORKER_TOOLS.get("expand"),
        rerank=_WORKER_TOOLS.get("rerank"),
        select_evidence=_WORKER_TOOLS.get("select_evidence"),
        explain=_WORKER_TOOLS.get("explain"),
        enrich=enrich,
        top_only=True,
    )


def check(
    draft: str | Path,
    *,
    backend,
    llm_tools: Mapping[str, Any],
    limit: int = DEFAULT_SEARCH_LIMIT,
    min_score: float = 0.0,
    show_progress: bool = True,
    enrich: bool = True,
    workers: int = 1,
    worker_settings: Mapping[str, Any] | None = None,
    no_llm: bool = False,
    disabled_stages: tuple[str, ...] = (),
    verbose: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    claims = parse_draft_file(draft)
    progress_cm = _progress_context(show_progress)
    with progress_cm as progress:
        task_id = None
        if progress is not None and claims:
            task_id = progress.add_task("Checking draft", total=len(claims))
        stages = (
            StageProgress(progress, task_id)
            if progress is not None and task_id is not None
            else None
        )
        if worker_settings is not None and workers > 1 and claims:
            results = _check_claims_in_processes(
                claims,
                settings=worker_settings,
                workers=workers,
                no_llm=no_llm,
                disabled_stages=disabled_stages,
                limit=limit,
                min_score=min_score,
                enrich=enrich,
                progress=progress,
                task_id=task_id,
                stages=stages,
            )
        else:
            results = []
            for claim in claims:
                def update_progress(description: str) -> None:
                    if stages is not None:
                        stages.update(f"[{len(results) + 1}/{len(claims)}] {description}")

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
                if task_id is not None and progress is not None:
                    progress.advance(task_id)
                    progress.refresh()
        for result in results:
            for item in result["items"]:
                item["claim_synthesis"] = result.get("synthesis")
        items = [item for result in results for item in result["items"]]
        if progress is not None and results:
            totals: dict[str, dict[str, float | int]] = {}
            for result in results:
                for stage, sample in result.get("timings", {}).items():
                    aggregate = totals.setdefault(stage, {"seconds": 0.0, "calls": 0})
                    aggregate["seconds"] += sample["seconds"]
                    aggregate["calls"] += sample["calls"]
            if totals:
                progress.console.print(
                    f"Aggregate workflow work across {len(results)} claims "
                    "(parallel claim durations can overlap):\n"
                    f"{format_workflow_timings(totals, verbose=verbose)}"
                )
        if task_id is not None and stages is not None:
            stages.complete("✓ Draft check complete")
    return {
        "claims": claims,
        "results": results,
        "items": items,
        "markdown": render_check(items, verbose=verbose),
    }


def _check_claims_in_processes(
    claims,
    *,
    settings,
    workers,
    no_llm,
    disabled_stages,
    limit,
    min_score,
    enrich,
    progress,
    task_id,
    stages,
):
    results = [None] * len(claims)
    worker_count = min(workers, len(claims))
    if progress is not None:
        progress.console.print(
            f"Running {len(claims)} claims in {worker_count} isolated worker process(es)"
        )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_check_worker,
        initargs=(dict(settings), no_llm, tuple(disabled_stages)),
    ) as executor:
        futures = {
            executor.submit(_check_claim_in_worker, claim, limit, min_score, enrich): index
            for index, claim in enumerate(claims)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if task_id is not None and progress is not None:
                if stages is not None:
                    stages.update(
                        f"[{completed}/{len(claims)}] Parallel claim retrieval · "
                        f"{completed} complete, {len(claims) - completed} remaining"
                    )
                progress.advance(task_id)
                progress.refresh()
    return results


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
