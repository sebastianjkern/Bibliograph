"""Read-only single-claim search workflow."""

from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from ..pipeline.retrieval import DEFAULT_SEARCH_LIMIT, search_claim
from ..pipeline.timing import format_workflow_timings
from ..render import render_search


def search(
    claim: str,
    *,
    backend,
    llm_tools: Mapping[str, Any],
    limit: int = DEFAULT_SEARCH_LIMIT,
    min_score: float = 0.0,
    show_progress: bool = True,
    enrich: bool = True,
    one_per_paper: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    progress_cm = _progress_context(show_progress)
    with progress_cm as progress:

        task_id = progress.add_task("Preparing search") if progress is not None else None
        def print_step(message: str | Text) -> None:
            if progress is not None:
                line = Text("✓ ", style="green")
                line.append_text(
                    message if isinstance(message, Text) else Text(message, style="dim")
                )
                progress.console.print(line)

        def report_stage(description: str) -> None:
            if progress is None or task_id is None:
                return
            if description.startswith("Phase "):
                progress.console.print(f"[bold cyan]{description}[/bold cyan]")
            elif description.startswith("Review assessments ·"):
                progress.console.print(f"[cyan]  {description}[/cyan]")
            elif description.startswith("Follow-up query "):
                line = Text("  ↳ ", style="cyan")
                line.append(description.removeprefix("Follow-up query "), style="dim")
                progress.console.print(line)
            elif description.startswith((
                "Retrieved initial evidence",
                "Retrieved follow-up evidence",
                "Assessed ",
                "Reviewed ",
                "Extended context for",
                "Generated ",
                "Selected passages",
                "Selected final evidence",
                "Synthesis generated",
            )):
                print_step(description)
            progress.update(
                task_id,
                description=_progress_description(description),
                refresh=True,
            )

        result = search_claim(
            {"text": claim, "line_start": None, "citation_keys": (), "source_format": None},
            backend=backend,
            limit=limit,
            min_score=min_score,
            expand=llm_tools.get("expand"),
            refine=llm_tools.get("refine"),
            rerank=llm_tools.get("rerank"),
            progress=report_stage if progress is not None else None,
            select_evidence=llm_tools.get("select_evidence"),
            explain=llm_tools.get("explain"),
            enrich=enrich,
            one_per_paper=one_per_paper,
        )
        items = result["items"]
        if task_id is not None and progress is not None:
            progress.update(
                task_id,
                description="Search complete",
                refresh=True,
            )
            progress.console.print(
                f"[bold green]Complete[/bold green] · "
                f"{result['timings'].get('total', {}).get('seconds', 0.0):.2f}s · "
                f"{result.get('refinement_rounds', 0)} additional search round(s)"
            )
            timings = result.get("timings", {})
            if timings:
                progress.console.print(
                    format_workflow_timings(timings, verbose=verbose)
                )
    return {
        "result": result,
        "items": items,
        "markdown": render_search(
            items,
            claim=claim,
            verbose=verbose,
            synthesis=result.get("synthesis"),
        ),
    }


def _progress_description(description: str) -> str:
    if description.startswith("Phase "):
        return description
    if description.startswith("Assess passage ") or description.startswith("Review passage "):
        return description
    if description.startswith("Plan query"):
        return "Searching the claim"
    if description.startswith("Retrieve candidates"):
        return description.replace("Retrieve candidates", "Retrieving", 1)
    if description.startswith("Retrieved candidates"):
        return description.replace("Retrieved candidates", "Retrieved", 1)
    if description.startswith("Gathered context"):
        return description.replace("Gathered context", "Preparing source context", 1)
    if description.startswith("Reranking candidates"):
        return description.replace("Reranking candidates", "Ranking passages", 1)
    if description.startswith("Assessed evidence"):
        return description.replace("Assessed evidence", "Evidence assessment", 1)
    if description.startswith("Extended context for") or description.startswith("Generated "):
        return description
    if description.startswith("Evidence workflow complete"):
        return "Finalizing evidence results"
    return description


def _progress_context(show_progress: bool):
    if not show_progress:
        from contextlib import nullcontext

        return nullcontext(None)
    return Progress(
        SpinnerColumn(style="cyan"),
        TimeElapsedColumn(),
        TextColumn("[bold cyan]{task.description}"),
        console=Console(stderr=True),
        transient=True,
    )
