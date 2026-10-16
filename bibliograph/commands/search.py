"""Read-only single-claim search workflow."""

import re
from collections.abc import Mapping
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from ..logging_utils import stderr_console
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
                line = Text()
                line.append("✓ ", style="green")
                if isinstance(message, Text):
                    line.append_text(message)
                else:
                    line.append_text(_style_relation_counts(message))
                progress.console.print(line)

        def report_stage(description: str) -> None:
            if progress is None or task_id is None:
                return
            if description.startswith("Phase "):
                gap_prefix = " · Check evidence gaps · "
                if gap_prefix in description:
                    heading, details = description.split(gap_prefix, 1)
                    line = Text()
                    line.append(heading + " · Check evidence gaps", style="bold cyan")
                    line.append(" · ", style="dim")
                    line.append(details)
                    progress.console.print(line)
                else:
                    progress.console.print(Text(description, style="bold cyan"))
            elif description.startswith("Review assessments ·"):
                progress.console.print(Text(f"  {description}"))
            elif description.startswith(('"', "Follow-up query ")):
                line = Text()
                line.append("  ↳ ", style="dim")
                line.append(description)
                progress.console.print(line)
            elif description.startswith(
                (
                    "Retrieved initial evidence",
                    "Retrieved follow-up evidence",
                    "Assessed ",
                    "Reviewed ",
                    "Extended context for",
                    "Generated ",
                    "Selected passages",
                    "Selected final evidence",
                    "Synthesis generated",
                )
            ):
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
            rounds = int(result.get("refinement_rounds", 0))
            round_label = "round" if rounds == 1 else "rounds"
            complete_line = Text()
            complete_line.append("Complete", style="bold green")
            complete_line.append(
                f" · {result['timings'].get('total', {}).get('seconds', 0.0):.2f}s"
                f" · {rounds} additional search {round_label}"
            )
            progress.console.print(complete_line, highlight=False)
            timings = result.get("timings", {})
            if timings:
                progress.console.print(
                    format_workflow_timings(timings, verbose=verbose),
                    highlight=False,
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


def _style_relation_counts(message: str) -> Text:
    styles = {
        "supporting": "green",
        "partial": "yellow",
        "contradicting": "red",
        "mixed": "yellow",
        "unresolved": "dim",
    }
    pattern = re.compile(r"\b(\d+) (supporting|partial|contradicting|mixed|unresolved)\b")
    line = Text()
    position = 0
    for match in pattern.finditer(message):
        line.append(message[position : match.start()])
        style = styles[match.group(2)] if int(match.group(1)) else "dim"
        line.append(match.group(), style=style)
        position = match.end()
    line.append(message[position:])
    return line


def _assessment_progress_text(message: str, suffix: str = "") -> Text:
    """Render relation totals with dim context and colors matching the result view."""

    styles = {
        "support": "green",
        "contradict": "red",
        "mixed": "yellow",
        "unresolved": "cyan",
    }
    pattern = re.compile(r"\b(\d+)\s+(support|contradict|mixed|unresolved)\b")
    line = Text()
    position = 0
    for match in pattern.finditer(message):
        line.append(message[position : match.start()], style="dim")
        line.append(match.group(1), style=styles[match.group(2)])
        line.append(" " + match.group(2), style=styles[match.group(2)])
        position = match.end()
    line.append(message[position:], style="dim")
    line.append(suffix)
    return line


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
    if description.startswith("Context extension"):
        return description.replace("Context extension", "Extending local context", 1)
    if description.startswith("Query extension"):
        return description.replace("Query extension", "Planning follow-up queries", 1)
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
        console=stderr_console(),
        transient=True,
    )
