"""Read-only single-claim search workflow."""

import re
from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from ..pipeline.retrieval import search_claim
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
    one_per_paper: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    progress_cm = _progress_context(show_progress)
    with progress_cm as progress:

        task_id = progress.add_task("Preparing search") if progress is not None else None
        pending_assessment: list[str] = []

        def print_step(message: str | Text) -> None:
            if progress is not None:
                line = Text("✓ ", style="green")
                line.append_text(
                    message if isinstance(message, Text) else Text(message, style="dim")
                )
                progress.console.print(line)

        def flush_assessment(suffix: str = "") -> None:
            if pending_assessment:
                detail = pending_assessment.pop()
                print_step(_assessment_progress_text(detail, suffix))

        def report_stage(description: str) -> None:
            if progress is None or task_id is None:
                return
            if description.startswith("Retrieved candidates"):
                detail = description.removeprefix("Retrieved candidates · ")
                print_step(f"Retrieval pass · {detail}")
            elif description.startswith("Assessed evidence"):
                pending_assessment.append(description.removeprefix("Assessed evidence · "))
            elif description.startswith("Context extension"):
                extension = description.removeprefix("Context extension · ")
                flush_assessment(f" · {extension}")
            elif description.startswith("Query extension"):
                flush_assessment()
                query_plan = description.removeprefix("Query extension · ")
                print_step(f"Query planning · {query_plan}")
            elif description.startswith("Evidence workflow complete"):
                flush_assessment()
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
            assessed = sum(
                item.get("evidence_relation")
                in {"supports", "partial", "contradicts", "mixed"}
                for item in items
            )
            progress.console.print(
                f"[green]✓[/green] Search complete · {assessed} assessed passages · "
                f"{result.get('refinement_rounds', 0)} query extension round(s)"
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


def _assessment_progress_text(detail: str, suffix: str = "") -> Text:
    relation_styles = {
        "support": "bold green",
        "partial": "bold magenta",
        "contradict": "bold red",
        "mixed": "bold yellow",
        "unresolved": "bold cyan",
    }
    text = Text("Evidence assessment · ", style="dim")
    relation_pattern = re.compile(
        r"(?P<count>\d+)\s+(?P<relation>support|partial|contradict|mixed|unresolved)\b"
    )
    cursor = 0
    for match in relation_pattern.finditer(detail):
        text.append(detail[cursor : match.start()], style="dim")
        text.append(match.group("count"), style=relation_styles[match.group("relation")])
        text.append(" " + match.group("relation"), style=relation_styles[match.group("relation")])
        cursor = match.end()
    text.append(detail[cursor:] + suffix, style="dim")
    return text


def _progress_description(description: str) -> str:
    if description.startswith("Plan query"):
        return "Searching the claim"
    if description.startswith("Retrieve candidates"):
        return description.replace("Retrieve candidates", "Retrieving", 1)
    if description.startswith("Retrieved candidates"):
        return description.replace("Retrieved candidates", "Retrieved", 1)
    if description.startswith("Gathered context"):
        return description.replace("Gathered context", "Preparing source context", 1)
    if description.startswith("Reranking candidates"):
        return description.replace("Reranking candidates", "Reranking passages", 1)
    if description.startswith("Assessed evidence"):
        return description.replace("Assessed evidence", "Evidence assessment", 1)
    if description.startswith("Context extension"):
        return description.replace("Context extension", "Extending local context", 1)
    if description.startswith("Query extension"):
        return description.replace("Query extension", "Planning follow-up queries", 1)
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
