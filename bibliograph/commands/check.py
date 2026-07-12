"""Read-only, format-aware draft checking workflow."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..pipeline.drafts import parse_draft_file
from ..pipeline.retrieval import enrich_hits, retrieve_claims
from ..render import render_check


def check(
    draft: str | Path,
    *,
    embedding: Mapping[str, Any],
    store,
    llm_tools: Mapping[str, Any],
    limit: int = 5,
    min_score: float = 0.0,
) -> dict[str, Any]:
    claims = parse_draft_file(draft)
    results = retrieve_claims(
        claims,
        embed_queries=embedding["embed_queries"],
        search=store.search,
        limit=limit,
        min_score=min_score,
        rerank=llm_tools.get("rerank"),
    )
    items = enrich_hits(
        results,
        context_for=store.context_for,
        select_evidence=llm_tools.get("select_evidence"),
        explain=llm_tools.get("explain"),
        top_only=True,
    )
    return {"claims": claims, "results": results, "items": items, "markdown": render_check(items)}
