"""Read-only single-claim search workflow."""

from collections.abc import Mapping
from typing import Any

from ..pipeline.retrieval import enrich_hits, search_claim
from ..render import render_search


def search(
    claim: str,
    *,
    embedding: Mapping[str, Any],
    store,
    llm_tools: Mapping[str, Any],
    limit: int = 5,
    min_score: float = 0.0,
) -> dict[str, Any]:
    result = search_claim(
        {"text": claim, "line_start": None, "citation_keys": (), "source_format": None},
        embed_queries=embedding["embed_queries"],
        search=store.search,
        limit=limit,
        min_score=min_score,
        rerank=llm_tools.get("rerank"),
    )
    items = enrich_hits(
        [result],
        context_for=store.context_for,
        select_evidence=llm_tools.get("select_evidence"),
        explain=llm_tools.get("explain"),
    )
    return {"result": result, "items": items, "markdown": render_search(items)}
