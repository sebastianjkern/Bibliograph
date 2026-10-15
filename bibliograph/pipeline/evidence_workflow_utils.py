"""Small policies and compatibility helpers used by the evidence workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
from typing import Any, cast

from .retrieval import ContextFor


def _unresolved_candidate_ids(
    assessments: dict[str, dict[str, object] | None],
) -> list[str]:
    return [
        identifier
        for identifier, assessment in assessments.items()
        if assessment is None or assessment.get("relation") in {"partial", "insufficient", "mixed"}
    ]


def _evidence_ledger(state: Mapping[str, Any]) -> list[str]:
    ledger = []
    ranked = state.get("ranked_hits", [])
    for chunk, _score in ranked[:8]:
        assessment = state["assessment_results"].get(chunk.chunk_id) or {}
        relation = str(assessment.get("relation", "not_assessed"))
        quote = str(assessment.get("matched_quote", ""))
        reason = str(assessment.get("reason", ""))
        scope = assessment.get("scope", {})
        scope_text = ""
        if isinstance(scope, dict):
            stated = [f"{key}={value}" for key, value in scope.items() if value]
            if stated:
                scope_text = "; stated scope: " + ", ".join(stated)
        context = " ".join(state["contexts"].get(chunk.chunk_id, chunk.text).split())
        if len(context) > 700:
            context = context[:697].rsplit(" ", 1)[0] + "..."
        page = f"p. {chunk.page}" if chunk.page is not None else "page unknown"
        parts = [
            f"Source {chunk.paper.title}, {page}, chunk {chunk.chunk_id}: {relation}.",
        ]
        if quote:
            parts.append(f"Matched source text: {quote}")
        if reason:
            parts.append(f"Assessment gap/reason: {reason}")
        if scope_text:
            parts.append(scope_text.lstrip("; "))
        components = assessment.get("components", [])
        if isinstance(components, list):
            gaps = [
                f"{component.get('name')}: {component.get('status')}"
                + (f" ({component.get('reason')})" if component.get("reason") else "")
                for component in components
                if isinstance(component, dict)
                and component.get("status") in {"partial", "unresolved", "contradicted"}
            ]
            if gaps:
                parts.append("Claim component coverage gaps: " + "; ".join(gaps))
        parts.append(f"Retrieved context: {context}")
        ledger.append(" ".join(parts))
    return ledger


def _context_for_window(backend: Any, chunk, window: int, fallback: str) -> str:
    context_for = getattr(backend, "context_for", None)
    if not callable(context_for):
        return fallback
    try:
        supports_window = "window" in signature(context_for).parameters
    except (TypeError, ValueError):
        supports_window = False
    if not supports_window:
        return fallback
    expanded = context_for(chunk, window=window)
    return expanded if isinstance(expanded, str) and expanded.strip() else fallback


def _unique_queries(claim: str, proposed: Sequence[str], *, limit: int) -> list[str]:
    queries = [claim]
    seen = {claim.casefold()}
    for candidate in proposed:
        query = " ".join(candidate.split()).strip()
        if query and query.casefold() not in seen:
            queries.append(query)
            seen.add(query.casefold())
        if len(queries) >= limit:
            break
    return queries


def _context_lookup(backend: Any) -> ContextFor:
    context_for = getattr(backend, "context_for", None)
    if callable(context_for):
        return cast(ContextFor, context_for)
    return lambda chunk: chunk.text


def _accepts_keyword(function, name: str) -> bool:
    try:
        parameters = signature(function).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    )
