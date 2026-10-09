"""LangGraph workflow for iterative, evidence-grounded retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from inspect import signature
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from ..domain import Claim, ScoredChunk
from ..knowledge import QueryRequest
from .retrieval import (
    ContextFor,
    Expand,
    Explain,
    ProgressUpdate,
    Rerank,
    SelectEvidence,
    _limit_hits,
    _render_subtask_results,
    _run_reranker,
    enrich_hits,
)

RefineQueries = Callable[[str, Sequence[str]], list[str]]


class EvidenceState(TypedDict):
    claim: Claim
    pending_queries: list[str]
    queries: list[str]
    refinement_round: int
    candidates: dict[str, ScoredChunk]
    candidate_queries: dict[str, list[str]]
    score_details: dict[str, dict[str, float]]
    query_traces: list[dict[str, object]]
    contexts: dict[str, str]
    ranked_hits: list[ScoredChunk]
    assessment_results: dict[str, dict[str, object] | None]
    context_windows: dict[str, int]
    refinement_trace: list[dict[str, object]]
    result: dict


def build_evidence_workflow(
    *,
    backend,
    limit: int,
    min_score: float,
    expand: Expand | None,
    refine: RefineQueries | None,
    rerank: Rerank | None,
    select_evidence: SelectEvidence | None,
    explain: Explain | None,
    progress: ProgressUpdate | None,
    enrich: bool,
    top_only: bool,
    one_per_paper: bool,
    max_refinements: int = 1,
    max_context_extensions: int = 1,
):
    """Build a bounded retrieve-context-assess-refine LangGraph workflow."""
    if max_refinements < 0 or max_context_extensions < 0:
        raise ValueError("workflow iteration limits cannot be negative")

    context_lookup = _context_lookup(backend)
    retrieve_request = getattr(backend, "retrieve_request", None)

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    def plan(state: EvidenceState) -> dict:
        claim_text = state["claim"]["text"].strip()
        queries = [claim_text]
        report("Plan query · initial claim search")
        return {
            "pending_queries": queries,
            "queries": queries,
            "refinement_round": 0,
            "candidates": {},
            "candidate_queries": {},
            "score_details": {},
            "query_traces": [],
            "assessment_results": {},
            "contexts": {},
            "ranked_hits": [],
            "context_windows": {},
            "refinement_trace": [],
        }

    def retrieve(state: EvidenceState) -> dict:
        candidates = dict(state["candidates"])
        candidate_queries = {
            identifier: list(queries)
            for identifier, queries in state["candidate_queries"].items()
        }
        score_details = dict(state["score_details"])
        query_traces = list(state["query_traces"])
        pending = state["pending_queries"]
        pass_trace_start = len(query_traces)
        report(f"Retrieve candidates · {len(pending)} queries")
        for index, query in enumerate(pending, start=1):
            report(f"Retrieving evidence · {index}/{len(pending)} queries")
            if retrieve_request is not None:
                engine_result = retrieve_request(
                    QueryRequest.from_alternatives(query),
                    limit=max(10, limit * 4),
                    include_trace=True,
                )
                hits = list(engine_result.hits)
                details = engine_result.score_details
                trace = engine_result.trace
            else:
                hits, details = backend.retrieve(
                    query,
                    alternatives=(),
                    limit=max(10, limit * 4),
                )
                trace = None
            query_traces.append({"query": query, "trace": trace})
            for chunk, score in hits:
                if score < min_score:
                    continue
                identifier = chunk.chunk_id
                previous = candidates.get(identifier)
                if previous is None or score > previous[1]:
                    candidates[identifier] = (chunk, score)
                candidate_queries.setdefault(identifier, []).append(query)
                if identifier in details:
                    score_details[identifier] = dict(details[identifier])
        pass_traces = [
            entry["trace"] for entry in query_traces[pass_trace_start:]
        ]
        pass_hits = sorted(candidates.values(), key=lambda hit: hit[1], reverse=True)
        report(
            _render_subtask_results(
                QueryRequest.from_alternatives(state["claim"]["text"]),
                pass_hits,
                pass_traces,
                query_count=len(pending),
            )
        )
        return {
            "candidates": candidates,
            "candidate_queries": candidate_queries,
            "score_details": score_details,
            "query_traces": query_traces,
            "pending_queries": [],
        }

    def gather_context(state: EvidenceState) -> dict:
        contexts = dict(state.get("contexts", {}))
        context_windows = dict(state.get("context_windows", {}))
        ranked = sorted(state["candidates"].values(), key=lambda hit: hit[1], reverse=True)
        for chunk, _score in ranked[: max(5, limit * 2)]:
            if chunk.chunk_id not in contexts:
                contexts[chunk.chunk_id] = context_lookup(chunk)
                context_windows[chunk.chunk_id] = 1
        report(f"Gathered context · {len(contexts)} passages")
        return {"contexts": contexts, "context_windows": context_windows}

    def assess(state: EvidenceState) -> dict:
        candidates = list(state["candidates"].values())
        ranked = _run_reranker(
            rerank,
            state["claim"]["text"],
            candidates,
            progress,
        ) if rerank is not None else sorted(candidates, key=lambda hit: hit[1], reverse=True)
        assessment_results = dict(state.get("assessment_results", {}))
        if select_evidence is not None:
            for hit in ranked[: max(5, limit * 2)]:
                chunk, _score = hit
                context = state["contexts"].get(chunk.chunk_id, chunk.text)
                selected = select_evidence(state["claim"]["text"], hit, context)
                assessment_results[chunk.chunk_id] = _normalise_assessment(selected, context)
        relation_counts = {
            relation: sum(
                value is not None and value.get("relation") == relation
                for value in assessment_results.values()
            )
            for relation in ("supports", "contradicts", "mixed", "insufficient")
        }
        report(
            "Assessed evidence · "
            f"{sum(relation_counts.values())} passages · "
            f"{relation_counts['supports']} support, "
            f"{relation_counts['contradicts']} contradict, "
            f"{relation_counts['mixed']} mixed, "
            f"{relation_counts['insufficient']} unresolved"
        )
        return {"ranked_hits": ranked, "assessment_results": assessment_results}

    def route_after_assessment(state: EvidenceState) -> str:
        unresolved = _unresolved_candidate_ids(state["assessment_results"])
        if not unresolved:
            if (
                not state["candidates"]
                and (refine is not None or expand is not None)
                and state["refinement_round"] < max_refinements
            ):
                return "refine"
            return "finish"
        if any(
            state["context_windows"].get(identifier, 1) < 1 + max_context_extensions
            for identifier in unresolved
        ):
            return "extend_context"
        if refine is not None and state["refinement_round"] < max_refinements:
            return "refine"
        return "finish"

    def extend_context(state: EvidenceState) -> dict:
        contexts = dict(state["contexts"])
        context_windows = dict(state["context_windows"])
        unresolved = _unresolved_candidate_ids(state["assessment_results"])
        candidates = state["candidates"]
        extended = 0
        for identifier in unresolved:
            current_window = context_windows.get(identifier, 1)
            if current_window >= 1 + max_context_extensions:
                continue
            chunk = candidates[identifier][0]
            next_window = current_window + 1
            expanded = _context_for_window(backend, chunk, next_window, contexts[identifier])
            contexts[identifier] = expanded
            context_windows[identifier] = next_window
            extended += int(expanded != state["contexts"][identifier])
        extension_round = max(context_windows.values(), default=1) - 1
        report(
            f"Context extension · round {extension_round} · "
            f"{extended} passage(s) updated"
        )
        return {"contexts": contexts, "context_windows": context_windows}

    def refine_queries(state: EvidenceState) -> dict:
        ledger = _evidence_ledger(state)
        claim_text = state["claim"]["text"]
        if ledger and refine is not None:
            proposed = refine(claim_text, ledger)
            planning_basis = "assessed evidence ledger"
        elif not ledger and not state["candidates"] and expand is not None:
            proposed = expand(claim_text)
            planning_basis = "no initial candidates; claim-only fallback expansion"
        else:
            proposed = []
            planning_basis = "no actionable evidence gap"
        queries = _unique_queries(
            state["claim"]["text"],
            [*state["queries"], *proposed],
            limit=12,
        )
        already_searched = {query.casefold() for query in state["queries"]}
        pending = [query for query in queries if query.casefold() not in already_searched]
        refinement_trace = list(state.get("refinement_trace", []))
        refinement_trace.append(
            {
                "round": state["refinement_round"] + 1,
                "evidence_ledger": ledger,
                "queries": pending,
                "planning_basis": planning_basis,
            }
        )
        report(
            f"Query extension · round {state['refinement_round'] + 1} · "
            f"{len(pending)} new queries · {planning_basis}"
        )
        return {
            "queries": queries,
            "pending_queries": pending,
            "refinement_round": state["refinement_round"] + 1,
            "refinement_trace": refinement_trace,
        }

    def route_after_refinement(state: EvidenceState) -> str:
        return "retrieve" if state["pending_queries"] else "finish"

    def finish(state: EvidenceState) -> dict:
        ranked = state.get("ranked_hits", [])
        relation_priority = {"supports": 0, "contradicts": 0, "mixed": 1, "insufficient": 2}
        ranked = sorted(
            ranked,
            key=lambda hit: (
                relation_priority.get(
                    str(
                        (state["assessment_results"].get(hit[0].chunk_id) or {}).get(
                            "relation", "insufficient"
                        )
                    ),
                    4,
                ),
                -hit[1],
            ),
        )
        selected_hits = _limit_hits(ranked, limit=limit, one_per_paper=one_per_paper)
        base = {
            "claim": state["claim"],
            "queries": tuple(state["queries"]),
            "hits": selected_hits,
            "score_details": state["score_details"],
        }
        contexts = {
            chunk.chunk_id: state["contexts"].get(chunk.chunk_id, chunk.text)
            for chunk, _score in selected_hits
        }
        items = enrich_hits(
            [{**base, "hits": selected_hits[:1] if top_only else selected_hits}],
            context_for=lambda chunk: contexts[chunk.chunk_id],
            explain=explain if select_evidence is None else None,
        ) if enrich else enrich_hits(
            [base],
            context_for=lambda chunk: contexts[chunk.chunk_id],
            top_only=top_only,
        )
        for item in items:
            identifier = item["chunk"].chunk_id
            if identifier in state["assessment_results"]:
                assessment = state["assessment_results"][identifier]
                item["evidence"] = contexts[identifier]
                relation = (
                    str(assessment.get("relation", "insufficient"))
                    if assessment
                    else "insufficient"
                )
                item["evidence_relation"] = relation
                item["evidence_status"] = relation
                item["matched_excerpt"] = (
                    assessment.get("matched_quote") or None if assessment else None
                )
                item["assessment_reason"] = (
                    assessment.get("reason", "") if assessment else ""
                )
                item["evidence_scope"] = (
                    assessment.get("scope", {}) if assessment else {}
                )
                item["quote_role"] = (
                    assessment.get("quote_role", "other") if assessment else "other"
                )
                item["rationale"] = {
                    "supports": "The source passage supports the claim within its stated scope.",
                    "contradicts": (
                        "The source passage contradicts the claim within its stated scope."
                    ),
                    "mixed": (
                        "The source passage contains both supporting and contradicting evidence."
                    ),
                    "insufficient": (
                        "The passage does not establish a supporting or contradicting relation."
                    ),
                }[relation]
            elif select_evidence is not None:
                item["evidence_status"] = "not_assessed"
        traces = state["query_traces"]
        retrieval_trace = traces[0]["trace"] if len(traces) == 1 else tuple(traces)
        result = {
            **base,
            "items": items,
            "synthesis": _synthesize_cross_paper(items),
            "retrieval_trace": retrieval_trace,
            "query_traces": traces,
            "candidate_queries": state["candidate_queries"],
            "refinement_rounds": state["refinement_round"],
            "refinement_trace": state["refinement_trace"],
            "evidence_ledger": _evidence_ledger(state),
            "context_windows": dict(state["context_windows"]),
        }
        report("Evidence workflow complete")
        return {"result": result}

    graph = StateGraph(EvidenceState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("context", gather_context)
    graph.add_node("assess", assess)
    graph.add_node("extend_context", extend_context)
    graph.add_node("refine", refine_queries)
    graph.add_node("finish", finish)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "context")
    graph.add_edge("context", "assess")
    graph.add_conditional_edges(
        "assess",
        route_after_assessment,
        {
            "extend_context": "extend_context",
            "refine": "refine",
            "finish": "finish",
        },
    )
    graph.add_edge("extend_context", "assess")
    graph.add_conditional_edges(
        "refine",
        route_after_refinement,
        {"retrieve": "retrieve", "finish": "finish"},
    )
    graph.add_edge("finish", END)
    return graph.compile()


def run_evidence_workflow(
    claim: Claim,
    *,
    backend,
    limit: int = 5,
    min_score: float = 0.0,
    expand: Expand | None = None,
    refine: RefineQueries | None = None,
    rerank: Rerank | None = None,
    select_evidence: SelectEvidence | None = None,
    explain: Explain | None = None,
    progress: ProgressUpdate | None = None,
    enrich: bool = True,
    top_only: bool = False,
    one_per_paper: bool = False,
    max_refinements: int = 1,
    max_context_extensions: int = 1,
) -> dict:
    """Execute iterative retrieval and return the existing search result shape."""
    text = claim["text"].strip()
    if not text:
        raise ValueError("A claim is required")
    graph = build_evidence_workflow(
        backend=backend,
        limit=limit,
        min_score=min_score,
        expand=expand,
        refine=refine,
        rerank=rerank,
        select_evidence=select_evidence,
        explain=explain,
        progress=progress,
        enrich=enrich,
        top_only=top_only,
        one_per_paper=one_per_paper,
        max_refinements=max_refinements,
        max_context_extensions=max_context_extensions,
    )
    initial_state = cast(EvidenceState, {"claim": claim})
    return graph.invoke(initial_state)["result"]


def _synthesize_cross_paper(items: Sequence[dict]) -> dict[str, object]:
    """Summarize independently assessed source cards without merging their evidence."""
    sources: list[dict[str, object]] = []
    paper_relations: dict[str, set[str]] = {}
    scope_values: dict[str, set[str]] = {
        key: set() for key in ("population", "unit", "outcome", "geography", "time")
    }
    for item in items:
        relation = item.get("evidence_relation", "not_assessed")
        if relation == "not_assessed":
            continue
        chunk = item["chunk"]
        source_id = chunk.paper.zotero_key
        scope = item.get("evidence_scope", {})
        scope = scope if isinstance(scope, dict) else {}
        for key in scope_values:
            value = str(scope.get(key, "")).strip()
            if value:
                scope_values[key].add(value)
        paper_relations.setdefault(source_id, set()).add(str(relation))
        sources.append(
            {
                "source_id": source_id,
                "title": chunk.paper.title,
                "page": chunk.page,
                "relation": relation,
                "matched_quote": item.get("matched_excerpt"),
                "quote_role": item.get("quote_role", "other"),
                "scope": scope,
            }
        )
    relation_sources = {
        relation: sorted(
            source_id
            for source_id, relations in paper_relations.items()
            if relation in relations
        )
        for relation in ("supports", "contradicts", "mixed", "insufficient")
    }
    support_count = len(relation_sources["supports"])
    contradiction_count = len(relation_sources["contradicts"])
    mixed_count = len(relation_sources["mixed"])
    scope_variation = {
        key: sorted(values) for key, values in scope_values.items() if len(values) > 1
    }
    if not sources:
        summary = "No independently assessed source passages were available to synthesize."
    else:
        summary = (
            f"Across {len(paper_relations)} distinct papers, assessed passages include "
            f"{support_count} supporting, {contradiction_count} contradicting, and "
            f"{mixed_count} mixed source(s)."
        )
        if support_count and contradiction_count:
            summary += " The assessed sources contain disagreement."
        if scope_variation:
            summary += " Stated scope varies across sources."
        summary += " This is a descriptive summary, not a pooled estimate."
    return {
        "summary": summary,
        "sources": sources,
        "source_ids_by_relation": relation_sources,
        "scope_variation": scope_variation,
    }


def _unresolved_candidate_ids(
    assessments: dict[str, dict[str, object] | None],
) -> list[str]:
    return [
        identifier
        for identifier, assessment in assessments.items()
        if assessment is None
        or assessment.get("relation") in {"insufficient", "mixed"}
    ]


def _evidence_ledger(state: EvidenceState) -> list[str]:
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


def _normalise_assessment(selection, context: str) -> dict[str, object]:
    if selection is None:
        return {
            "relation": "insufficient",
            "matched_quote": "",
            "reason": "",
            "scope": {},
        }
    if isinstance(selection, tuple):
        quote, reason = selection
        selection = {
            "relation": "supports",
            "matched_quote": quote,
            "reason": reason,
            "quote_directness": "direct",
        }
    if not isinstance(selection, dict):
        return {
            "relation": "insufficient",
            "matched_quote": "",
            "reason": "",
            "scope": {},
        }
    relation = str(selection.get("relation", "insufficient"))
    quote_directness = str(selection.get("quote_directness", "unrelated"))
    quote = " ".join(str(selection.get("matched_quote", "")).split())
    normalized_context = " ".join(context.split())
    if relation not in {"supports", "contradicts", "mixed"}:
        relation = "insufficient"
        quote = ""
    elif (
        quote_directness != "direct"
        or not quote
        or quote not in normalized_context
    ):
        relation = "insufficient"
        quote = ""
    raw_scope = selection.get("scope", {})
    return {
        "relation": relation,
        "matched_quote": quote,
        "reason": str(selection.get("reason", "")),
        "scope": raw_scope if isinstance(raw_scope, dict) else {},
        "quote_role": str(selection.get("quote_role", "other")),
        "quote_directness": quote_directness,
    }


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
