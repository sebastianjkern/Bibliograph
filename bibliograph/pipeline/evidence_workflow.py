"""LangGraph workflow for iterative, evidence-grounded retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
    assessment_results: dict[str, tuple[str, str] | None]
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
):
    """Build a bounded retrieve-context-assess-refine LangGraph workflow."""
    if max_refinements < 0:
        raise ValueError("max_refinements cannot be negative")

    context_lookup = _context_lookup(backend)
    retrieve_request = getattr(backend, "retrieve_request", None)

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    def plan(state: EvidenceState) -> dict:
        claim_text = state["claim"]["text"].strip()
        alternatives = expand(claim_text) if expand is not None else []
        queries = _unique_queries(claim_text, alternatives, limit=8)
        report(f"Plan query · {len(queries) - 1} alternatives ready")
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
        ranked = sorted(state["candidates"].values(), key=lambda hit: hit[1], reverse=True)
        for chunk, _score in ranked[: max(5, limit * 2)]:
            if chunk.chunk_id not in contexts:
                contexts[chunk.chunk_id] = context_lookup(chunk)
        report(f"Gathered context · {len(contexts)} passages")
        return {"contexts": contexts}

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
                selected = select_evidence(
                    state["claim"]["text"],
                    hit,
                    state["contexts"].get(chunk.chunk_id, chunk.text),
                )
                assessment_results[chunk.chunk_id] = selected
        supported_count = sum(value is not None for value in assessment_results.values())
        report(f"Assessed evidence · {supported_count} supporting excerpts")
        return {"ranked_hits": ranked, "assessment_results": assessment_results}

    def route_after_assessment(state: EvidenceState) -> str:
        if any(value is not None for value in state["assessment_results"].values()):
            return "finish"
        if refine is None:
            return "finish"
        if state["refinement_round"] >= max_refinements:
            return "finish"
        return "refine"

    def refine_queries(state: EvidenceState) -> dict:
        excerpts = []
        ranked = state.get("ranked_hits", [])
        for chunk, _score in ranked[:5]:
            context = state["contexts"].get(chunk.chunk_id, chunk.text)
            excerpts.append(context)
        if refine is None:
            return {
                "pending_queries": [],
                "refinement_round": state["refinement_round"] + 1,
            }
        proposed = refine(state["claim"]["text"], excerpts)
        queries = _unique_queries(
            state["claim"]["text"],
            [*state["queries"], *proposed],
            limit=12,
        )
        already_searched = {query.casefold() for query in state["queries"]}
        pending = [query for query in queries if query.casefold() not in already_searched]
        report(f"Refined queries · {len(pending)} new terms")
        return {
            "queries": queries,
            "pending_queries": pending,
            "refinement_round": state["refinement_round"] + 1,
        }

    def route_after_refinement(state: EvidenceState) -> str:
        return "retrieve" if state["pending_queries"] else "finish"

    def finish(state: EvidenceState) -> dict:
        ranked = state.get("ranked_hits", [])
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
                if assessment is None:
                    item["evidence_status"] = "no_supporting_excerpt_selected"
                    item["matched_excerpt"] = None
                    item["rationale"] = (
                        "No exact supporting excerpt was verified within this context."
                    )
                else:
                    matched_excerpt, _selection_rationale = assessment
                    item["matched_excerpt"] = matched_excerpt
                    item["evidence_status"] = "supporting_excerpt_selected"
                    item["rationale"] = (
                        explain(state["claim"]["text"], item["evidence"])
                        if explain is not None
                        else "The displayed context contains the verified supporting excerpt."
                    )
            elif select_evidence is not None:
                item["evidence_status"] = "not_assessed"
        traces = state["query_traces"]
        retrieval_trace = traces[0]["trace"] if len(traces) == 1 else tuple(traces)
        result = {
            **base,
            "items": items,
            "retrieval_trace": retrieval_trace,
            "query_traces": traces,
            "candidate_queries": state["candidate_queries"],
            "refinement_rounds": state["refinement_round"],
        }
        report("Evidence workflow complete")
        return {"result": result}

    graph = StateGraph(EvidenceState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("context", gather_context)
    graph.add_node("assess", assess)
    graph.add_node("refine", refine_queries)
    graph.add_node("finish", finish)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "context")
    graph.add_edge("context", "assess")
    graph.add_conditional_edges(
        "assess",
        route_after_assessment,
        {"refine": "refine", "finish": "finish"},
    )
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
    )
    initial_state = cast(EvidenceState, {"claim": claim})
    return graph.invoke(initial_state)["result"]


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
