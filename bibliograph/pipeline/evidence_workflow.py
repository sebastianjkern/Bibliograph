"""LangGraph workflow for iterative, evidence-grounded retrieval."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from inspect import signature
from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from ..domain import Claim, ScoredChunk
from ..knowledge import QueryRequest
from ..text_matching import contains_text
from .retrieval import (
    DEFAULT_SEARCH_LIMIT,
    ContextFor,
    Expand,
    Explain,
    ProgressUpdate,
    Rerank,
    SelectEvidence,
    _limit_hits,
    _run_expander,
    _run_reranker,
    enrich_hits,
    initial_retrieval_limit,
)
from .timing import TimingRecorder

RefineQueries = Callable[[str, Sequence[str]], list[str]]

_COMPONENT_GAP_PATTERNS = (
    re.compile(
        r"\b(?:does|do|did) not\s+(?:directly\s+)?"
        r"(?:establish|demonstrate|show|support|quantif\w*|measure|link|relate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not|never)\s+(?:directly\s+)?"
        r"(?:established|demonstrated|shown|supported|quantified|measured)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot|can't)\s+(?:establish|demonstrate|show|support|quantif\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:direct|clear|specific)\s+(?:evidence|measurement|comparison)\b",
        re.IGNORECASE,
    ),
)
_COMPONENT_PARTIAL_MARKERS = re.compile(
    r"\b(?:narrower|limited to|specific (?:districts|markets|settlements|groups)|"
    r"only (?:for|among|in)|in part|partially|while .*? shows|but|however|although|yet)\b",
    re.IGNORECASE,
)
_COMPONENT_POSITIVE_MARKERS = re.compile(
    r"\b(?:shows?|reports?|finds?|estimates?|quantif\w*|indicates?|describes?|states?)\b",
    re.IGNORECASE,
)


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
    timings: TimingRecorder | None = None,
):
    """Build a bounded retrieve-context-assess-refine LangGraph workflow."""
    if max_refinements < 0 or max_context_extensions < 0:
        raise ValueError("workflow iteration limits cannot be negative")

    context_lookup = _context_lookup(backend)
    retrieve_request = getattr(backend, "retrieve_request", None)
    assessment_pass = 0

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    def plan(state: EvidenceState) -> dict:
        claim_text = state["claim"]["text"].strip()
        report("Phase 1/5 · Prepare claim and search query")
        if expand is not None and timings is not None:
            with timings.measure("query_expansion"):
                proposed = _run_expander(expand, claim_text, progress)
        else:
            proposed = _run_expander(expand, claim_text, progress) if expand is not None else []
        queries = _unique_queries(claim_text, proposed, limit=12)
        variant_count = len(queries)
        label = "variant" if variant_count == 1 else "variants"
        report(f"Prepared claim · {variant_count} query {label} ready")
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
            identifier: list(queries) for identifier, queries in state["candidate_queries"].items()
        }
        score_details = dict(state["score_details"])
        query_traces = list(state["query_traces"])
        pending = state["pending_queries"]
        existing_ids = set(candidates)
        retrieval_kind = "initial" if state["refinement_round"] == 0 else "follow-up"
        report(
            "Phase 2/5 · Retrieve initial evidence"
            if retrieval_kind == "initial"
            else "Phase 4/5 · Retrieve follow-up evidence"
        )
        for index, query in enumerate(pending, start=1):
            report(f"Retrieving evidence · {index}/{len(pending)} queries")
            if retrieve_request is not None:
                retrieve_kwargs = {
                    "limit": initial_retrieval_limit(limit),
                    "include_trace": True,
                }
                if timings is not None and _accepts_keyword(retrieve_request, "timings"):
                    retrieve_kwargs["timings"] = timings
                engine_result = retrieve_request(
                    QueryRequest.from_alternatives(query), **retrieve_kwargs
                )
                hits = list(engine_result.hits)
                details = engine_result.score_details
                trace = engine_result.trace
            else:
                hits, details = backend.retrieve(
                    query,
                    alternatives=(),
                    limit=initial_retrieval_limit(limit),
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
        new_count = len(set(candidates) - existing_ids)
        if retrieval_kind == "initial":
            report(f"Retrieved initial evidence · {len(candidates)} unique candidates")
        else:
            report(
                f"Retrieved follow-up evidence · {new_count} new candidates · "
                f"{len(candidates)} total"
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
        nonlocal assessment_pass
        assessment_pass += 1
        candidates = list(state["candidates"].values())
        if timings is None:
            ranked = (
                _run_reranker(rerank, state["claim"]["text"], candidates, progress)
                if rerank is not None
                else sorted(candidates, key=lambda hit: hit[1], reverse=True)
            )
        else:
            with timings.measure("reranking"):
                ranked = (
                    _run_reranker(rerank, state["claim"]["text"], candidates, progress)
                    if rerank is not None
                    else sorted(candidates, key=lambda hit: hit[1], reverse=True)
                )
        assessment_results = dict(state.get("assessment_results", {}))
        contexts = dict(state.get("contexts", {}))
        context_windows = dict(state.get("context_windows", {}))
        if select_evidence is not None:
            finalists = ranked[: max(5, limit * 2)]
            old_assessments = {
                identifier: _assessment_classification_signature(value)
                for identifier, value in assessment_results.items()
                if value is not None
            }
            component_names = _component_names(assessment_results)
            phase_name = "Assess" if assessment_pass == 1 else "Review"
            if assessment_pass == 1:
                report(f"Phase 3/5 · Assess evidence · {len(finalists)} passages")
            else:
                report(f"Review assessments · {len(finalists)} passages")
            for index, hit in enumerate(finalists, start=1):
                chunk, _score = hit
                if chunk.chunk_id not in contexts:
                    context = context_lookup(chunk)
                    contexts[chunk.chunk_id] = (
                        context if isinstance(context, str) and context.strip() else chunk.text
                    )
                    context_windows.setdefault(chunk.chunk_id, 1)
                context = contexts[chunk.chunk_id]
                report(
                    f"{phase_name} passage {index}/{len(finalists)} · "
                    "waiting for model response"
                )
                kwargs = (
                    {"components": component_names}
                    if component_names and _accepts_keyword(select_evidence, "components")
                    else {}
                )
                if timings is None:
                    selected = select_evidence(
                        state["claim"]["text"], hit, context, **kwargs
                    )
                else:
                    with timings.measure("evidence_assessment"):
                        selected = select_evidence(
                            state["claim"]["text"], hit, context, **kwargs
                        )
                assessment_results[chunk.chunk_id] = _normalise_assessment(selected, context)
                if not component_names:
                    component_names = _component_names(assessment_results)
            reclassified = sum(
                old_assessments.get(identifier)
                != _assessment_classification_signature(value)
                for identifier, value in assessment_results.items()
                if identifier in old_assessments and value is not None
            )
        else:
            reclassified = 0
        relation_counts = {
            relation: sum(
                value is not None and value.get("relation") == relation
                for value in assessment_results.values()
            )
            for relation in ("supports", "partial", "contradicts", "mixed", "insufficient")
        }
        completed_label = "Reviewed" if assessment_pass > 1 else "Assessed"
        report(
            f"{completed_label} {sum(relation_counts.values())} passages · "
            f"{relation_counts['supports']} supporting · "
            f"{relation_counts['partial']} partial · "
            f"{relation_counts['contradicts']} contradicting · "
            f"{relation_counts['mixed']} mixed · "
            f"{relation_counts['insufficient']} unresolved"
            + (f" · {reclassified} reclassified" if assessment_pass > 1 else "")
        )
        return {
            "ranked_hits": ranked,
            "assessment_results": assessment_results,
            "contexts": contexts,
            "context_windows": context_windows,
        }

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
            candidate = candidates.get(identifier)
            if candidate is None:
                continue
            chunk = candidate[0]
            fallback_context = contexts.get(identifier)
            if not isinstance(fallback_context, str) or not fallback_context.strip():
                fallback_context = context_lookup(chunk)
                if not isinstance(fallback_context, str) or not fallback_context.strip():
                    fallback_context = chunk.text
                contexts[identifier] = fallback_context
            next_window = current_window + 1
            expanded = _context_for_window(backend, chunk, next_window, fallback_context)
            if not isinstance(expanded, str) or not expanded.strip():
                expanded = fallback_context
            contexts[identifier] = expanded
            context_windows[identifier] = next_window
            extended += int(expanded != fallback_context)
        extension_round = max(context_windows.values(), default=1) - 1
        report(
            f"Extended context for {extended} unresolved passage(s) · round {extension_round}"
        )
        return {"contexts": contexts, "context_windows": context_windows}

    def refine_queries(state: EvidenceState) -> dict:
        ledger = _evidence_ledger(state)
        claim_text = state["claim"]["text"]
        planning_reason = _assessment_gap_summary(state["assessment_results"])
        report(f"Phase 4/5 · Check evidence gaps · {planning_reason}")
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
                "planning_reason": planning_reason,
            }
        )
        report(
            f"Generated {len(pending)} follow-up queries · "
            f"{planning_reason} · round {state['refinement_round'] + 1}"
        )
        for index, query in enumerate(pending, start=1):
            report(f"Follow-up query {index}/{len(pending)} · {query}")
        return {
            "queries": queries,
            "pending_queries": pending,
            "refinement_round": state["refinement_round"] + 1,
            "refinement_trace": refinement_trace,
        }

    def route_after_refinement(state: EvidenceState) -> str:
        return "retrieve" if state["pending_queries"] else "finish"

    def finish(state: EvidenceState) -> dict:
        report("Phase 5/5 · Select final evidence and synthesize")
        ranked = state.get("ranked_hits", [])
        relation_priority = {
            "supports": 0,
            "contradicts": 0,
            "partial": 1,
            "mixed": 2,
            "insufficient": 3,
        }
        def final_rank_key(hit: ScoredChunk) -> tuple[int, int, float]:
            assessment = state["assessment_results"].get(hit[0].chunk_id)
            if not assessment:
                return (1, relation_priority["insufficient"], -hit[1])
            return (
                0,
                relation_priority.get(str(assessment.get("relation", "insufficient")), 4),
                -hit[1],
            )

        ranked = sorted(ranked, key=final_rank_key)
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
        items = (
            enrich_hits(
                [{**base, "hits": selected_hits[:1] if top_only else selected_hits}],
                context_for=lambda chunk: contexts[chunk.chunk_id],
                explain=explain if select_evidence is None else None,
            )
            if enrich
            else enrich_hits(
                [base],
                context_for=lambda chunk: contexts[chunk.chunk_id],
                top_only=top_only,
            )
        )
        for item in items:
            identifier = item["chunk"].chunk_id
            assessment = state["assessment_results"].get(identifier)
            if assessment is not None:
                item["evidence"] = contexts[identifier]
                relation = str(assessment.get("relation", "insufficient"))
                item["evidence_relation"] = relation
                item["evidence_status"] = relation
                item["matched_excerpt"] = assessment.get("matched_quote") or None
                item["assessment_reason"] = assessment.get("reason", "")
                item["evidence_scope"] = assessment.get("scope", {})
                item["claim_components"] = assessment.get("components", [])
                item["quote_role"] = assessment.get("quote_role", "other")
                item["rationale"] = str(assessment.get("reason", "")).strip()
                if not item["rationale"]:
                    item["rationale"] = {
                        "supports": (
                            "The source passage supports the claim within its stated scope."
                        ),
                        "partial": (
                            "The evidence supports part of the claim but does not establish "
                            "its full scope."
                        ),
                        "contradicts": (
                            "The source passage contradicts the claim within its stated scope."
                        ),
                        "mixed": (
                            "The source passage contains both supporting and contradicting "
                            "evidence."
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
        final_papers = {item["chunk"].paper.zotero_key for item in items}
        assessed_items = [
            item
            for item in items
            if item.get("evidence_relation")
            in {"supports", "partial", "contradicts", "mixed", "insufficient"}
        ]
        assessed_papers = {
            item["chunk"].paper.zotero_key for item in assessed_items
        }
        unresolved_count = sum(
            item.get("evidence_relation") == "insufficient" for item in assessed_items
        )
        not_assessed_count = len(items) - len(assessed_items)
        paper_word = "paper" if len(final_papers) == 1 else "papers"
        assessed_paper_word = "paper" if len(assessed_papers) == 1 else "papers"
        report(
            f"Selected passages · {len(items)} across {len(final_papers)} source {paper_word} · "
            f"{len(assessed_items)} assessed across {len(assessed_papers)} source "
            f"{assessed_paper_word} · {unresolved_count} unresolved · "
            f"{not_assessed_count} not assessed"
        )
        report(f"Synthesis generated · {result['synthesis']['verdict']}")
        report("Evidence workflow complete")
        return {"result": result}

    graph = StateGraph(EvidenceState)

    def timed(stage: str, node):
        if timings is None:
            return node

        def run(state: EvidenceState):
            with timings.measure(stage):
                return node(state)

        return run

    graph.add_node("plan", timed("plan", plan))
    graph.add_node("retrieve", timed("retrieval", retrieve))
    graph.add_node("context", timed("context", gather_context))
    graph.add_node("assess", timed("assessment_and_rerank", assess))
    graph.add_node("extend_context", timed("context_extension", extend_context))
    graph.add_node("refine", timed("query_refinement", refine_queries))
    graph.add_node("finish", timed("finish", finish))
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
    limit: int = DEFAULT_SEARCH_LIMIT,
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
    timings = TimingRecorder()
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
        timings=timings,
    )
    initial_state = cast(EvidenceState, {"claim": claim})
    with timings.measure("total"):
        result = graph.invoke(initial_state)["result"]
    result["timings"] = timings.snapshot()
    return result


def _synthesize_cross_paper(items: Sequence[dict]) -> dict[str, object]:
    """Summarize passage relations by distinct paper without merging quotations."""
    sources: list[dict[str, object]] = []
    paper_relations: dict[str, set[str]] = {}
    paper_scopes: dict[str, dict[str, set[str]]] = {}
    relation_passages = {
        relation: 0 for relation in ("supports", "partial", "contradicts", "mixed", "insufficient")
    }
    scope_keys = ("population", "unit", "outcome", "geography", "time")
    for item in items:
        relation = str(item.get("evidence_relation", "not_assessed"))
        if relation == "not_assessed":
            continue
        chunk = item["chunk"]
        source_id = chunk.paper.zotero_key
        scope = item.get("evidence_scope", {})
        scope = scope if isinstance(scope, dict) else {}
        paper_scope = paper_scopes.setdefault(source_id, {key: set() for key in scope_keys})
        for key in scope_keys:
            value = str(scope.get(key, "")).strip()
            if value:
                paper_scope[key].add(value)
        paper_relations.setdefault(source_id, set()).add(relation)
        if relation in relation_passages:
            relation_passages[relation] += 1
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
            source_id for source_id, relations in paper_relations.items() if relation in relations
        )
        for relation in relation_passages
    }
    scope_variation: dict[str, list[str]] = {}
    if len(paper_scopes) > 1:
        for key in scope_keys:
            values = {value for paper_scope in paper_scopes.values() for value in paper_scope[key]}
            if len(values) > 1:
                scope_variation[key] = sorted(values)

    paper_counts = {relation: len(source_ids) for relation, source_ids in relation_sources.items()}
    if not sources:
        summary = "No independently assessed source passages were available to synthesize."
    else:
        paper_word = "paper" if len(paper_relations) == 1 else "papers"
        summary = (
            f"Evidence from {len(paper_relations)} distinct {paper_word} across "
            f"{len(sources)} assessed passages: {relation_passages['supports']} supporting, "
            f"{relation_passages['partial']} partial, "
            f"{relation_passages['contradicts']} contradicting, "
            f"{relation_passages['mixed']} mixed, and "
            f"{relation_passages['insufficient']} unresolved."
        )
        within_paper_disagreement = any(
            "supports" in relations and "contradicts" in relations
            for relations in paper_relations.values()
        )
        if within_paper_disagreement:
            summary += " Results differ across passages within at least one paper."
        elif paper_counts["supports"] and paper_counts["contradicts"]:
            summary += " Assessed papers disagree."
        if scope_variation:
            summary += " Stated scope differs across papers."
        summary += " This is descriptive, not a pooled estimate."
        summary += " Passages from one paper are not independent confirmations; "
        summary += "independence across papers is unverified."
    component_coverage = _aggregate_claim_components(items)
    unresolved_questions = [
        {
            "component": component["name"],
            "reason": component["reason"],
            "status": component["status"],
        }
        for component in component_coverage
        if component["status"] in {"partial", "unresolved", "contested"}
    ]
    quantitative_findings = [
        source
        for source in sources
        if source.get("matched_quote") and re.search(r"\d", str(source["matched_quote"]))
    ]
    return {
        "summary": summary,
        "conclusion": _overall_conclusion(paper_counts, component_coverage),
        "verdict": _claim_verdict(paper_counts, component_coverage),
        "paper_count": len(paper_relations),
        "passage_count": len(sources),
        "study_count": len(paper_relations),
        "study_count_basis": (
            "counted by distinct source paper; independence across papers is unverified"
        ),
        "claim_coverage": component_coverage,
        "unresolved_questions": unresolved_questions,
        "revision_guidance": (
            "Narrow or qualify: "
            + "; ".join(str(question["component"]) for question in unresolved_questions)
            if unresolved_questions
            else "No material claim qualification was left unresolved by the selected evidence."
        ),
        "quantitative_findings": quantitative_findings,
        "sources": sources,
        "source_ids_by_relation": relation_sources,
        "relation_passages": relation_passages,
        "scope_variation": scope_variation,
    }


def _aggregate_claim_components(items: Sequence[dict]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for item in items:
        source_id = item["chunk"].paper.zotero_key
        components = item.get("claim_components", [])
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict) or not component.get("name"):
                continue
            name = " ".join(str(component["name"]).split())
            key = name.casefold()
            entry = grouped.setdefault(
                key,
                {"name": name, "assessments": [], "sources": set()},
            )
            status = str(component.get("status", "unresolved"))
            entry["sources"].add(source_id)
            reason = " ".join(str(component.get("reason", "")).split())
            entry["assessments"].append(
                {
                    "status": status,
                    "reason": reason,
                    "evidence_quote": " ".join(
                        str(component.get("evidence_quote", "")).split()
                    ),
                    "source_title": item["chunk"].paper.title,
                    "page": item["chunk"].page,
                }
            )

    output = []
    for entry in grouped.values():
        assessments = entry["assessments"]
        statuses = {str(assessment["status"]) for assessment in assessments}
        if "established" in statuses and "contradicted" in statuses:
            status = "contested"
        elif "contradicted" in statuses:
            status = "contradicted"
        elif "partial" in statuses:
            status = "partial"
        elif "established" in statuses:
            status = "established"
        else:
            status = "unresolved"
        relevant_statuses = {
            "contested": {"established", "contradicted"},
            "contradicted": {"contradicted"},
            "partial": {"partial"},
            "established": {"established"},
            "unresolved": {"unresolved"},
        }[status]
        reasons = list(
            dict.fromkeys(
                str(assessment["reason"])
                for assessment in assessments
                if assessment["status"] in relevant_statuses and assessment["reason"]
            )
        )
        component_evidence = list(
            {
                (assessment["source_title"], assessment["page"], assessment["evidence_quote"]): {
                    "title": assessment["source_title"],
                    "page": assessment["page"],
                    "quote": assessment["evidence_quote"],
                }
                for assessment in assessments
                if assessment["status"] in relevant_statuses
                and assessment["evidence_quote"]
            }.values()
        )
        output.append(
            {
                "name": entry["name"],
                "status": status,
                "reason": "; ".join(reasons),
                "evidence": component_evidence,
                "study_count": len(entry["sources"]),
            }
        )
    return output


def _claim_verdict(
    paper_counts: dict[str, int], component_coverage: Sequence[dict[str, object]]
) -> str:
    statuses = {str(component["status"]) for component in component_coverage}
    if "contested" in statuses or ("contradicted" in statuses and "established" in statuses):
        return "Mixed evidence across material claim components"
    contradicted = [
        str(component["name"])
        for component in component_coverage
        if component["status"] == "contradicted"
    ]
    if contradicted:
        return "Not supported — contradicted components: " + ", ".join(contradicted)
    gaps = [
        str(component["name"])
        for component in component_coverage
        if component["status"] in {"partial", "unresolved"}
    ]
    if gaps and statuses.intersection({"established", "partial"}):
        return "Partially supported — qualification needed: " + ", ".join(gaps)
    if gaps:
        return "Not established — unresolved components: " + ", ".join(gaps)
    if component_coverage and statuses == {"established"}:
        if paper_counts.get("contradicts", 0) and any(
            paper_counts.get(relation, 0)
            for relation in ("supports", "partial", "mixed")
        ):
            return "Mixed evidence across assessed passages and claim components"
        if paper_counts.get("contradicts", 0):
            return "Not supported by the assessed passages"
        if paper_counts.get("mixed", 0):
            return "Mixed evidence within assessed source papers"
        if paper_counts.get("partial", 0):
            return "Partially supported — passage-level scope remains unresolved"
        return "Supported across the assessed claim components"
    if paper_counts.get("partial", 0):
        return "Partially supported — some assessed passages leave material scope unresolved"
    return _overall_conclusion(paper_counts)


def _overall_conclusion(
    paper_counts: dict[str, int], component_coverage: Sequence[dict[str, object]] = ()
) -> str:
    supporting = paper_counts.get("supports", 0)
    partial = paper_counts.get("partial", 0)
    contradicting = paper_counts.get("contradicts", 0)
    mixed = paper_counts.get("mixed", 0)
    if not any((supporting, partial, contradicting, mixed)):
        return "The retrieved passages do not establish a conclusion about the claim."
    if contradicting and (supporting or partial or mixed):
        return (
            "Overall, the assessed evidence is mixed: some source passages support the claim "
            "within their stated scope, while others contradict it or report mixed findings."
        )
    if contradicting:
        return "Overall, the assessed evidence weighs against the claim within the reported scopes."
    gaps = [
        str(component["name"])
        for component in component_coverage
        if component["status"] in {"partial", "unresolved", "contested"}
    ]
    if gaps and supporting:
        return "Partially supported; unresolved claim components: " + ", ".join(gaps) + "."
    if partial:
        return (
            "Overall, the evidence is partial: the assessed passages support only part of the "
            "claim or do not establish its full scope."
        )
    if mixed:
        return "Overall, at least one source reports both supporting and contradicting evidence."
    if partial:
        return (
            "Overall, the assessed evidence supports the claim in part, but does not establish "
            "its full scope."
        )
    return (
        "Overall, the assessed evidence supports the claim within the sources' reported scopes; "
        "it should not be generalized beyond those settings."
    )


def _unresolved_candidate_ids(
    assessments: dict[str, dict[str, object] | None],
) -> list[str]:
    return [
        identifier
        for identifier, assessment in assessments.items()
        if assessment is None or assessment.get("relation") in {"partial", "insufficient", "mixed"}
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
        components = assessment.get("components", [])
        if isinstance(components, list):
            gaps = [
                f"{component.get('name')}: {component.get('status')}"
                + (
                    f" ({component.get('reason')})"
                    if component.get("reason")
                    else ""
                )
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


def _normalise_assessment(selection, context: str) -> dict[str, object]:
    if selection is None:
        return {
            "relation": "insufficient",
            "matched_quote": "",
            "reason": "",
            "scope": {},
            "components": [],
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
            "components": [],
        }
    relation = str(selection.get("relation", "insufficient"))
    quote_directness = str(selection.get("quote_directness", "unrelated"))
    quote = " ".join(str(selection.get("matched_quote", "")).split())

    if relation not in {"supports", "partial", "contradicts", "mixed"}:
        relation = "insufficient"
        quote = ""
    elif quote_directness != "direct" or not quote or not contains_text(context, quote):
        relation = "insufficient"
        quote = ""
    raw_scope = selection.get("scope", {})
    raw_components = selection.get("components", [])
    components = []
    if isinstance(raw_components, list):
        seen: set[str] = set()
        allowed = {"established", "partial", "unresolved", "contradicted"}
        for component in raw_components:
            if not isinstance(component, dict):
                continue
            name = " ".join(str(component.get("name", "")).split())
            status = str(component.get("status", "unresolved")).casefold()
            evidence_quote = " ".join(str(component.get("evidence_quote", "")).split())
            if not name or name.casefold() in seen or status not in allowed:
                continue
            component_reason = " ".join(str(component.get("reason", "")).split())
            if status == "established" and not contains_text(context, evidence_quote):
                status = "unresolved"
                component_reason = "No exact source quote was supplied for this component."
            components.append(
                {
                    "name": name,
                    "status": status,
                    "reason": component_reason,
                    "evidence_quote": evidence_quote,
                }
            )
            seen.add(name.casefold())
    for component in components:
        if component["status"] != "established":
            continue
        component_reason = str(component["reason"])
        has_explicit_gap = any(
            pattern.search(component_reason) for pattern in _COMPONENT_GAP_PATTERNS
        )
        if relation == "insufficient" or has_explicit_gap:
            has_positive_basis = bool(_COMPONENT_POSITIVE_MARKERS.search(component_reason))
            has_scope_qualification = bool(
                _COMPONENT_PARTIAL_MARKERS.search(component_reason)
            )
            component["status"] = (
                "partial"
                if relation != "insufficient" and has_positive_basis and has_scope_qualification
                else "unresolved"
            )
            if not component_reason:
                component["reason"] = "The passage does not establish this component."
    component_statuses = {component["status"] for component in components}
    if relation == "supports" and component_statuses.intersection({"partial", "unresolved"}):
        relation = "partial"
    if relation in {"supports", "partial"} and "contradicted" in component_statuses:
        relation = "mixed" if "established" in component_statuses else "contradicts"
    reason = str(selection.get("reason", ""))
    if relation != str(selection.get("relation", "insufficient")):
        qualifiers = [
            f"{component['name']}: {component['reason'] or component['status']}"
            for component in components
            if component["status"] in {"partial", "unresolved", "contradicted"}
        ]
        if qualifiers:
            detail = "Component check qualifies the passage-level verdict: " + "; ".join(
                qualifiers
            )
            reason = f"{reason} {detail}".strip()
    return {
        "relation": relation,
        "matched_quote": quote,
        "reason": reason,
        "scope": raw_scope if isinstance(raw_scope, dict) else {},
        "quote_role": str(selection.get("quote_role", "other")),
        "quote_directness": quote_directness,
        "components": components,
    }


def _assessment_classification_signature(assessment: dict[str, object]) -> tuple:
    """Return only verdict labels, excluding explanatory wording."""
    components = assessment.get("components", [])
    component_labels = tuple(
        sorted(
            (str(component.get("name", "")).casefold(), str(component.get("status", "")))
            for component in components
            if isinstance(component, dict)
        )
    ) if isinstance(components, list) else ()
    return str(assessment.get("relation", "insufficient")), component_labels


def _component_names(assessments: dict[str, dict[str, object] | None]) -> list[str]:
    for assessment in assessments.values():
        if assessment is None:
            continue
        components = assessment.get("components", [])
        if isinstance(components, list):
            names = [
                str(component.get("name", ""))
                for component in components
                if isinstance(component, dict) and component.get("name")
            ]
            if names:
                return names
    return []


def _assessment_gap_summary(
    assessments: dict[str, dict[str, object] | None],
) -> str:
    gaps: dict[str, str] = {}
    for assessment in assessments.values():
        if assessment is None:
            continue
        components = assessment.get("components", [])
        if isinstance(components, list):
            for component in components:
                if not isinstance(component, dict) or component.get("status") not in {
                    "partial",
                    "unresolved",
                    "contradicted",
                }:
                    continue
                name = str(component.get("name", "claim component"))
                reason = str(component.get("reason", ""))
                gaps.setdefault(name.casefold(), f"{name}: {reason}".rstrip(": "))
    if gaps:
        return "; ".join(gaps.values())
    reasons = [
        str(assessment.get("reason", "")).strip()
        for assessment in assessments.values()
        if assessment is not None
        and assessment.get("relation") in {"partial", "insufficient", "mixed", "contradicts"}
        and str(assessment.get("reason", "")).strip()
    ]
    return reasons[0] if reasons else "No specific evidence gap identified"


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
