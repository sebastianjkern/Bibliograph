"""LangGraph workflow for iterative, evidence-grounded retrieval."""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Annotated, Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..domain import Claim, ScoredChunk, citation_label
from ..knowledge import QueryRequest
from .assessment_cache import assessment_cache_key
from .evidence_assessment import (
    _assessment_classification_signature,
    _assessment_gap_summary,
    _component_names,
    _normalise_assessment,
    _unavailable_assessment,
)
from .evidence_synthesis import _synthesize_cross_paper
from .evidence_workflow_utils import (
    _accepts_keyword,
    _context_for_window,
    _context_lookup,
    _evidence_ledger,
    _unique_queries,
    _unresolved_candidate_ids,
)
from .retrieval import (
    DEFAULT_SEARCH_LIMIT,
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


class _RetrievalBatchState(TypedDict):
    tasks: list[dict[str, Any]]
    results: Annotated[dict[int, dict[str, Any]], operator.or_]


class _RetrievalTask(TypedDict):
    index: int
    query: str
    query_vector: Sequence[float] | None


class _AssessmentBatchState(TypedDict):
    tasks: list[dict[str, Any]]
    results: Annotated[dict[str, dict[str, object]], operator.or_]
    cache_hits: Annotated[list[bool], operator.add]


class _AssessmentTask(TypedDict):
    claim: str
    index: int
    total: int
    hit: ScoredChunk
    context: str
    components: Sequence[str]
    phase_name: str
    on_complete: Callable[[], None]


class _RerankBatchState(TypedDict):
    tasks: list[dict[str, Any]]
    results: Annotated[dict[int, list[ScoredChunk]], operator.or_]
    cacheable: Annotated[dict[int, bool], operator.or_]


class _RerankTask(TypedDict):
    claim: str
    index: int
    total: int
    hits: list[ScoredChunk]
    on_complete: Callable[[int], None]


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
    assessment_workers: int = 4,
    rerank_workers: int = 4,
    timings: TimingRecorder | None = None,
):
    """Build a bounded retrieve-context-assess-refine LangGraph workflow."""
    if max_refinements < 0 or max_context_extensions < 0:
        raise ValueError("workflow iteration limits cannot be negative")
    if assessment_workers < 1:
        raise ValueError("assessment_workers must be positive")
    if rerank_workers < 1:
        raise ValueError("rerank_workers must be positive")

    context_lookup = _context_lookup(backend)
    retrieve_request = getattr(backend, "retrieve_request", None)
    workflow_cache_factory = getattr(backend, "workflow_cache", None)
    cache_is_configured = any(
        getattr(tool, identity, None)
        for tool, identity in (
            (rerank, "rerank_cache_identity"),
            (select_evidence, "assessment_cache_identity"),
        )
    )
    workflow_cache = (
        workflow_cache_factory()
        if callable(workflow_cache_factory) and cache_is_configured
        else None
    )
    assessment_pass = 0
    assessment_provider_failed = Event()

    def retrieve_query(task: _RetrievalTask) -> dict:
        query = task["query"]
        if retrieve_request is not None:
            retrieve_kwargs = {
                "limit": initial_retrieval_limit(limit),
                "include_trace": True,
            }
            if timings is not None and _accepts_keyword(retrieve_request, "timings"):
                retrieve_kwargs["timings"] = timings
            if task.get("query_vector") is not None:
                retrieve_kwargs["query_vector"] = task["query_vector"]
            engine_result = retrieve_request(
                QueryRequest.from_alternatives(query), **retrieve_kwargs
            )
            result = {
                "hits": list(engine_result.hits),
                "details": engine_result.score_details,
                "trace": engine_result.trace,
            }
        else:
            hits, details = backend.retrieve(
                query,
                alternatives=(),
                limit=initial_retrieval_limit(limit),
            )
            result = {"hits": hits, "details": details, "trace": None}
        return {"results": {task["index"]: result}}

    def dispatch_retrieval(state: _RetrievalBatchState):
        return [Send("retrieve_query", task) for task in state["tasks"]]

    retrieval_graph = StateGraph(_RetrievalBatchState)
    retrieval_graph.add_node("retrieve_query", retrieve_query)
    retrieval_graph.add_conditional_edges(START, dispatch_retrieval)
    retrieval_graph.add_edge("retrieve_query", END)
    retrieval_graph = retrieval_graph.compile()

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    def assess_passage(task: _AssessmentTask) -> dict:
        hit = task["hit"]
        chunk, _score = hit
        key = None
        cache_identity = getattr(select_evidence, "assessment_cache_identity", None)
        if workflow_cache is not None and cache_identity:
            key = assessment_cache_key(
                {
                    "identity": cache_identity,
                    "claim": task["claim"],
                    "chunk_id": chunk.chunk_id,
                    "citation": citation_label(chunk.paper),
                    "page": chunk.page,
                    "evidence_role": chunk.evidence_role,
                    "content_kind": chunk.content_kind,
                    "context": task["context"],
                    "components": list(task["components"]),
                }
            )
            cached = workflow_cache.get(key)
            if cached is not None:
                task["on_complete"]()
                return {"results": {chunk.chunk_id: cached}, "cache_hits": [True]}

        if assessment_provider_failed.is_set():
            return {
                "results": {chunk.chunk_id: _unavailable_assessment()},
                "cache_hits": [False],
            }

        kwargs = (
            {"components": task["components"]}
            if task["components"] and _accepts_keyword(select_evidence, "components")
            else {}
        )
        if timings is None:
            selected = select_evidence(task["claim"], hit, task["context"], **kwargs)
        else:
            with timings.measure("evidence_assessment"):
                selected = select_evidence(task["claim"], hit, task["context"], **kwargs)
        cacheable = not (
            isinstance(selected, dict) and selected.get("__cacheable__", True) is False
        )
        if isinstance(selected, dict) and "__cacheable__" in selected:
            selected = {name: value for name, value in selected.items() if name != "__cacheable__"}
        assessment = _normalise_assessment(selected, task["context"])
        if assessment.get("relation") == "unavailable":
            assessment_provider_failed.set()
        if key is not None and cacheable:
            workflow_cache.set(key, assessment)
        task["on_complete"]()
        return {"results": {chunk.chunk_id: assessment}, "cache_hits": [False]}

    def dispatch_assessments(state: _AssessmentBatchState):
        return [Send("assess_passage", task) for task in state["tasks"]]

    assessment_graph = StateGraph(_AssessmentBatchState)
    assessment_graph.add_node("assess_passage", assess_passage)
    assessment_graph.add_conditional_edges(START, dispatch_assessments)
    assessment_graph.add_edge("assess_passage", END)
    assessment_graph = assessment_graph.compile()

    def rerank_batch(task: _RerankTask) -> dict:
        ranked_batch = _run_reranker(rerank, task["claim"], task["hits"], None)
        task["on_complete"](len(task["hits"]))
        return {
            "results": {task["index"]: ranked_batch},
            "cacheable": {task["index"]: getattr(ranked_batch, "cacheable", True) is not False},
        }

    def dispatch_reranking(state: _RerankBatchState):
        return [Send("rerank_batch", task) for task in state["tasks"]]

    reranking_graph = StateGraph(_RerankBatchState)
    reranking_graph.add_node("rerank_batch", rerank_batch)
    reranking_graph.add_conditional_edges(START, dispatch_reranking)
    reranking_graph.add_edge("rerank_batch", END)
    reranking_graph = reranking_graph.compile()

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
        for query in queries[1:]:
            report(f'"{query}"')
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
        query_vectors: dict[str, Sequence[float]] = {}
        batch_embed_queries = getattr(backend, "embed_queries", None)
        if (
            pending
            and callable(retrieve_request)
            and callable(batch_embed_queries)
            and _accepts_keyword(retrieve_request, "query_vector")
        ):
            report(f"Embedding {len(pending)} search queries in one batch")
            if timings is None:
                vectors = batch_embed_queries(pending)
            else:
                with timings.measure("embedding"):
                    vectors = batch_embed_queries(pending)
            if len(vectors) != len(pending):
                raise ValueError("Query embedder returned a different number of vectors")
            query_vectors = dict(zip(pending, vectors, strict=True))
        report(
            "Phase 2/5 · Retrieve initial evidence"
            if retrieval_kind == "initial"
            else "Phase 4/5 · Retrieve follow-up evidence"
        )
        batch_result = retrieval_graph.invoke(
            {
                "tasks": [
                    {
                        "index": index,
                        "query": query,
                        "query_vector": query_vectors.get(query),
                    }
                    for index, query in enumerate(pending)
                ],
                "results": {},
            }
        )
        for index, query in enumerate(pending):
            report(f"Retrieving evidence · {index + 1}/{len(pending)} queries")
            query_result = batch_result["results"][index]
            hits = query_result["hits"]
            details = query_result["details"]
            trace = query_result["trace"]
            query_traces.append({"query": query, "trace": trace})
            for chunk, score in hits:
                if score < min_score:
                    continue
                identifier = chunk.chunk_id
                previous = candidates.get(identifier)
                if previous is None or score > previous[1]:
                    candidates[identifier] = (chunk, score)
                queries_for_candidate = candidate_queries.setdefault(identifier, [])
                if query not in queries_for_candidate:
                    queries_for_candidate.append(query)
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
        # Retrieval branches may return the same chunk for several queries. Keep
        # one candidate per stable chunk id before starting any model work.
        candidates_by_id: dict[str, ScoredChunk] = {}
        for hit in state["candidates"].values():
            identifier = hit[0].chunk_id
            previous = candidates_by_id.get(identifier)
            if previous is None or hit[1] > previous[1]:
                candidates_by_id[identifier] = hit
        candidates = list(candidates_by_id.values())
        rerank_identity = getattr(rerank, "rerank_cache_identity", None)
        rerank_key = None
        candidate_by_id = {hit[0].chunk_id: hit for hit in candidates}
        if workflow_cache is not None and rerank_identity and rerank is not None:
            rerank_key = assessment_cache_key(
                {
                    "identity": rerank_identity,
                    "claim": state["claim"]["text"],
                    "candidates": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "retrieval_score": float(score),
                            "citation": citation_label(chunk.paper),
                            "page": chunk.page,
                            "section": chunk.section,
                            "content_kind": chunk.content_kind,
                            "evidence_role": chunk.evidence_role,
                        }
                        for chunk, score in candidates
                    ],
                }
            )
        cached_ranking = (
            workflow_cache.get_reranking(rerank_key) if rerank_key is not None else None
        )
        ranked = None
        rerank_cacheable = True
        if cached_ranking is not None:
            try:
                if len(cached_ranking) == len(candidate_by_id) and {
                    str(row["chunk_id"]) for row in cached_ranking
                } == set(candidate_by_id):
                    ranked = [
                        (candidate_by_id[str(row["chunk_id"])][0], float(row["score"]))
                        for row in cached_ranking
                    ]
            except (KeyError, TypeError, ValueError):
                ranked = None
        rerank_batches: list[list[ScoredChunk]] = []
        if ranked is not None:
            report(f"Reranking · reused cached order for {len(ranked)} passages")
        else:
            if rerank is None:
                ranked = sorted(candidates, key=lambda hit: hit[1], reverse=True)
            elif candidates:
                batch_size = 10
                rerank_batches = [
                    candidates[start : start + batch_size]
                    for start in range(0, len(candidates), batch_size)
                ]

                def run_reranking_graph():
                    completed_passages = 0
                    completion_lock = Lock()

                    def passage_batch_completed(count: int) -> None:
                        nonlocal completed_passages
                        with completion_lock:
                            completed_passages += count
                            report(
                                f"Reranking passages · {completed_passages}/"
                                f"{len(candidates)} complete"
                            )

                    def execute():
                        report(
                            f"Reranking {len(candidates)} passages in {len(rerank_batches)} batches"
                        )
                        mapped = reranking_graph.invoke(
                            {
                                "tasks": [
                                    {
                                        "claim": state["claim"]["text"],
                                        "index": index,
                                        "total": len(rerank_batches),
                                        "hits": batch,
                                        "on_complete": passage_batch_completed,
                                    }
                                    for index, batch in enumerate(rerank_batches, start=1)
                                ],
                                "results": {},
                                "cacheable": {},
                            },
                            config={"max_concurrency": rerank_workers},
                        )
                        merged = [
                            hit
                            for index in range(1, len(rerank_batches) + 1)
                            for hit in mapped["results"][index]
                        ]
                        return (
                            sorted(merged, key=lambda hit: hit[1], reverse=True),
                            all(mapped["cacheable"].values()),
                        )

                    if timings is None:
                        return execute()
                    with timings.measure("reranking"):
                        return execute()
            else:
                ranked = []
        assessment_results = dict(state.get("assessment_results", {}))
        contexts = dict(state.get("contexts", {}))
        context_windows = dict(state.get("context_windows", {}))
        if select_evidence is not None:
            finalist_count = max(5, limit * 2)
            old_assessments = {
                identifier: _assessment_classification_signature(value)
                for identifier, value in assessment_results.items()
                if value is not None
            }
            component_names = _component_names(assessment_results)
            phase_name = "Assess" if assessment_pass == 1 else "Review"
            provisional = sorted(candidates, key=lambda hit: hit[1], reverse=True)[:finalist_count]
            if assessment_pass == 1:
                report(f"Phase 3/5 · Assess evidence · up to {finalist_count} candidate passages")
            else:
                report(f"Review assessments · up to {finalist_count} candidate passages")

            def prepare(hits: Sequence[ScoredChunk]) -> list[tuple[int, ScoredChunk, str]]:
                prepared: list[tuple[int, ScoredChunk, str]] = []
                for index, hit in enumerate(hits, start=1):
                    chunk, _score = hit
                    if chunk.chunk_id not in contexts:
                        context = context_lookup(chunk)
                        contexts[chunk.chunk_id] = (
                            context if isinstance(context, str) and context.strip() else chunk.text
                        )
                        context_windows.setdefault(chunk.chunk_id, 1)
                    prepared.append((index, hit, contexts[chunk.chunk_id]))
                return prepared

            def assess_many(
                tasks: Sequence[tuple[int, ScoredChunk, str]],
                shared_components: Sequence[str],
            ) -> tuple[dict[str, dict[str, object]], list[bool]]:
                if not tasks:
                    return {}, []
                batch_result = assessment_graph.invoke(
                    {
                        "tasks": [
                            {
                                "claim": state["claim"]["text"],
                                "index": index,
                                "total": len(tasks),
                                "hit": hit,
                                "context": context,
                                "components": shared_components,
                                "phase_name": phase_name,
                                "on_complete": assessment_completed,
                            }
                            for index, hit, context in tasks
                        ],
                        "results": {},
                        "cache_hits": [],
                    },
                    config={"max_concurrency": assessment_workers},
                )
                return batch_result["results"], batch_result["cache_hits"]

            completed_assessments = 0
            assessment_completion_lock = Lock()

            def assessment_completed() -> None:
                nonlocal completed_assessments
                with assessment_completion_lock:
                    completed_assessments += 1
                    report(f"{phase_name} progress · {completed_assessments} passages complete")

            provisional_prepared = prepare(provisional)
            prepared_results: dict[str, dict[str, object]] = {}
            cache_hits: list[bool] = []
            remaining = provisional_prepared
            if not component_names and remaining:
                # Establish shared claim components before fanning out assessments.
                index, hit, context = remaining[0]
                first = assess_passage(
                    {
                        "claim": state["claim"]["text"],
                        "index": index,
                        "total": len(provisional_prepared),
                        "hit": hit,
                        "context": context,
                        "components": (),
                        "phase_name": phase_name,
                        "on_complete": assessment_completed,
                    }
                )
                prepared_results.update(first["results"])
                cache_hits.extend(first["cache_hits"])
                component_names = _component_names(first["results"])
                remaining = remaining[1:]

            assessment_unavailable = any(
                value.get("relation") == "unavailable" for value in prepared_results.values()
            )

            # Assessment of likely finalists and reranking are independent once
            # the component schema is known. Start both branches together. If a
            # later rerank promotes a different passage, assess that remainder
            # below before returning the graph state.
            if ranked is None and callable(rerank) and assessment_unavailable:
                # The first assessment can act as a circuit breaker: don't send
                # more requests to a provider that just failed, but still finish
                # reranking so retrieved evidence can be returned in order.
                ranked, rerank_cacheable = run_reranking_graph()
            elif ranked is None and callable(rerank):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    rank_future = executor.submit(run_reranking_graph)
                    assess_future = executor.submit(assess_many, remaining, component_names)
                    ranked, rerank_cacheable = rank_future.result()
                    more_results, more_cache_hits = assess_future.result()
                prepared_results.update(more_results)
                cache_hits.extend(more_cache_hits)
                assessment_unavailable = any(
                    value.get("relation") == "unavailable" for value in prepared_results.values()
                )
            else:
                if ranked is None:
                    if timings is None:
                        ranked, rerank_cacheable = run_reranking_graph()
                    else:
                        with timings.measure("reranking"):
                            ranked, rerank_cacheable = run_reranking_graph()
                if assessment_unavailable:
                    more_results, more_cache_hits = {}, []
                else:
                    more_results, more_cache_hits = assess_many(remaining, component_names)
                prepared_results.update(more_results)
                cache_hits.extend(more_cache_hits)
                assessment_unavailable = assessment_unavailable or any(
                    value.get("relation") == "unavailable" for value in more_results.values()
                )

            if (
                rerank_key is not None
                and rerank_cacheable
                and len(ranked) == len(candidate_by_id)
                and {hit[0].chunk_id for hit in ranked} == set(candidate_by_id)
            ):
                workflow_cache.set_reranking(
                    rerank_key,
                    [{"chunk_id": hit[0].chunk_id, "score": float(hit[1])} for hit in ranked],
                )

            finalists = ranked[:finalist_count]
            finalist_ids = {hit[0].chunk_id for hit in finalists}
            missing_finalists = [
                hit for hit in finalists if hit[0].chunk_id not in prepared_results
            ]
            if missing_finalists:
                missing_prepared = prepare(missing_finalists)
                if assessment_unavailable:
                    for _index, hit, _context in missing_prepared:
                        prepared_results[hit[0].chunk_id] = _unavailable_assessment()
                else:
                    missing_results, missing_cache_hits = assess_many(
                        missing_prepared, component_names
                    )
                    prepared_results.update(missing_results)
                    cache_hits.extend(missing_cache_hits)

            if assessment_pass == 1:
                report(f"Phase 3/5 · Assess evidence · {len(finalists)} passages")
            else:
                report(f"Review assessments · {len(finalists)} passages")

            # Keep speculative assessments for later candidate decisions and
            # context extension out of state unless the passage made the final
            # globally ranked pool.
            prepared_results = {
                identifier: value
                for identifier, value in prepared_results.items()
                if identifier in finalist_ids
            }
            assessment_results.update(prepared_results)
            for hit in finalists:
                chunk = hit[0]
                if chunk.chunk_id not in contexts:
                    context = context_lookup(chunk)
                    contexts[chunk.chunk_id] = (
                        context if isinstance(context, str) and context.strip() else chunk.text
                    )
                    context_windows.setdefault(chunk.chunk_id, 1)

            if cache_hits:
                report(
                    f"Evidence assessment cache · {sum(cache_hits)} reused · "
                    f"{len(cache_hits) - sum(cache_hits)} model call(s)"
                )

            reclassified = sum(
                old_assessments.get(identifier) != _assessment_classification_signature(value)
                for identifier, value in assessment_results.items()
                if identifier in old_assessments and value is not None
            )
        else:
            reclassified = 0
            if ranked is None:
                ranked, rerank_cacheable = run_reranking_graph()
            if (
                rerank_key is not None
                and rerank_cacheable
                and len(ranked) == len(candidate_by_id)
                and {hit[0].chunk_id for hit in ranked} == set(candidate_by_id)
            ):
                workflow_cache.set_reranking(
                    rerank_key,
                    [{"chunk_id": hit[0].chunk_id, "score": float(hit[1])} for hit in ranked],
                )
        relation_counts = {
            relation: sum(
                value is not None and value.get("relation") == relation
                for value in assessment_results.values()
            )
            for relation in ("supports", "partial", "contradicts", "mixed", "insufficient")
        }
        completed_label = "Reviewed" if assessment_pass > 1 else "Assessed"
        unavailable_count = sum(
            value is not None and value.get("relation") == "unavailable"
            for value in assessment_results.values()
        )
        report(
            f"{completed_label} {sum(relation_counts.values())} passages · "
            f"{relation_counts['supports']} supporting · "
            f"{relation_counts['partial']} partial · "
            f"{relation_counts['contradicts']} contradicting · "
            f"{relation_counts['mixed']} mixed · "
            f"{relation_counts['insufficient']} unresolved"
            + (f" · {unavailable_count} provider unavailable" if unavailable_count else "")
            + (f" · {reclassified} reclassified" if assessment_pass > 1 else "")
        )
        return {
            "ranked_hits": ranked,
            "assessment_results": assessment_results,
            "contexts": contexts,
            "context_windows": context_windows,
        }

    def route_after_assessment(state: EvidenceState) -> str:
        if any(
            assessment is not None and assessment.get("relation") == "unavailable"
            for assessment in state["assessment_results"].values()
        ):
            # A provider outage is a technical failure, not an evidence gap.
            return "finish"
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
        report(f"Extended context for {extended} unresolved passage(s) · round {extension_round}")
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
        if pending:
            report(
                f"Generated {len(pending)} follow-up queries · "
                f"round {state['refinement_round'] + 1}"
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
        assessed_papers = {item["chunk"].paper.zotero_key for item in assessed_items}
        unresolved_count = sum(
            item.get("evidence_relation") == "insufficient" for item in assessed_items
        )
        not_assessed_count = len(items) - len(assessed_items)
        unavailable_count = sum(item.get("evidence_status") == "unavailable" for item in items)
        paper_word = "paper" if len(final_papers) == 1 else "papers"
        assessed_paper_word = "paper" if len(assessed_papers) == 1 else "papers"
        report(
            f"Selected passages · {len(items)} across {len(final_papers)} source {paper_word} · "
            f"{len(assessed_items)} assessed across {len(assessed_papers)} source "
            f"{assessed_paper_word} · {unresolved_count} unresolved · "
            f"{not_assessed_count} not assessed"
            + (f" · {unavailable_count} provider unavailable" if unavailable_count else "")
        )
        if unavailable_count:
            result["assessment_incomplete"] = True
            result["assessment_unavailable_count"] = unavailable_count
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
    assessment_workers: int = 4,
    rerank_workers: int = 4,
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
        assessment_workers=assessment_workers,
        rerank_workers=rerank_workers,
        timings=timings,
    )
    initial_state = cast(EvidenceState, {"claim": claim})
    with timings.measure("total"):
        result = graph.invoke(initial_state)["result"]
    result["timings"] = timings.snapshot()
    return result
