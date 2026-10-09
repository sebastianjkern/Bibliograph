"""Injected text retrieval and LLM enrichment pipeline."""

import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import signature

from ..domain import Chunk, Claim, ScoredChunk
from ..knowledge import QueryRequest
from .llm_tasks import heuristic_rerank, template_rationale

EmbedQueries = Callable[[Sequence[str]], list[list[float]]]
Search = Callable[..., list[ScoredChunk]]
ContextFor = Callable[[Chunk], str]
Rerank = Callable[[str, Sequence[ScoredChunk]], list[ScoredChunk]]
Expand = Callable[[str], list[str]]
ProgressUpdate = Callable[[str], None]
SelectEvidence = Callable[[str, ScoredChunk, str], tuple[str, str] | None]
Explain = Callable[[str, str], str]


def _search_claim_with_backend(
    claim: Claim,
    *,
    backend,
    limit: int = 5,
    min_score: float = 0.0,
    expand: Expand | None = None,
    rerank: Rerank | None = None,
    progress: ProgressUpdate | None = None,
    select_evidence: SelectEvidence | None = None,
    explain: Explain | None = None,
    enrich: bool = True,
    top_only: bool = False,
    one_per_paper: bool = False,
    include_trace: bool = False,
) -> dict:
    """Retrieve through the configured backend and retain Bibliograph's citation result contract."""
    text = claim["text"].strip()
    if not text:
        raise ValueError("A claim is required")
    alternatives: list[str] = []
    if expand is not None:
        if progress is not None:
            progress("Plan query · generating alternatives")
        alternatives = _run_expander(expand, text, progress)
        if progress is not None:
            progress(f"Plan query · {len(alternatives)} alternatives ready")
    if progress is not None:
        progress(f"Retrieve candidates · {1 + len(alternatives)} queries")
    request = QueryRequest.from_alternatives(text, alternatives)
    retrieve_request = getattr(backend, "retrieve_request", None)
    if retrieve_request is not None:
        engine_result = retrieve_request(
            request,
            limit=max(limit, 10),
            include_trace=include_trace or progress is not None,
        )
        hits = list(engine_result.hits)
        score_details = dict(engine_result.score_details)
        retrieval_trace = engine_result.trace
    else:
        # Compatibility for injected legacy backends during the adapter migration.
        hits, score_details = backend.retrieve(
            text,
            alternatives=alternatives,
            limit=max(limit, 10),
        )
        retrieval_trace = None
    hits = [hit for hit in hits if hit[1] >= min_score]
    if progress is not None:
        progress(_render_subtask_results(request, hits, retrieval_trace, score_details))
    if rerank is not None:
        if progress is not None:
            progress("Reranking retrieval candidates")
        hits = _run_reranker(rerank, text, hits, progress)
    selected_hits = _limit_hits(hits, limit=limit, one_per_paper=one_per_paper)
    result = {
        "claim": claim,
        "queries": (text, *alternatives),
        "hits": selected_hits,
        "score_details": score_details,
        "retrieval_trace": retrieval_trace,
    }
    if enrich:
        enrichment_hits = selected_hits[:1] if top_only else selected_hits
        result["items"] = enrich_hits(
            [{**result, "hits": enrichment_hits}],
            context_for=lambda chunk: chunk.text,
            select_evidence=select_evidence,
            explain=explain,
        )
    else:
        result["items"] = enrich_hits(
            [result],
            context_for=lambda chunk: chunk.text,
            top_only=top_only,
        )
    return result


def _render_subtask_results(request, hits, trace, score_details) -> str:
    lines = ["Search subtask results:", f"  Query: {request.text}"]
    if request.hypotheses:
        lines.append(f"  Query plan: {len(request.hypotheses)} alternatives")

    if hits:
        lines.append("  Evidence candidates:")
        for chunk, score in hits[:5]:
            paper = chunk.paper
            title = paper.title.strip() or paper.zotero_key
            if "_" in title:
                title = title.replace("_", " ").title()
            location = f", p. {chunk.page}" if chunk.page is not None else ""
            role = f" · {chunk.evidence_role}" if chunk.evidence_role else ""
            components = score_details.get(chunk.chunk_id, {})
            component_text = " · ".join(
                f"{name} {float(components[name]):.2f}"
                for name in ("vector", "lexical", "graph")
                if name in components
            )
            suffix = f" ({component_text})" if component_text else ""
            excerpt = " ".join(chunk.text.split())
            if len(excerpt) > 180:
                excerpt = excerpt[:177].rsplit(" ", 1)[0] + "…"
            lines.append(
                f"    - {title}{location}{role} · relevance {float(score):.3f}{suffix}"
            )
            if excerpt:
                lines.append(f"      “{excerpt}”")
    else:
        lines.append("  Evidence candidates: none")

    if trace is not None:
        seed_count = len(trace.metadata.get("seeds", ()))
        path_count = len(trace.paths)
        edge_count = len(trace.edges)
        if path_count:
            lines.append(f"  Retrieval context: graph expansion followed {path_count} paths")
        elif edge_count:
            lines.append(f"  Retrieval context: graph expansion used {edge_count} links")
        elif seed_count:
            lines.append(f"  Retrieval context: started from {seed_count} indexed passages")
        else:
            lines.append("  Retrieval context: no graph expansion paths returned")
    else:
        lines.append("  Retrieval context: provenance unavailable")
    return "\n".join(lines)


def search_claim(
    claim: Claim,
    *,
    backend=None,
    embed_queries: EmbedQueries | None = None,
    search: Search | None = None,
    **options,
) -> dict:
    """Dispatch to the backend or injected-search retrieval interface."""
    if backend is not None:
        if embed_queries is not None or search is not None:
            raise TypeError("Pass either backend or both embed_queries and search")
        return _search_claim_with_backend(claim, backend=backend, **options)
    if embed_queries is None or search is None:
        raise TypeError("Pass backend or both embed_queries and search")
    return _search_claim_with_explicit_search(
        claim,
        embed_queries=embed_queries,
        search=search,
        **options,
    )


def _search_claim_with_explicit_search(
    claim: Claim,
    *,
    embed_queries: EmbedQueries,
    search: Search,
    limit: int = 5,
    min_score: float = 0.0,
    expand: Expand | None = None,
    rerank: Rerank | None = None,
    progress: ProgressUpdate | None = None,
    context_for: ContextFor | None = None,
    select_evidence: SelectEvidence | None = None,
    explain: Explain | None = None,
    enrich: bool = True,
    top_only: bool = False,
    one_per_paper: bool = False,
) -> dict:
    text = claim["text"].strip()
    if not text:
        raise ValueError("A claim is required")
    variants = [text]
    if expand is not None:
        if progress is not None:
            progress("Preparing query expansion")
        seen_variants = {text.casefold()}
        for expanded in _run_expander(expand, text, progress):
            normalized = expanded.strip()
            key = normalized.casefold()
            if normalized and key not in seen_variants:
                variants.append(normalized)
                seen_variants.add(key)
    vectors = embed_queries(variants)
    if len(vectors) != len(variants):
        raise ValueError("Embedding provider returned a different number of query vectors")
    if progress is not None:
        progress("Embedding expanded queries")

    # Search each hypothesis independently, then retain the strongest score for
    # each chunk. This avoids duplicate passages overwhelming the final reranker.
    by_chunk: dict[str, ScoredChunk] = {}
    score_details: dict[str, dict[str, float]] = {}
    for index, (variant, vector) in enumerate(zip(variants, vectors, strict=True), start=1):
        if progress is not None:
            progress(f"Retrieving evidence · {index}/{len(variants)} queries")
        for hit in search(vector, limit=limit, query_text=variant):
            if hit[1] < min_score:
                continue
            key = hit[0].chunk_id
            previous = by_chunk.get(key)
            if previous is None or hit[1] > previous[1]:
                by_chunk[key] = hit
                score_details[key] = {
                    "vector": float(getattr(hit[1], "semantic", hit[1])),
                    "lexical": float(getattr(hit[1], "lexical", 0.0)),
                    "retrieval": float(hit[1]),
                }
    hits = sorted(by_chunk.values(), key=lambda hit: hit[1], reverse=True)
    if progress is not None:
        progress("Reranking candidates")
    reranker = rerank or heuristic_rerank
    ordered_hits = _run_reranker(reranker, text, hits, progress)
    selected_hits = _limit_hits(ordered_hits, limit=limit, one_per_paper=one_per_paper)
    result = {
        "claim": claim,
        "queries": tuple(variants),
        "hits": selected_hits,
        "score_details": score_details,
    }
    if context_for is not None:
        if enrich:
            enrichment_hits = selected_hits[:1] if top_only else selected_hits
            contexts = {
                chunk.chunk_id: context_for(chunk) for chunk, _score in enrichment_hits
            }

            def enrich_one(hit: ScoredChunk) -> dict:
                return enrich_hits(
                    [
                        {
                            "claim": claim,
                            "hits": [hit],
                            "score_details": score_details,
                        }
                    ],
                    context_for=lambda chunk: contexts[chunk.chunk_id],
                    select_evidence=select_evidence,
                    explain=explain,
                )[0]

            enriched: list[dict | None] = [None] * len(enrichment_hits)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(enrich_one, hit): index
                    for index, hit in enumerate(enrichment_hits)
                }
                completed = 0
                for future in as_completed(futures):
                    enriched[futures[future]] = future.result()
                    completed += 1
                    if progress is not None:
                        progress(
                            f"Enriching finalists · {completed}/{len(enrichment_hits)} complete"
                        )
            result["items"] = [item for item in enriched if item is not None]
        else:
            contexts = {
                chunk.chunk_id: context_for(chunk) for chunk, _score in selected_hits
            }
            result["items"] = enrich_hits(
                [result],
                context_for=lambda chunk: contexts[chunk.chunk_id],
                top_only=top_only,
            )
    return result


def _limit_hits(
    hits: Sequence[ScoredChunk], *, limit: int, one_per_paper: bool
) -> list[ScoredChunk]:
    """Rank passages and optionally keep only the strongest passage per paper."""
    ranked = sorted(hits, key=lambda hit: hit[1], reverse=True)
    if not one_per_paper:
        return ranked[:limit]
    by_article: dict[str, ScoredChunk] = {}
    for hit in ranked:
        by_article.setdefault(hit[0].paper.zotero_key, hit)
    return list(by_article.values())[:limit]


def _run_expander(
    expand: Expand, text: str, progress: ProgressUpdate | None
) -> list[str]:
    """Call newer progress-aware expanders without breaking injected legacy ones."""
    if progress is None:
        return expand(text)
    try:
        accepts_progress = "progress" in signature(expand).parameters
    except (TypeError, ValueError):
        accepts_progress = False
    if accepts_progress:
        return expand(text, progress=progress)  # type: ignore[call-arg]
    return expand(text)


def _run_reranker(
    rerank: Rerank,
    text: str,
    hits: Sequence[ScoredChunk],
    progress: ProgressUpdate | None,
) -> list[ScoredChunk]:
    if progress is None:
        return rerank(text, hits)
    try:
        parameters = signature(rerank).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs = {}
    accepts_kwargs = any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if "progress" in parameters or accepts_kwargs:
        kwargs["progress"] = progress
    if kwargs:
        return rerank(text, hits, **kwargs)  # type: ignore[call-arg]
    return rerank(text, hits)


def retrieve_claims(
    claims: Iterable[Claim],
    *,
    embed_queries: EmbedQueries,
    search: Search,
    limit: int = 5,
    min_score: float = 0.0,
    expand: Expand | None = None,
    rerank: Rerank | None = None,
    progress: ProgressUpdate | None = None,
) -> list[dict]:
    return [
        _search_claim_with_explicit_search(
            claim,
            embed_queries=embed_queries,
            search=search,
            limit=limit,
            min_score=min_score,
            expand=expand,
            rerank=rerank,
            progress=progress,
        )
        for claim in claims
    ]


def enrich_hits(
    results: Iterable[dict],
    *,
    context_for: ContextFor,
    select_evidence: SelectEvidence | None = None,
    explain: Explain | None = None,
    top_only: bool = False,
) -> list[dict]:
    """Return render-ready plain dictionaries without calling transport code directly."""
    enriched: list[dict] = []
    for result in results:
        claim = result["claim"]
        hits: list[ScoredChunk] = result["hits"]
        selected_hits = hits[:1] if top_only else hits
        for hit in selected_hits:
            chunk, score = hit
            context = context_for(chunk)
            evidence = _best_excerpt(claim["text"], context or chunk.text)
            selected_rationale = ""
            if select_evidence is not None:
                selection = select_evidence(claim["text"], hit, context or chunk.text)
                if selection is not None:
                    selected_evidence, selected_rationale = selection
                    validated = _validated_excerpt(
                        selected_evidence,
                        context or chunk.text,
                        fallback=evidence,
                    )
                    if validated != " ".join(selected_evidence.split()).strip():
                        selected_rationale = ""
                    evidence = validated or evidence
            rationale_function = explain or template_rationale
            rationale = selected_rationale or rationale_function(claim["text"], evidence)
            enriched.append(
                {
                    "claim": claim,
                    "chunk": chunk,
                    "score": score,
                    "vector_score": result.get("score_details", {})
                    .get(chunk.chunk_id, {})
                    .get("vector", score),
                    "lexical_score": result.get("score_details", {})
                    .get(chunk.chunk_id, {})
                    .get("lexical", 0.0),
                    "retrieval_score": result.get("score_details", {})
                    .get(chunk.chunk_id, {})
                    .get("retrieval", score),
                    "rerank_score": score,
                    "evidence": evidence,
                    "rationale": rationale,
                    "context": context,
                }
            )
    return enriched


def _validated_excerpt(excerpt: str, context: str, *, fallback: str) -> str:
    normalized_excerpt = " ".join(excerpt.split()).strip()
    normalized_context = " ".join(context.split())
    if not normalized_excerpt:
        return fallback
    if _looks_like_metadata(normalized_excerpt):
        return fallback if not _looks_like_metadata(fallback) else ""
    if normalized_excerpt.casefold().startswith("section:"):
        return fallback
    if normalized_excerpt not in normalized_context:
        return fallback
    if _looks_like_heading(normalized_excerpt):
        return fallback
    return normalized_excerpt


def _best_excerpt(claim: str, context: str) -> str:
    normalized_context = " ".join(context.split()).strip()
    if not normalized_context:
        return ""
    candidates = _excerpt_candidates(normalized_context)
    candidates = [candidate for candidate in candidates if not _looks_like_metadata(candidate)]
    if not candidates:
        return (
            ""
            if _looks_like_metadata(normalized_context)
            else _truncate_excerpt(normalized_context)
        )
    scored = [
        (candidate, _excerpt_score(claim, candidate))
        for candidate in candidates
        if not _looks_like_heading(candidate)
    ]
    if not scored:
        scored = [(normalized_context, _excerpt_score(claim, normalized_context))]
    best_candidate, _score = max(scored, key=lambda item: (item[1], len(item[0])))
    return _truncate_excerpt(best_candidate)


def _excerpt_candidates(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    candidates = list(sentences)
    candidates.extend(
        f"{sentences[index]} {sentences[index + 1]}"
        for index in range(len(sentences) - 1)
    )
    return candidates


def _excerpt_score(claim: str, excerpt: str) -> float:
    claim_terms = _content_terms(claim)
    excerpt_terms = _content_terms(excerpt)
    if not claim_terms or not excerpt_terms:
        return 0.0
    overlap = claim_terms & excerpt_terms
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(claim_terms)
    density = len(overlap) / len(excerpt_terms)
    return coverage * 2.0 + density


def _content_terms(text: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
    return {
        term
        for term in re.findall(r"[\w-]+", text.casefold())
        if term not in stop_words and len(term) > 2
    }


def _truncate_excerpt(text: str, *, max_words: int = 60) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]) + " ..."


def _looks_like_heading(text: str) -> bool:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return True
    if normalized.casefold().startswith("section:"):
        return True
    if re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", normalized):
        return True
    words = normalized.split()
    if len(words) <= 6 and normalized == normalized.title() and not re.search(r"[.!?]", normalized):
        return True
    return False


def _looks_like_metadata(text: str) -> bool:
    """Reject administrative PDF text as citation evidence."""

    normalized = " ".join(text.casefold().split())
    if not normalized or len(normalized.split()) > 60:
        return False
    markers = (
        "published by ",
        "copyright ",
        "all rights reserved",
        "street, ",
        " avenue, ",
        " boulevard, ",
        " oxford ox",
        " malden, ma ",
    )
    return any(marker in f" {normalized}" for marker in markers)
