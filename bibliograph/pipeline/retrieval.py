"""Injected text retrieval and LLM enrichment pipeline."""

import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import signature

from ..domain import Chunk, Claim, ScoredChunk
from .llm_tasks import heuristic_rerank, template_rationale

EmbedQueries = Callable[[Sequence[str]], list[list[float]]]
Search = Callable[..., list[ScoredChunk]]
ContextFor = Callable[[Chunk], str]
Rerank = Callable[[str, Sequence[ScoredChunk]], list[ScoredChunk]]
Expand = Callable[[str], list[str]]
ProgressUpdate = Callable[[str], None]
SelectEvidence = Callable[[str, ScoredChunk, str], tuple[str, str] | None]
Explain = Callable[[str, str], str]


def search_claim(
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
    selected_hits = _top_article_hits(ordered_hits, limit=10)
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


def _top_article_hits(hits: Sequence[ScoredChunk], *, limit: int) -> list[ScoredChunk]:
    """Keep the highest-scoring passage from each article, globally ranked."""
    by_article: dict[str, ScoredChunk] = {}
    for hit in hits:
        article_key = hit[0].paper.zotero_key
        previous = by_article.get(article_key)
        if previous is None or hit[1] > previous[1]:
            by_article[article_key] = hit
    return sorted(by_article.values(), key=lambda hit: hit[1], reverse=True)[:limit]


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
        search_claim(
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
    if not candidates:
        return _truncate_excerpt(normalized_context)
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
