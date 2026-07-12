"""Injected text retrieval and LLM enrichment pipeline."""

from collections.abc import Callable, Iterable, Sequence
import re

from ..domain import Chunk, Claim, ScoredChunk
from .llm_tasks import heuristic_rerank, template_rationale

EmbedQueries = Callable[[Sequence[str]], list[list[float]]]
Search = Callable[..., list[ScoredChunk]]
ContextFor = Callable[[Chunk], str]
Rerank = Callable[[str, Sequence[ScoredChunk]], list[ScoredChunk]]
SelectEvidence = Callable[[str, ScoredChunk, str], tuple[str, str] | None]
Explain = Callable[[str, str], str]


def search_claim(
    claim: Claim,
    *,
    embed_queries: EmbedQueries,
    search: Search,
    limit: int = 5,
    min_score: float = 0.0,
    rerank: Rerank | None = None,
) -> dict:
    text = claim["text"].strip()
    if not text:
        raise ValueError("A claim is required")
    vector = embed_queries([text])[0]
    hits = [hit for hit in search(vector, limit=limit, query_text=text) if hit[1] >= min_score]
    return {"claim": claim, "hits": (rerank or heuristic_rerank)(text, hits)}


def retrieve_claims(
    claims: Iterable[Claim],
    *,
    embed_queries: EmbedQueries,
    search: Search,
    limit: int = 5,
    min_score: float = 0.0,
    rerank: Rerank | None = None,
) -> list[dict]:
    return [
        search_claim(
            claim,
            embed_queries=embed_queries,
            search=search,
            limit=limit,
            min_score=min_score,
            rerank=rerank,
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
