"""Injected text retrieval and LLM enrichment pipeline."""

from collections.abc import Callable, Iterable, Sequence

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
            evidence = chunk.text
            selected_rationale = ""
            if select_evidence is not None:
                selection = select_evidence(claim["text"], hit, context_for(chunk))
                if selection is not None:
                    evidence, selected_rationale = selection
            rationale_function = explain or template_rationale
            rationale = selected_rationale or rationale_function(claim["text"], evidence)
            enriched.append(
                {
                    "claim": claim,
                    "chunk": chunk,
                    "score": score,
                    "evidence": evidence,
                    "rationale": rationale,
                }
            )
    return enriched
