import re
from collections.abc import Iterable

from .drafts import DraftClaim
from .embeddings import Embedder
from .logging_utils import get_logger
from .models import DraftMatch
from .store import SQLiteIndex

logger = get_logger("retrieval")


def _log_retrieval(claim: str, sources: tuple, min_score: float) -> None:
    logger.debug(
        "Vector search claim=%r min_score=%.3f candidates=%d",
        claim,
        min_score,
        len(sources),
    )
    for rank, source in enumerate(sources, start=1):
        logger.debug(
            "Vector candidate rank=%d score=%.6f chunk_id=%s source=%r page=%s text=%r",
            rank,
            source.score,
            source.chunk.chunk_id,
            source.chunk.paper.citation_label,
            source.chunk.page or "unknown",
            source.chunk.text,
        )


def split_draft(text: str, min_words: int = 4) -> list[str]:
    """Return meaningful paragraphs/sentences from a draft."""
    candidates = re.split(r"\n\s*\n|(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in candidates if len(part.split()) >= min_words]


def find_citations(
    draft: str,
    index: SQLiteIndex,
    embedder: Embedder,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[DraftMatch]:
    """Match each meaningful draft passage to indexed evidence."""
    passages = split_draft(draft)
    matches: list[DraftMatch] = []
    for passage, embedding in zip(passages, embedder.embed(passages), strict=True):
        sources = tuple(
            source for source in index.search(embedding, limit) if source.score >= min_score
        )
        _log_retrieval(passage, sources, min_score)
        matches.append(DraftMatch(passage, sources))
    return matches


def find_claim_sources(
    claim: str,
    index: SQLiteIndex,
    embedder: Embedder,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[DraftMatch]:
    """Find evidence for one claim without parsing or synchronizing a draft."""
    claim = claim.strip()
    if not claim:
        raise ValueError("A claim is required")
    embedding = embedder.embed([claim])[0]
    sources = tuple(
        source
        for source in index.search(embedding, limit, query_text=claim)
        if source.score >= min_score
    )
    _log_retrieval(claim, sources, min_score)
    return [DraftMatch(claim, sources)]


def find_claim_citations(
    claims: Iterable[DraftClaim],
    index: SQLiteIndex,
    embedder: Embedder,
    limit: int = 5,
    min_score: float = 0.0,
) -> list[DraftMatch]:
    """Retrieve hybrid semantic/lexical evidence for parsed draft claims."""
    claims = list(claims)
    embeddings = embedder.embed([claim.text for claim in claims])
    matches = [
        DraftMatch(
            claim.text,
            tuple(
                source
                for source in index.search(
                    embedding,
                    limit,
                    query_text=claim.text,
                )
                if source.score >= min_score
            ),
            claim.line_start,
            claim.citation_keys,
            claim.source_format,
        )
        for claim, embedding in zip(claims, embeddings, strict=True)
    ]
    for match in matches:
        _log_retrieval(match.draft_text, match.sources, min_score)
    return matches


def flatten_sources(matches: Iterable[DraftMatch]) -> list[tuple[str, str, float]]:
    """Create a compact export-friendly view of draft-to-source matches."""
    return [
        (match.draft_text, source.chunk.paper.citation_label, source.score)
        for match in matches
        for source in match.sources
    ]
