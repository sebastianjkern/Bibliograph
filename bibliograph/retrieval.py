import re
from collections.abc import Iterable

from .drafts import DraftClaim
from .embeddings import Embedder
from .models import DraftMatch
from .store import SQLiteIndex


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
        matches.append(DraftMatch(passage, sources))
    return matches


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
    return [
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


def flatten_sources(matches: Iterable[DraftMatch]) -> list[tuple[str, str, float]]:
    """Create a compact export-friendly view of draft-to-source matches."""
    return [
        (match.draft_text, source.chunk.paper.citation_label, source.score)
        for match in matches
        for source in match.sources
    ]
