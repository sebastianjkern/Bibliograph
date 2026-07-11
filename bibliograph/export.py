from collections.abc import Callable, Iterable

from .logging_utils import get_logger
from .models import DraftMatch, TextChunk
from .suggestions import CitationSuggestion, EvidenceExtractor

logger = get_logger("export")


def to_markdown(suggestions: Iterable[CitationSuggestion]) -> str:
    suggestions = list(suggestions)
    lines = ["# Citation suggestions", "", f"Found {len(suggestions)} suggestion(s).", ""]
    for number, suggestion in enumerate(suggestions, start=1):
        paper = suggestion.source.chunk.paper
        lines.extend(
            [
                f"## Suggestion {number}",
                "",
                f"**Citation:** {suggestion.citation}",
                f"**Support score:** `{suggestion.source.score:.2f}`",
                "",
                "### Draft claim",
                *_blockquote(suggestion.draft_text),
                "",
                "### Evidence",
                *_blockquote(suggestion.evidence),
                "",
                "### Rationale",
                suggestion.rationale,
                "",
                "### Source",
                f"- **Paper:** {paper.title}",
                f"- **Authors:** {', '.join(paper.authors) or 'Unknown'}",
                f"- **Page:** {suggestion.source.chunk.page or 'Unknown'}",
                f"- **DOI:** {paper.doi or 'Unknown'}",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _blockquote(text: str) -> list[str]:
    return [f"> {line}" for line in text.splitlines()] or ["> "]


def sources_to_markdown(
    matches: Iterable[DraftMatch],
    evidence_extractor: EvidenceExtractor | None = None,
    context_provider: Callable[[TextChunk], str] | None = None,
) -> str:
    """Format every retrieved source for a single-claim lookup."""
    matches = list(matches)
    claim = matches[0].draft_text if matches else ""
    lines = ["# Local sources", "", f"> Claim: {claim}", ""]
    for match in matches:
        for index, source in enumerate(match.sources, start=1):
            paper = source.chunk.paper
            evidence = source.chunk.text
            if evidence_extractor is not None and context_provider is not None:
                try:
                    context = context_provider(source.chunk)
                    selection = evidence_extractor.extract(match.draft_text, source, context)
                    if selection.supports_claim and selection.quote in context:
                        evidence = selection.quote
                except Exception as error:
                    logger.warning(
                        "Evidence extraction failed for %s; using the retrieved chunk: %s",
                        paper.title,
                        error,
                    )
            lines.extend(
                [
                    f"## {index}. {paper.citation_label}",
                    f"Support score: {source.score:.2f}",
                    f"DOI: {paper.doi or 'unknown'}",
                    f"Page: {source.chunk.page or 'unknown'}",
                    f"> Evidence: {evidence}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"
