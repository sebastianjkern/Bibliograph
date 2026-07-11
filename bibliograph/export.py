from collections.abc import Iterable

from .models import DraftMatch
from .suggestions import CitationSuggestion


def to_markdown(suggestions: Iterable[CitationSuggestion]) -> str:
    lines = ["# Citation suggestions", ""]
    for suggestion in suggestions:
        lines.extend(
            [
                f"## {suggestion.citation}",
                f"Support score: {suggestion.source.score:.2f}",
                f"> Draft: {suggestion.draft_text}",
                f"> Evidence: {suggestion.evidence}",
                "",
                f"Rationale: {suggestion.rationale}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def sources_to_markdown(matches: Iterable[DraftMatch]) -> str:
    """Format every retrieved source for a single-claim lookup."""
    matches = list(matches)
    claim = matches[0].draft_text if matches else ""
    lines = ["# Local sources", "", f"> Claim: {claim}", ""]
    for match in matches:
        for index, source in enumerate(match.sources, start=1):
            paper = source.chunk.paper
            lines.extend(
                [
                    f"## {index}. {paper.citation_label}",
                    f"Support score: {source.score:.2f}",
                    f"DOI: {paper.doi or 'unknown'}",
                    f"Page: {source.chunk.page or 'unknown'}",
                    f"> Evidence: {source.chunk.text}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"
