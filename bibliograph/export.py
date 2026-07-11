from collections.abc import Iterable

from .models import DraftMatch
from .suggestions import CitationSuggestion


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
