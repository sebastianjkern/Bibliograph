from collections.abc import Iterable

from .suggestions import CitationSuggestion


def to_markdown(suggestions: Iterable[CitationSuggestion]) -> str:
    lines = ["# Citation suggestions", ""]
    for suggestion in suggestions:
        lines.extend(
            [
                f"## {suggestion.citation}",
                f"> Draft: {suggestion.draft_text}",
                f"> Evidence: {suggestion.evidence}",
                "",
                f"Rationale: {suggestion.rationale}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
