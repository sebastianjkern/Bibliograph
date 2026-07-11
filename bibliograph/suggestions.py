from dataclasses import dataclass
from typing import Protocol

from .config import DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_BASE_URL
from .models import CitationSource, DraftMatch


@dataclass(frozen=True)
class CitationSuggestion:
    draft_text: str
    citation: str
    evidence: str
    source: CitationSource
    rationale: str


class SuggestionGenerator(Protocol):
    def explain(self, draft_text: str, evidence: str) -> str: ...


class GroundedTemplateGenerator:
    """Offline generator that never invents claims or bibliographic fields."""

    def explain(self, draft_text: str, evidence: str) -> str:
        return (
            "Retrieved evidence overlaps with the draft passage; "
            "verify the source before citing."
        )


class OpenAISuggestionGenerator:
    def __init__(
        self,
        model: str,
        api_key: str = DEFAULT_OPENAI_API_KEY,
        base_url: str | None = None,
    ):
        from openai import OpenAI

        kwargs = {
            "api_key": api_key,
            "base_url": base_url or DEFAULT_OPENAI_BASE_URL,
        }
        self.client = OpenAI(**kwargs)
        self.model = model

    def explain(self, draft_text: str, evidence: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You assess whether retrieved evidence supports a draft passage. "
                        "Do not add facts not present in the evidence. Give one concise rationale."
                    ),
                },
                {"role": "user", "content": f"DRAFT:\n{draft_text}\nEVIDENCE:\n{evidence}"},
            ],
        )
        return response.choices[0].message.content or ""


def format_citation(source: CitationSource) -> str:
    paper = source.chunk.paper
    authors = ", ".join(paper.authors) if paper.authors else paper.title
    year = f" ({paper.year})" if paper.year else ""
    doi = f" DOI: {paper.doi}" if paper.doi else ""
    page = f", p. {source.chunk.page}" if source.chunk.page else ""
    return f"{authors}{year}, {paper.title}{page}.{doi}"


def suggest_citations(
    matches: list[DraftMatch],
    generator: SuggestionGenerator | None = None,
    min_score: float = 0.0,
) -> list[CitationSuggestion]:
    generator = generator or GroundedTemplateGenerator()
    suggestions: list[CitationSuggestion] = []
    for match in matches:
        eligible = [source for source in match.sources if source.score >= min_score]
        if not eligible:
            continue
        source = eligible[0]
        suggestions.append(
            CitationSuggestion(
                draft_text=match.draft_text,
                citation=format_citation(source),
                evidence=source.chunk.text,
                source=source,
                rationale=generator.explain(match.draft_text, source.chunk.text),
            )
        )
    return suggestions
