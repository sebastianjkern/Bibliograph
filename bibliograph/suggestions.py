import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .config import DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_BASE_URL
from .models import CitationSource, DraftMatch, TextChunk


@dataclass(frozen=True)
class EvidenceSelection:
    supports_claim: bool
    quote: str
    rationale: str


class EvidenceExtractor(Protocol):
    def extract(
        self, draft_text: str, source: CitationSource, context: str
    ) -> EvidenceSelection: ...


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
            "api_key": api_key or DEFAULT_OPENAI_API_KEY,
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


class OpenAIEvidenceExtractor:
    """Select an exact, source-grounded quote from retrieved context."""

    def __init__(
        self,
        model: str,
        api_key: str = DEFAULT_OPENAI_API_KEY,
        base_url: str | None = None,
        client=None,
    ):
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key or DEFAULT_OPENAI_API_KEY,
                base_url=base_url or DEFAULT_OPENAI_BASE_URL,
            )
        self.client = client
        self.model = model

    def extract(
        self, draft_text: str, source: CitationSource, context: str
    ) -> EvidenceSelection:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select exact evidence for a draft claim. Return JSON only with "
                        "supports_claim (boolean), quote (exact text copied from CONTEXT), "
                        "and rationale (short explanation). Never invent or paraphrase a quote."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CLAIM:\n{draft_text}\n\nSOURCE:\n"
                        f"{source.chunk.paper.citation_label}, "
                        f"page {source.chunk.page or 'unknown'}\n\n"
                        f"CONTEXT:\n{context}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = _parse_evidence_payload(content)
        return EvidenceSelection(
            bool(payload.get("supports_claim", False)),
            str(payload.get("quote", "")).strip(),
            str(payload.get("rationale", "")).strip(),
        )


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
    evidence_extractor: EvidenceExtractor | None = None,
    context_provider: Callable[[TextChunk], str] | None = None,
) -> list[CitationSuggestion]:
    generator = generator or GroundedTemplateGenerator()
    suggestions: list[CitationSuggestion] = []
    for match in matches:
        eligible = [source for source in match.sources if source.score >= min_score]
        if not eligible:
            continue
        source = eligible[0]
        evidence = source.chunk.text
        rationale = generator.explain(match.draft_text, evidence)
        if evidence_extractor is not None and context_provider is not None:
            try:
                context = context_provider(source.chunk)
                selection = evidence_extractor.extract(match.draft_text, source, context)
                if selection.supports_claim and selection.quote and selection.quote in context:
                    evidence = selection.quote
                    rationale = selection.rationale or rationale
            except Exception:
                pass
        suggestions.append(
            CitationSuggestion(
                draft_text=match.draft_text,
                citation=format_citation(source),
                evidence=evidence,
                source=source,
                rationale=rationale,
            )
        )
    return suggestions


def _parse_evidence_payload(content: str | list[dict]) -> dict:
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise ValueError("LLM evidence response content is not text")
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM evidence response must be a JSON object")
    return payload
