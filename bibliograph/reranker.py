import json
import re
from collections.abc import Sequence
from typing import Protocol

from .config import DEFAULT_OPENAI_API_KEY, DEFAULT_OPENAI_BASE_URL
from .models import CitationSource, DraftMatch


class Reranker(Protocol):
    def rerank(
        self, draft_text: str, sources: Sequence[CitationSource]
    ) -> list[tuple[int, float]]: ...


class HeuristicReranker:
    """Offline fallback that preserves retrieval order and scores."""

    def rerank(self, draft_text: str, sources: Sequence[CitationSource]) -> list[tuple[int, float]]:
        return [(index, source.score) for index, source in enumerate(sources)]


class OpenAIReranker:
    def __init__(
        self,
        model: str,
        api_key: str = DEFAULT_OPENAI_API_KEY,
        base_url: str | None = None,
        client=None,
    ):
        api_key = api_key or DEFAULT_OPENAI_API_KEY
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "LLM reranking requires the optional 'llm' dependencies"
                ) from exc
            client = OpenAI(api_key=api_key, base_url=base_url or DEFAULT_OPENAI_BASE_URL)
        self.client = client
        self.model = model

    def rerank(self, draft_text: str, sources: Sequence[CitationSource]) -> list[tuple[int, float]]:
        candidates = "\n\n".join(
            f"[{index}] {source.chunk.text}\nSOURCE: {source.chunk.paper.citation_label}, "
            f"page {source.chunk.page or 'unknown'}"
            for index, source in enumerate(sources)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rerank evidence for a draft claim. Return JSON only with an "
                        "'items' array. "
                        "Each item must contain candidate (integer index) and support (0 to 1). "
                        "Only judge whether the supplied passage supports the claim."
                    ),
                },
                {"role": "user", "content": f"CLAIM:\n{draft_text}\n\nCANDIDATES:\n{candidates}"},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = _parse_json_payload(content)
        items = payload.get("items", [])
        ranking = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            index = item.get("candidate")
            try:
                score = float(item.get("support", 0.0))
            except (TypeError, ValueError):
                continue
            if isinstance(index, int) and 0 <= index < len(sources) and index not in seen:
                ranking.append((index, max(0.0, min(1.0, score))))
                seen.add(index)
        ranking.extend(
            (index, source.score)
            for index, source in enumerate(sources)
            if index not in seen
        )
        return ranking


def _parse_json_payload(content: str | list[dict]) -> dict:
    """Parse JSON from plain, fenced, or reasoning-wrapped model output."""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise ValueError("LLM response content is not text")
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if isinstance(payload, list):
        return {"items": payload}
    if not isinstance(payload, dict):
        raise ValueError("LLM reranker response must be a JSON object or array")
    return payload


def rerank_matches(matches: Sequence[DraftMatch], reranker: Reranker) -> list[DraftMatch]:
    reranked: list[DraftMatch] = []
    for match in matches:
        ranking = reranker.rerank(match.draft_text, match.sources)
        sources = tuple(
            CitationSource(match.sources[index].chunk, score)
            for index, score in ranking
        )
        reranked.append(
            DraftMatch(
                match.draft_text,
                sources,
                match.line_start,
                match.citation_keys,
                match.source_format,
            )
        )
    return reranked
