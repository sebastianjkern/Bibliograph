"""Provider-independent prompt construction and response parsing."""

import json
import re
from collections.abc import Callable, Sequence

from ..domain import ScoredChunk, citation_label

Complete = Callable[..., str]


def heuristic_rerank(_claim: str, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Keep the retrieval order when an LLM is disabled or unavailable."""
    return list(hits)


def rerank(claim: str, hits: Sequence[ScoredChunk], *, complete: Complete) -> list[ScoredChunk]:
    if not hits:
        return []
    candidates = "\n\n".join(
        f"[{index}] {chunk.text}\nSOURCE: {citation_label(chunk.paper)}, "
        f"page {chunk.page or 'unknown'}"
        for index, (chunk, _score) in enumerate(hits)
    )
    content = complete(
        [
            {
                "role": "system",
                "content": (
                    "Rerank evidence for a draft claim. Return JSON only with an 'items' array. "
                    "Each item must contain candidate (integer index) and support (0 to 1)."
                ),
            },
            {"role": "user", "content": f"CLAIM:\n{claim}\n\nCANDIDATES:\n{candidates}"},
        ],
        json_mode=True,
    )
    payload = _json_object(content)
    ordered: list[ScoredChunk] = []
    seen: set[int] = set()
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        index = item.get("candidate")
        try:
            score = float(item.get("support", 0.0))
        except (TypeError, ValueError):
            continue
        if isinstance(index, int) and 0 <= index < len(hits) and index not in seen:
            ordered.append((hits[index][0], max(0.0, min(1.0, score))))
            seen.add(index)
    ordered.extend(hit for index, hit in enumerate(hits) if index not in seen)
    return ordered


def select_evidence(
    claim: str,
    hit: ScoredChunk,
    context: str,
    *,
    complete: Complete,
) -> tuple[str, str] | None:
    chunk, _score = hit
    content = complete(
        [
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
                    f"CLAIM:\n{claim}\n\nSOURCE:\n{citation_label(chunk.paper)}, "
                    f"page {chunk.page or 'unknown'}\n\nCONTEXT:\n{context}"
                ),
            },
        ],
        json_mode=True,
    )
    payload = _json_object(content)
    quote = str(payload.get("quote", "")).strip()
    rationale = str(payload.get("rationale", "")).strip()
    if payload.get("supports_claim") and quote and quote in context:
        return quote, rationale
    return None


def explain(claim: str, evidence: str, *, complete: Complete) -> str:
    return complete(
        [
            {
                "role": "system",
                "content": (
                    "Assess whether retrieved evidence supports a draft passage. Do not add facts "
                    "not present in the evidence. Give one concise rationale."
                ),
            },
            {"role": "user", "content": f"DRAFT:\n{claim}\n\nEVIDENCE:\n{evidence}"},
        ],
        json_mode=False,
    ).strip()


def template_rationale(_claim: str, _evidence: str) -> str:
    return "Retrieved evidence overlaps with the draft passage; verify the source before citing."


def _json_object(content: str) -> dict:
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
        raise ValueError("LLM response must be a JSON object or array")
    return payload
