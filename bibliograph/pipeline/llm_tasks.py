"""Provider-independent prompt construction and response parsing."""

import asyncio
import json
import re
from collections.abc import Callable, Sequence

from ..domain import ScoredChunk, citation_label

Complete = Callable[..., str]
ProgressUpdate = Callable[[str], None]

_EXPANSION_TYPES = frozenset(
    {
        "paraphrase",
        "stronger",
        "weaker",
        "supporting",
        "contradicting",
        "causal",
        "consequence",
        "assumption",
        "alternative",
        "domain",
        "expert",
        "statistical",
    }
)

_EXPANSION_PROMPTS = (
    ("semantic", ("paraphrase", "stronger", "weaker")),
    ("evidence", ("supporting", "contradicting", "causal", "consequence")),
    ("reasoning", ("assumption", "alternative")),
    ("terminology", ("domain", "expert", "statistical")),
)


def heuristic_expand(claim: str, *, progress: ProgressUpdate | None = None) -> list[str]:
    """Return no extra queries when LLM expansion is disabled or unavailable."""
    return [claim]


def expand_query(
    claim: str,
    *,
    complete: Complete,
    progress: ProgressUpdate | None = None,
) -> list[str]:
    """Generate bounded, typed alternate hypotheses using focused prompts."""
    payloads = asyncio.run(_expand_prompts(claim, complete=complete, progress=progress))
    expanded: list[str] = []
    seen: set[str] = {" ".join(claim.split()).casefold()}
    for content, types in payloads:
        try:
            payload = _json_object(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for item in payload.get("items", []):
            if (
                not isinstance(item, dict)
                or str(item.get("type", "")).casefold() not in types
            ):
                continue
            text = " ".join(str(item.get("text", "")).split()).strip()
            key = text.casefold()
            if text and key not in seen and len(text) <= 500:
                expanded.append(text)
                seen.add(key)
            if len(expanded) >= 12:
                return expanded
    return expanded


def _expansion_messages(claim: str, perspective: str, types: Sequence[str]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                f"Create alternate search queries from the {perspective} perspective. "
                "Return JSON only with an 'items' array. Each item must contain type "
                "and text. Generate at most one useful item for each requested type: "
                + ", ".join(types)
                + ". Preserve the claim's meaning; do not invent entities, numbers, "
                "or citations. Keep each query concise and searchable."
            ),
        },
        {"role": "user", "content": f"CLAIM:\n{claim}"},
    ]


async def _expand_prompts(
    claim: str,
    *,
    complete: Complete,
    progress: ProgressUpdate | None = None,
) -> list[tuple[str, Sequence[str]]]:
    async def run_prompt(index: int, perspective: str, types: Sequence[str]):
        content = await asyncio.to_thread(
            complete,
            _expansion_messages(claim, perspective, types),
            json_mode=True,
        )
        return index, content, types

    results: list[tuple[str, Sequence[str]] | None] = [None] * len(_EXPANSION_PROMPTS)
    tasks = [
        asyncio.create_task(run_prompt(index, perspective, types))
        for index, (perspective, types) in enumerate(_EXPANSION_PROMPTS)
    ]
    completed_count = 0
    for completed in asyncio.as_completed(tasks):
        index, content, types = await completed
        results[index] = (content, types)
        completed_count += 1
        if progress is not None:
            progress(
                f"Expanding queries · {completed_count}/{len(_EXPANSION_PROMPTS)} complete"
            )
    return [result for result in results if result is not None]


def heuristic_rerank(
    _claim: str,
    hits: Sequence[ScoredChunk],
    *,
    progress: ProgressUpdate | None = None,
) -> list[ScoredChunk]:
    """Keep the retrieval order when an LLM is disabled or unavailable."""
    return list(hits)


def rerank(
    claim: str,
    hits: Sequence[ScoredChunk],
    *,
    complete: Complete,
    progress: ProgressUpdate | None = None,
) -> list[ScoredChunk]:
    if not hits:
        return []
    batches = [hits[start : start + 10] for start in range(0, len(hits), 10)]
    return asyncio.run(
        _rerank_batches(
            claim,
            batches,
            complete=complete,
            progress=progress,
        )
    )


async def _rerank_batches(
    claim: str,
    batches: Sequence[Sequence[ScoredChunk]],
    *,
    complete: Complete,
    progress: ProgressUpdate | None = None,
) -> list[ScoredChunk]:
    async def run_batch(index: int, batch: Sequence[ScoredChunk]):
        result = await asyncio.to_thread(_rerank_batch, claim, batch, complete=complete)
        return index, result

    results: list[list[ScoredChunk] | None] = [None] * len(batches)
    tasks = [asyncio.create_task(run_batch(index, batch)) for index, batch in enumerate(batches)]
    completed_count = 0
    for completed in asyncio.as_completed(tasks):
        index, result = await completed
        results[index] = result
        completed_count += 1
        if progress is not None:
            progress(
                f"Reranking candidates · {completed_count}/{len(batches)} batches complete"
            )
    return [hit for result in results if result is not None for hit in result]


def _rerank_batch(
    claim: str, hits: Sequence[ScoredChunk], *, complete: Complete
) -> list[ScoredChunk]:
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
    if payload.get("supports_claim") and _is_valid_quote(quote, context):
        return " ".join(quote.split()), rationale
    return None


def explain(claim: str, evidence: str, *, complete: Complete) -> str:
    content = complete(
        [
            {
                "role": "system",
                "content": (
                    "Assess whether retrieved evidence supports a draft passage. Return only the "
                    "explanation text. Do not prefix the answer with labels like SUPPORT, "
                    "RATIONALE, or ANSWER. Do not use bullets, numbering, or markdown. Keep it "
                    "to one concise paragraph."
                ),
            },
            {"role": "user", "content": f"DRAFT:\n{claim}\n\nEVIDENCE:\n{evidence}"},
        ],
        json_mode=False,
    ).strip()
    return _strip_label_prefix(content)


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


def _is_valid_quote(quote: str, context: str) -> bool:
    normalized_quote = " ".join(quote.split()).strip()
    normalized_context = " ".join(context.split())
    if not normalized_quote or normalized_quote not in normalized_context:
        return False
    if normalized_quote.casefold().startswith("section:"):
        return False
    if re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", normalized_quote):
        return False
    words = normalized_quote.split()
    if len(words) <= 3 and normalized_quote == normalized_quote.title():
        return False
    return True


def _strip_label_prefix(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^(?:support|rationale|explanation|answer)\.?:\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:support|rationale|explanation|answer)\.\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned
