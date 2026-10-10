"""Provider-independent prompt construction and response parsing."""

import asyncio
import json
import re
from collections.abc import Callable, Mapping, Sequence
from inspect import signature

from ..domain import ScoredChunk, citation_label
from ..text_matching import contains_text

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

_RERANK_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate",
                    "relation",
                    "evidence_role",
                    "directly_entails_claim",
                    "contains_claim_specific_result",
                    "contains_explicit_finding",
                    "requires_unsupported_inference",
                    "source_type",
                ],
                "properties": {
                    "candidate": {"type": "integer", "minimum": 0},
                    "relation": {"type": "string", "enum": ["supports", "contradicts", "neutral"]},
                    "evidence_role": {
                        "type": "string",
                        "enum": [
                            "result",
                            "conclusion",
                            "method",
                            "objective",
                            "definition",
                            "background",
                            "secondary",
                            "other",
                        ],
                    },
                    "directly_entails_claim": {"type": "boolean"},
                    "contains_claim_specific_result": {"type": "boolean"},
                    "contains_explicit_finding": {"type": "boolean"},
                    "requires_unsupported_inference": {"type": "boolean"},
                    "source_type": {"type": "string", "enum": ["primary", "secondary", "unknown"]},
                },
            },
        }
    },
}

_EVIDENCE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "relation",
        "quote",
        "reason",
        "scope",
        "quote_role",
        "quote_directness",
        "components",
    ],
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["supports", "partial", "contradicts", "mixed", "insufficient"],
        },
        "quote": {"type": "string"},
        "reason": {"type": "string"},
        "quote_role": {
            "type": "string",
            "enum": [
                "finding",
                "method",
                "data_description",
                "interpretation",
                "limitation",
                "background",
                "other",
            ],
        },
        "quote_directness": {
            "type": "string",
            "enum": ["direct", "contextual", "unrelated"],
        },
        "scope": {
            "type": "object",
            "additionalProperties": False,
            "required": ["population", "unit", "outcome", "geography", "time"],
            "properties": {
                "population": {"type": "string"},
                "unit": {"type": "string"},
                "outcome": {"type": "string"},
                "geography": {"type": "string"},
                "time": {"type": "string"},
            },
        },
        "components": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "status", "reason", "evidence_quote"],
                "properties": {
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["established", "partial", "unresolved", "contradicted"],
                    },
                    "reason": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
            },
        },
    },
}


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


def refine_queries(
    claim: str,
    evidence_ledger: Sequence[str],
    *,
    complete: Complete,
) -> list[str]:
    """Target unresolved evidence gaps using only already retrieved source material."""
    context = "\n\n".join(
        f"[{index}] {' '.join(entry.split())[:1400]}"
        for index, entry in enumerate(evidence_ledger[:8], start=1)
        if entry.strip()
    )
    if not context:
        return []
    messages = [
        {
            "role": "system",
            "content": (
                "Propose at most four targeted follow-up queries for the original claim. "
                "Treat the supplied evidence ledger as the baseline: identify explicit "
                "uncertainties in the assessed passages (such as a missing population, unit, "
                "outcome, geography, time period, or a mixed finding) and make each query "
                "address one such gap using terminology present in the source material. Do "
                "not generate queries from general associations or model background knowledge, "
                "do not invent entities or relations, and do not broaden the claim. If the ledger "
                "does not reveal a specific gap that another search could address, return an "
                "empty list. Return JSON only as {\"queries\": [\"...\"]}."
            ),
        },
        {
            "role": "user",
            "content": f"ORIGINAL CLAIM:\n{claim}\n\nASSESSED EVIDENCE LEDGER:\n{context}",
        },
    ]
    payload = _json_object(_complete_json(complete, messages, {
        "type": "object",
        "additionalProperties": False,
        "required": ["queries"],
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    }))
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        return []
    return [
        " ".join(query.split()).strip()
        for query in queries
        if isinstance(query, str) and query.strip() and len(query) <= 500
    ][:4]


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
        f"page {chunk.page or 'unknown'}, section {chunk.section or 'unknown'}, "
        f"content kind {chunk.content_kind}, indexed evidence role "
        f"{chunk.evidence_role or 'unclassified'}"
        for index, (chunk, _score) in enumerate(hits)
    )
    messages = [
        {
            "role": "system",
            "content": (
                    "Rerank evidence for a draft claim. Return one structured object with an "
                    "'items' array. "
                    "Evaluate each candidate against the claim itself, not merely topical overlap. "
                    "Each item must contain candidate (integer index), relation, evidence_role, "
                    "directly_entails_claim, contains_claim_specific_result, "
                    "contains_explicit_finding, requires_unsupported_inference, and source_type. "
                    "The four evidence fields must be JSON booleans, not numeric scores. "
                    "The source_type must be exactly primary, secondary, or unknown. "
                    "Do not calculate or return a final ranking score; the application "
                    "calculates it. "
                    "The relation must be exactly one of supports, contradicts, or neutral. "
                    "The evidence_role must be exactly one of result, conclusion, method, "
                    "objective, definition, background, secondary, or other. "
                    "Support means direct factual entailment: the passage must state or clearly "
                    "report the relationship in the claim. Preserve the claim's population, "
                    "unit of analysis, geographic scale, outcome, and time period. Do not assume "
                    "a finding generalizes across populations or measurement units unless the "
                    "passage establishes that link. A passage that only shares terminology, "
                    "states the "
                    "paper's objective, describes a method, or discusses previous studies "
                    "is neutral, "
                    "not supporting. Do not infer importance, causality, or results that are "
                    "absent. "
                    "Set contains_explicit_finding to false when a passage has no actual finding, "
                    "estimate, comparison, treatment/outcome relation, or explicit conclusion. "
                    "Penalize "
                    "introductions, literature reviews, and second-hand claims, but do not reject "
                    "them from retrieval. For the example claim 'Road network improvements "
                    "increased "
                    "overland trade in "
                    "Sub-Saharan Africa' and passage 'Our objective is to develop a gravity model "
                    "for "
                    "inter-city trade', return relation neutral, directly_entails_claim false, "
                    "and low evidence quality. "
                    "Return every candidate exactly once."
            ),
        },
        {"role": "user", "content": f"CLAIM:\n{claim}\n\nCANDIDATES:\n{candidates}"},
    ]
    content = _complete_json(complete, messages, _RERANK_SCHEMA)
    payload = _json_object(content)
    ordered: list[ScoredChunk] = []
    seen: set[int] = set()
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        index = item.get("candidate")
        if not isinstance(index, int) or not 0 <= index < len(hits) or index in seen:
            continue
        try:
            score = _rerank_support(item, hits[index][0].text)
        except (TypeError, ValueError):
            continue
        ordered.append((hits[index][0], max(0.0, min(1.0, score))))
        seen.add(index)
    ordered.extend(hit for index, hit in enumerate(hits) if index not in seen)
    return sorted(ordered, key=lambda hit: hit[1], reverse=True)


def _rerank_support(item: dict, text: str = "") -> float:
    """Calculate a deterministic score from the model's evidence subscores.

    New responses use boolean evidence judgments. Older providers may return
    the previous numeric schema; that format remains supported during rollout.
    """

    if "directly_entails_claim" in item:
        return _boolean_rerank_support(item, text)
    return _legacy_rerank_support(item, text)


def _boolean_rerank_support(item: dict, text: str) -> float:
    score = (
        0.45 * _boolean_value(item.get("directly_entails_claim"))
        + 0.20 * _boolean_value(item.get("contains_claim_specific_result"))
        + 0.20 * _boolean_value(item.get("contains_explicit_finding"))
        + 0.15 * (str(item.get("source_type", "unknown")).casefold() == "primary")
    )
    if _boolean_value(item.get("requires_unsupported_inference")):
        score -= 0.35
    return _apply_rerank_caps(item, text, score)


def _legacy_rerank_support(item: dict, text: str) -> float:
    support = _score_value(item.get("support"))
    metadata = _is_metadata_text(text)
    quality_fields = (
        "claim_specificity",
        "result_presence",
        "primary_source",
        "background_penalty",
        "secondary_citation_penalty",
        "unsupported_inference_penalty",
    )
    if not any(field in item for field in quality_fields) and "relation" not in item:
        return min(support, 0.05) if metadata else support

    score = (
        0.40 * support
        + 0.20 * _score_value(item.get("claim_specificity"))
        + 0.20 * _score_value(item.get("result_presence"))
        + 0.15 * _score_value(item.get("primary_source"))
        - 0.25 * _score_value(item.get("unsupported_inference_penalty"))
    )
    return _apply_rerank_caps(item, text, score)


def _apply_rerank_caps(item: dict, text: str, score: float) -> float:
    relation = str(item.get("relation", "")).casefold().strip()
    role = str(item.get("evidence_role", "")).casefold().strip()
    metadata = _is_metadata_text(text)
    if relation == "neutral":
        score = min(score, 0.30)
    elif relation == "contradicts":
        score = min(score, 0.15)
    if role in {"objective", "method", "background", "definition"}:
        score = min(score, 0.35)
    elif role == "secondary":
        score = min(score, 0.45)
    if metadata:
        score = min(score, 0.05)
    return max(0.0, min(1.0, score))


def _boolean_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.casefold().strip() in {"true", "yes", "1"}
    return bool(value) if isinstance(value, (int, float)) else False


def _score_value(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _is_metadata_text(text: str) -> bool:
    """Recognize administrative PDF text that cannot support a research claim."""

    normalized = " ".join(text.casefold().split())
    if not normalized or len(normalized.split()) > 60:
        return False
    markers = (
        "published by ",
        "copyright ",
        "all rights reserved",
        "street, ",
        " avenue, ",
        " boulevard, ",
        " oxford ox",
        " malden, ma ",
    )
    return any(marker in f" {normalized}" for marker in markers)


def select_evidence(
    claim: str,
    hit: ScoredChunk,
    context: str,
    *,
    complete: Complete,
    components: Sequence[str] | None = None,
) -> dict[str, object]:
    chunk, _score = hit
    messages = [
        {
            "role": "system",
            "content": (
                "Assess the relation between CLAIM and CONTEXT using only the source text. "
                "Return relation as supports, partial, contradicts, mixed, or insufficient. "
                "Use partial when the passage directly supports a narrower claim but does not "
                "establish a required population, unit, outcome, geographic, or time scope; for "
                "example, market-level results do not by themselves establish settlement-level "
                "effects. Use supports only when the evidence establishes the claim at its stated "
                "scope. Choose the "
                "shortest exact quote that directly states the evidence relevant to the claim. "
                "Prefer an explicit finding/result sentence over a nearby methods, data-source, "
                "caption, or interpretation sentence when the claim is about a result. Do not "
                "select text merely because it mentions related entities or methods. If the "
                "context contains only topical methods/background and no direct result for the "
                "claim, set relation to insufficient. Classify quote_role and quote_directness; "
                "a non-insufficient assessment requires quote_directness=direct. The quote must "
                "be exact text copied from CONTEXT; reason must be concise and source-grounded. "
                "Break CLAIM into its material factual components and assess each separately. "
                "Return at least two components when CLAIM has multiple clauses or qualifiers; "
                "never return an empty components list. "
                "Include the core outcome, the claimed relationship, and every material "
                "qualifier such as population, unit, geography, time, and magnitude. If "
                "COMPONENTS are supplied, use those exact component names. For every component, "
                "return established only when this passage establishes it at the claimed scope "
                "and provide an exact component-specific evidence_quote copied from CONTEXT. "
                "For partial or contradicted components, include the exact quote that supports "
                "that classification; leave evidence_quote empty only when no relevant source "
                "text is present. "
                "partial when it establishes a narrower/different scope; unresolved when absent; "
                "contradicted only when the source explicitly conflicts. A material qualifier "
                "mismatch must prevent the whole claim from being called supported. In particular, "
                "evaluate magnitude and comparison words such as minor, major, large, or strong "
                "as material claim components: a number alone does not establish that an effect "
                "is major or minor without an explicit source comparison or benchmark. Each "
                "component reason must be concise and consistent with its status. If the reason "
                "says the source does not establish, quantify, demonstrate, or measure a required "
                "part, do not label that component established. Do not combine a positive finding "
                "and an unresolved caveat under an established status. "
                "distinguish conflict along routes serving a market from conflict occurring near "
                "the market or settlement. Also return scope fields population, unit, outcome, "
                "geography, and time. "
                "Copy only what the context explicitly states; leave a field empty when unstated. "
                "Do not infer contradiction from missing evidence. Use insufficient when the "
                "context is topical but does not establish a relation, including when a table, "
                "caption, or example cannot establish the claim's population, unit, outcome, "
                "geography, or time scope. Use mixed only when the context contains both "
                "supporting and contradicting evidence. A non-insufficient relation requires a "
                "quote that itself bears on the claim."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CLAIM:\n{claim}\n\nSOURCE:\n{citation_label(chunk.paper)}, "
                f"page {chunk.page or 'unknown'}\n"
                f"\nCHUNK ROLE: {chunk.evidence_role or 'unknown'}"
                f"\nCONTENT KIND: {chunk.content_kind}\n"
                + (
                    "\nCOMPONENTS:\n" + "\n".join(f"- {value}" for value in components)
                    if components
                    else ""
                )
                + f"\n\nCONTEXT:\n{context}"
            ),
        },
    ]
    payload = _json_object(_complete_json(complete, messages, _EVIDENCE_SCHEMA))
    relation = str(payload.get("relation", "insufficient"))
    if relation not in {"supports", "partial", "contradicts", "mixed", "insufficient"}:
        relation = "insufficient"
    quote = str(payload.get("quote", "")).strip()
    quote_directness = str(payload.get("quote_directness", "unrelated"))
    if relation != "insufficient" and (
        quote_directness != "direct" or not _is_valid_quote(quote, context)
    ):
        relation = "insufficient"
        quote = ""
    raw_scope = payload.get("scope", {})
    raw_scope = raw_scope if isinstance(raw_scope, dict) else {}
    scope = {
        key: str(raw_scope.get(key, "")).strip()
        for key in ("population", "unit", "outcome", "geography", "time")
    }
    return {
        "relation": relation,
        "matched_quote": " ".join(quote.split()) if quote else "",
        "reason": str(payload.get("reason", "")).strip(),
        "scope": scope,
        "quote_role": str(payload.get("quote_role", "other")),
        "quote_directness": quote_directness,
        "components": _normalise_claim_components(payload.get("components", []), context),
    }


def _normalise_claim_components(value: object, context: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    allowed = {"established", "partial", "unresolved", "contradicted"}
    components = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = " ".join(str(item.get("name", "")).split())
        status = str(item.get("status", "unresolved")).casefold()
        evidence_quote = " ".join(str(item.get("evidence_quote", "")).split())
        if not name or name.casefold() in seen or status not in allowed:
            continue
        reason = " ".join(str(item.get("reason", "")).split())
        if status == "established" and not contains_text(context, evidence_quote):
            status = "unresolved"
            reason = "No exact source quote was supplied for this component."
        components.append(
            {
                "name": name,
                "status": status,
                "reason": reason,
                "evidence_quote": evidence_quote,
            }
        )
        seen.add(name.casefold())
    return components


def explain(claim: str, evidence: str, *, complete: Complete) -> str:
    content = complete(
        [
            {
                "role": "system",
                "content": (
                    "Explain only what the supplied evidence explicitly states about the draft "
                    "claim. Every factual statement must be directly supported by the evidence; "
                    "do not add background knowledge, infer mechanisms, causality, comparisons, "
                    "or details that the evidence does not state. In particular, do not turn a "
                    "descriptive association into a causal effect or infer relative effects from "
                    "examples, figures, or labels. If the evidence does not establish part of "
                    "the claim, say so plainly and distinguish that limitation from what it does "
                    "establish. Preserve the stated population, unit of analysis, geographic "
                    "scale, and outcome. Return only one concise paragraph, without labels, "
                    "bullets, numbering, or markdown."
                ),
            },
            {"role": "user", "content": f"DRAFT:\n{claim}\n\nEVIDENCE:\n{evidence}"},
        ],
        json_mode=False,
    ).strip()
    return _strip_label_prefix(content)


def template_rationale(_claim: str, _evidence: str) -> str:
    return "Retrieved evidence overlaps with the draft passage; verify the source before citing."


def _complete_json(
    complete: Complete,
    messages: Sequence[Mapping[str, object]],
    schema: Mapping[str, object],
) -> str:
    """Request schema-constrained JSON when the injected provider supports it."""

    try:
        parameters = signature(complete).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_schema = "response_schema" in parameters or any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_schema:
        return complete(messages, json_mode=True, response_schema=dict(schema))
    return complete(messages, json_mode=True)


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
    if not normalized_quote or not contains_text(context, normalized_quote):
        return False
    if _is_metadata_text(normalized_quote):
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
