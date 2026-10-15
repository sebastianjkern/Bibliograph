"""Normalization and comparison policies for passage assessments."""

from __future__ import annotations

import re

from ..text_matching import contains_text

_COMPONENT_GAP_PATTERNS = (
    re.compile(
        r"\b(?:does|do|did) not\s+(?:directly\s+)?"
        r"(?:establish|demonstrate|show|support|quantif\w*|measure|link|relate)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:not|never)\s+(?:directly\s+)?"
        r"(?:established|demonstrated|shown|supported|quantified|measured)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:cannot|can't)\s+(?:establish|demonstrate|show|support|quantif\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:direct|clear|specific)\s+(?:evidence|measurement|comparison)\b",
        re.IGNORECASE,
    ),
)
_COMPONENT_PARTIAL_MARKERS = re.compile(
    r"\b(?:narrower|limited to|specific (?:districts|markets|settlements|groups)|"
    r"only (?:for|among|in)|in part|partially|while .*? shows|but|however|although|yet)\b",
    re.IGNORECASE,
)
_COMPONENT_POSITIVE_MARKERS = re.compile(
    r"\b(?:shows?|reports?|finds?|estimates?|quantif\w*|indicates?|describes?|states?)\b",
    re.IGNORECASE,
)


def _normalise_assessment(selection, context: str) -> dict[str, object]:
    if selection is None:
        return {
            "relation": "insufficient",
            "matched_quote": "",
            "reason": "",
            "scope": {},
            "components": [],
        }
    if isinstance(selection, tuple):
        quote, reason = selection
        selection = {
            "relation": "supports",
            "matched_quote": quote,
            "reason": reason,
            "quote_directness": "direct",
        }
    if not isinstance(selection, dict):
        return {
            "relation": "insufficient",
            "matched_quote": "",
            "reason": "",
            "scope": {},
            "components": [],
        }
    relation = str(selection.get("relation", "insufficient"))
    if relation == "unavailable":
        return _unavailable_assessment(str(selection.get("reason", "")))
    quote_directness = str(selection.get("quote_directness", "unrelated"))
    quote = " ".join(str(selection.get("matched_quote", "")).split())

    if relation not in {"supports", "partial", "contradicts", "mixed"}:
        relation = "insufficient"
        quote = ""
    elif quote_directness != "direct" or not quote or not contains_text(context, quote):
        relation = "insufficient"
        quote = ""
    raw_scope = selection.get("scope", {})
    raw_components = selection.get("components", [])
    components = []
    if isinstance(raw_components, list):
        seen: set[str] = set()
        allowed = {"established", "partial", "unresolved", "contradicted"}
        for component in raw_components:
            if not isinstance(component, dict):
                continue
            name = " ".join(str(component.get("name", "")).split())
            status = str(component.get("status", "unresolved")).casefold()
            evidence_quote = " ".join(str(component.get("evidence_quote", "")).split())
            if not name or name.casefold() in seen or status not in allowed:
                continue
            component_reason = " ".join(str(component.get("reason", "")).split())
            if status == "established" and not contains_text(context, evidence_quote):
                status = "unresolved"
                component_reason = "No exact source quote was supplied for this component."
            components.append(
                {
                    "name": name,
                    "status": status,
                    "reason": component_reason,
                    "evidence_quote": evidence_quote,
                }
            )
            seen.add(name.casefold())
    for component in components:
        if component["status"] != "established":
            continue
        component_reason = str(component["reason"])
        has_explicit_gap = any(
            pattern.search(component_reason) for pattern in _COMPONENT_GAP_PATTERNS
        )
        if relation == "insufficient" or has_explicit_gap:
            has_positive_basis = bool(_COMPONENT_POSITIVE_MARKERS.search(component_reason))
            has_scope_qualification = bool(_COMPONENT_PARTIAL_MARKERS.search(component_reason))
            component["status"] = (
                "partial"
                if relation != "insufficient" and has_positive_basis and has_scope_qualification
                else "unresolved"
            )
            if not component_reason:
                component["reason"] = "The passage does not establish this component."
    component_statuses = {component["status"] for component in components}
    if relation == "supports" and component_statuses.intersection({"partial", "unresolved"}):
        relation = "partial"
    if relation in {"supports", "partial"} and "contradicted" in component_statuses:
        relation = "mixed" if "established" in component_statuses else "contradicts"
    reason = str(selection.get("reason", ""))
    if relation != str(selection.get("relation", "insufficient")):
        qualifiers = [
            f"{component['name']}: {component['reason'] or component['status']}"
            for component in components
            if component["status"] in {"partial", "unresolved", "contradicted"}
        ]
        if qualifiers:
            detail = "Component check qualifies the passage-level verdict: " + "; ".join(qualifiers)
            reason = f"{reason} {detail}".strip()
    return {
        "relation": relation,
        "matched_quote": quote,
        "reason": reason,
        "scope": raw_scope if isinstance(raw_scope, dict) else {},
        "quote_role": str(selection.get("quote_role", "other")),
        "quote_directness": quote_directness,
        "components": components,
    }


def _unavailable_assessment(reason: str = "") -> dict[str, object]:
    return {
        "relation": "unavailable",
        "assessment_status": "provider_unavailable",
        "matched_quote": "",
        "reason": reason
        or "Evidence assessment could not be completed because the model provider was unavailable.",
        "scope": {},
        "components": [],
    }


def _assessment_classification_signature(assessment: dict[str, object]) -> tuple:
    """Return only verdict labels, excluding explanatory wording."""
    components = assessment.get("components", [])
    component_labels = (
        tuple(
            sorted(
                (str(component.get("name", "")).casefold(), str(component.get("status", "")))
                for component in components
                if isinstance(component, dict)
            )
        )
        if isinstance(components, list)
        else ()
    )
    return str(assessment.get("relation", "insufficient")), component_labels


def _component_names(assessments: dict[str, dict[str, object] | None]) -> list[str]:
    for assessment in assessments.values():
        if assessment is None:
            continue
        components = assessment.get("components", [])
        if isinstance(components, list):
            names = [
                str(component.get("name", ""))
                for component in components
                if isinstance(component, dict) and component.get("name")
            ]
            if names:
                return names
    return []


def _assessment_gap_summary(
    assessments: dict[str, dict[str, object] | None],
) -> str:
    gaps: dict[str, str] = {}
    for assessment in assessments.values():
        if assessment is None:
            continue
        components = assessment.get("components", [])
        if isinstance(components, list):
            for component in components:
                if not isinstance(component, dict) or component.get("status") not in {
                    "partial",
                    "unresolved",
                    "contradicted",
                }:
                    continue
                name = str(component.get("name", "claim component"))
                reason = str(component.get("reason", ""))
                gaps.setdefault(name.casefold(), f"{name}: {reason}".rstrip(": "))
    if gaps:
        return "; ".join(gaps.values())
    reasons = [
        str(assessment.get("reason", "")).strip()
        for assessment in assessments.values()
        if assessment is not None
        and assessment.get("relation") in {"partial", "insufficient", "mixed", "contradicts"}
        and str(assessment.get("reason", "")).strip()
    ]
    return reasons[0] if reasons else "No specific evidence gap identified"
