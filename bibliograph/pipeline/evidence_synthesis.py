"""Cross-paper summaries and claim component coverage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict


class _ComponentAssessment(TypedDict):
    status: str
    reason: str
    evidence_quote: str
    source_title: str
    zotero_key: str
    citation_key: str | None
    page: int | None


class _ComponentGroup(TypedDict):
    name: str
    assessments: list[_ComponentAssessment]
    sources: set[str]


def _synthesize_cross_paper(items: Sequence[dict]) -> dict[str, object]:
    """Summarize passage relations by distinct paper without merging quotations."""
    sources: list[dict[str, object]] = []
    paper_relations: dict[str, set[str]] = {}
    paper_scopes: dict[str, dict[str, set[str]]] = {}
    relation_passages = {
        relation: 0 for relation in ("supports", "partial", "contradicts", "mixed", "insufficient")
    }
    scope_keys = ("population", "unit", "outcome", "geography", "time")
    for item in items:
        relation = str(item.get("evidence_relation", "not_assessed"))
        if relation not in relation_passages:
            continue
        chunk = item["chunk"]
        source_id = chunk.paper.zotero_key
        scope = item.get("evidence_scope", {})
        scope = scope if isinstance(scope, dict) else {}
        paper_scope = paper_scopes.setdefault(source_id, {key: set() for key in scope_keys})
        for key in scope_keys:
            value = str(scope.get(key, "")).strip()
            if value:
                paper_scope[key].add(value)
        paper_relations.setdefault(source_id, set()).add(relation)
        if relation in relation_passages:
            relation_passages[relation] += 1
        sources.append(
            {
                "source_id": source_id,
                "title": chunk.paper.title,
                "citation_key": chunk.paper.citation_key,
                "page": chunk.page,
                "relation": relation,
                "matched_quote": item.get("matched_excerpt"),
                "quote_role": item.get("quote_role", "other"),
                "scope": scope,
            }
        )

    relation_sources = {
        relation: sorted(
            source_id for source_id, relations in paper_relations.items() if relation in relations
        )
        for relation in relation_passages
    }
    scope_variation: dict[str, list[str]] = {}
    if len(paper_scopes) > 1:
        for key in scope_keys:
            values = {value for paper_scope in paper_scopes.values() for value in paper_scope[key]}
            if len(values) > 1:
                scope_variation[key] = sorted(values)

    paper_counts = {relation: len(source_ids) for relation, source_ids in relation_sources.items()}
    if not sources:
        summary = "No independently assessed source passages were available to synthesize."
    else:
        paper_word = "paper" if len(paper_relations) == 1 else "papers"
        summary = (
            f"Evidence from {len(paper_relations)} distinct {paper_word} across "
            f"{len(sources)} assessed passages: {relation_passages['supports']} supporting, "
            f"{relation_passages['partial']} partial, "
            f"{relation_passages['contradicts']} contradicting, "
            f"{relation_passages['mixed']} mixed, and "
            f"{relation_passages['insufficient']} unresolved."
        )
        within_paper_disagreement = any(
            "supports" in relations and "contradicts" in relations
            for relations in paper_relations.values()
        )
        if within_paper_disagreement:
            summary += " Results differ across passages within at least one paper."
        elif paper_counts["supports"] and paper_counts["contradicts"]:
            summary += " Assessed papers disagree."
        if scope_variation:
            summary += " Stated scope differs across papers."
        summary += " This is descriptive, not a pooled estimate."
    component_coverage = _aggregate_claim_components(items)
    unresolved_questions = [
        {
            "component": component["name"],
            "reason": component["reason"],
            "status": component["status"],
        }
        for component in component_coverage
        if component["status"] in {"partial", "unresolved", "contested"}
    ]
    return {
        "summary": summary,
        "conclusion": _overall_conclusion(paper_counts, component_coverage),
        "verdict": _claim_verdict(paper_counts, component_coverage),
        "paper_count": len(paper_relations),
        "passage_count": len(sources),
        "study_count": len(paper_relations),
        "study_count_basis": "counted by distinct source paper",
        "claim_coverage": component_coverage,
        "unresolved_questions": unresolved_questions,
        "revision_guidance": (
            "Narrow or qualify: "
            + "; ".join(str(question["component"]) for question in unresolved_questions)
            if unresolved_questions
            else "No material claim qualification was left unresolved by the selected evidence."
        ),
        "sources": sources,
        "source_ids_by_relation": relation_sources,
        "relation_passages": relation_passages,
        "scope_variation": scope_variation,
    }


def _aggregate_claim_components(items: Sequence[dict]) -> list[dict[str, object]]:
    grouped: dict[str, _ComponentGroup] = {}
    for item in items:
        source_id = item["chunk"].paper.zotero_key
        components = item.get("claim_components", [])
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict) or not component.get("name"):
                continue
            name = " ".join(str(component["name"]).split())
            key = name.casefold()
            entry = grouped.setdefault(
                key,
                {"name": name, "assessments": [], "sources": set()},
            )
            status = str(component.get("status", "unresolved"))
            entry["sources"].add(source_id)
            reason = " ".join(str(component.get("reason", "")).split())
            evidence_quote = " ".join(str(component.get("evidence_quote", "")).split())
            if not evidence_quote:
                evidence_quote = " ".join(
                    str(item.get("matched_excerpt") or item["chunk"].text).split()
                )
                if len(evidence_quote) > 280:
                    evidence_quote = evidence_quote[:277].rsplit(" ", 1)[0] + "…"
            entry["assessments"].append(
                {
                    "status": status,
                    "reason": reason,
                    "evidence_quote": evidence_quote,
                    "source_title": item["chunk"].paper.title,
                    "zotero_key": source_id,
                    "citation_key": item["chunk"].paper.citation_key,
                    "page": item["chunk"].page,
                }
            )

    output = []
    for entry in grouped.values():
        assessments = entry["assessments"]
        statuses = {str(assessment["status"]) for assessment in assessments}
        if "established" in statuses and "contradicted" in statuses:
            status = "contested"
        elif "contradicted" in statuses:
            status = "contradicted"
        elif "partial" in statuses:
            status = "partial"
        elif "established" in statuses:
            status = "established"
        else:
            status = "unresolved"
        relevant_statuses = {
            "contested": {"established", "contradicted"},
            "contradicted": {"contradicted"},
            "partial": {"partial"},
            "established": {"established"},
            "unresolved": {"unresolved"},
        }[status]
        reasons = list(
            dict.fromkeys(
                str(assessment["reason"])
                for assessment in assessments
                if assessment["status"] in relevant_statuses and assessment["reason"]
            )
        )
        component_evidence = list(
            {
                (
                    assessment["source_title"],
                    assessment["zotero_key"],
                    assessment["page"],
                    assessment["citation_key"],
                    assessment["status"],
                    assessment["evidence_quote"],
                ): {
                    "title": assessment["source_title"],
                    "zotero_key": assessment["zotero_key"],
                    "citation_key": assessment["citation_key"],
                    "page": assessment["page"],
                    "status": assessment["status"],
                    "quote": assessment["evidence_quote"],
                }
                for assessment in assessments
                if assessment["status"] in relevant_statuses and assessment["evidence_quote"]
            }.values()
        )
        output.append(
            {
                "name": entry["name"],
                "status": status,
                "reason": "; ".join(reasons),
                "evidence": component_evidence,
                "study_count": len(entry["sources"]),
            }
        )
    return output


def _claim_verdict(
    paper_counts: dict[str, int], component_coverage: Sequence[dict[str, object]]
) -> str:
    statuses = {str(component["status"]) for component in component_coverage}
    if "contested" in statuses or ("contradicted" in statuses and "established" in statuses):
        return "Mixed evidence across material claim components"
    contradicted = [
        str(component["name"])
        for component in component_coverage
        if component["status"] == "contradicted"
    ]
    if contradicted:
        return "Not supported — contradicted components: " + ", ".join(contradicted)
    gaps = [
        str(component["name"])
        for component in component_coverage
        if component["status"] in {"partial", "unresolved"}
    ]
    if gaps and statuses.intersection({"established", "partial"}):
        return "Partially supported — qualification needed: " + ", ".join(gaps)
    if gaps:
        return "Not established — unresolved components: " + ", ".join(gaps)
    if component_coverage and statuses == {"established"}:
        if paper_counts.get("contradicts", 0) and any(
            paper_counts.get(relation, 0) for relation in ("supports", "partial", "mixed")
        ):
            return "Mixed evidence across assessed passages and claim components"
        if paper_counts.get("contradicts", 0):
            return "Not supported by the assessed passages"
        if paper_counts.get("mixed", 0):
            return "Mixed evidence within assessed source papers"
        if paper_counts.get("partial", 0):
            return "Partially supported — passage-level scope remains unresolved"
        return "Supported across the assessed claim components"
    if paper_counts.get("partial", 0):
        return "Partially supported — some assessed passages leave material scope unresolved"
    return _overall_conclusion(paper_counts)


def _overall_conclusion(
    paper_counts: dict[str, int], component_coverage: Sequence[dict[str, object]] = ()
) -> str:
    supporting = paper_counts.get("supports", 0)
    partial = paper_counts.get("partial", 0)
    contradicting = paper_counts.get("contradicts", 0)
    mixed = paper_counts.get("mixed", 0)
    if not any((supporting, partial, contradicting, mixed)):
        return "The retrieved passages do not establish a conclusion about the claim."
    if contradicting and (supporting or partial or mixed):
        return (
            "Overall, the assessed evidence is mixed: some source passages support the claim "
            "within their stated scope, while others contradict it or report mixed findings."
        )
    if contradicting:
        return "Overall, the assessed evidence weighs against the claim within the reported scopes."
    gaps = [
        str(component["name"])
        for component in component_coverage
        if component["status"] in {"partial", "unresolved", "contested"}
    ]
    if gaps and supporting:
        return "Partially supported; unresolved claim components: " + ", ".join(gaps) + "."
    if partial:
        return (
            "Overall, the evidence is partial: the assessed passages support only part of the "
            "claim or do not establish its full scope."
        )
    if mixed:
        return "Overall, at least one source reports both supporting and contradicting evidence."
    return (
        "Overall, the assessed evidence supports the claim within the sources' reported scopes; "
        "it should not be generalized beyond those settings."
    )
