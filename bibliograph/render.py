"""Pure Markdown renderers for workflow result dictionaries."""

import re
from collections.abc import Iterable

from .domain import citation_label
from .text_matching import find_text_span


def render_search(
    items: Iterable[dict],
    *,
    claim: str | None = None,
    verbose: bool = False,
    synthesis: dict | None = None,
) -> str:
    items = _sorted_items(items)
    lines = ["# Local sources", ""]

    synthesis_lines = (
        _render_claim_synthesis(synthesis, heading="")
        if synthesis and synthesis.get("sources")
        else []
    )
    if not items:
        lines.extend(
            [
                "No assessed supporting or contradicting evidence was found.",
                "Retrieved passages were not sufficient to verify a relation to the claim.",
            ]
        )
    if verbose:
        paper_groups = _group_by_paper(items)
        if paper_groups:
            lines.extend(["## Source-level evidence", ""])
        for index, group in enumerate(paper_groups, start=1):
            lines.extend(_render_paper_card(index, group, verbose=True))
    if synthesis_lines:
        lines.extend(synthesis_lines)
    return "\n".join(lines).rstrip() + "\n"


def render_check(items: Iterable[dict], *, verbose: bool = False) -> str:
    items = _sorted_items(items)
    by_claim: dict[str, list[dict]] = {}
    for item in items:
        by_claim.setdefault(item["claim"]["text"], []).append(item)
    lines = [
        "# Draft claim evidence",
        "",
        f"Found {len(by_claim)} claim(s) and {len(items)} suggestion(s).",
        "",
    ]
    for claim_number, (claim, claim_items) in enumerate(by_claim.items(), start=1):
        lines.extend([f"## Claim {claim_number}", "", f"> {claim}", ""])
        synthesis = claim_items[0].get("claim_synthesis")
        if synthesis:
            lines.extend(_render_claim_synthesis(synthesis))
        for number, item in enumerate(claim_items, start=1):
            lines.extend(_render_source_card(number, item, verbose=verbose))
            lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_claim_synthesis(
    synthesis: dict, *, heading: str = "### Claim assessment"
) -> list[str]:
    assessed_papers = int(synthesis.get("study_count", synthesis.get("paper_count", 0)))
    assessed_passages = int(synthesis.get("passage_count", 0))
    components = synthesis.get("claim_coverage", [])
    quoted_papers = {
        _paper_identity(evidence)
        for component in components
        if isinstance(component, dict)
        for evidence in component.get("evidence", [])
        if isinstance(evidence, dict) and evidence.get("quote")
    }
    quoted_count = len(quoted_papers)
    quoted_label = "distinct source paper" if quoted_count == 1 else "distinct source papers"
    assessed_paper_label = "distinct source paper" if assessed_papers == 1 else "distinct source papers"
    assessed_passage_label = "passage" if assessed_passages == 1 else "passages"
    lines = ([heading, ""] if heading else []) + [
        f"**Evidence base:** {quoted_count} {quoted_label} cited in claim coverage.",
        f"**Assessment scope:** {assessed_passages} {assessed_passage_label} from {assessed_papers} "
        f"{assessed_paper_label} assessed; these counts describe review scope, not the verdict.",
    ]
    if components:
        lines.extend(["", "**Claim coverage:**", ""])
        for component in components:
            status = str(component.get("status", "unresolved")).replace("_", " ")
            status_label = status.title()
            if status.casefold() == "contradicted":
                status_label = f"⟦contradict⟧{status_label}⟦/contradict⟧"
            reason = str(component.get("reason", "")).strip()
            line = f"- {status_label} — **{component.get('name', 'Claim component')}**"
            if reason:
                line += f": {reason}"
            lines.append(line)
            for evidence in component.get("evidence", []):
                if not isinstance(evidence, dict) or not evidence.get("quote"):
                    continue
                lines.extend(
                    [
                        f"  - “{evidence['quote']}”",
                        f"    — {_source_attribution(evidence)}",
                    ]
                )
    questions = synthesis.get("unresolved_questions", [])
    if questions:
        lines.extend(["", "**Unresolved questions:**"])
        for question in questions:
            line = f"- {question.get('component', 'Claim component')}"
            if question.get("reason"):
                line += f": {question['reason']}"
            lines.append(line)
    if synthesis.get("revision_guidance"):
        lines.extend(["", "**Verdict:** " + str(synthesis["revision_guidance"])])
    coverage = _passage_coverage_accounting(str(synthesis.get("summary", "")))
    lines.extend(["", "**Assessment coverage:** " + _colorize_passage_counts(coverage), ""])
    return lines


def _paper_identity(evidence: dict) -> str:
    zotero_key = str(evidence.get("zotero_key") or "").strip()
    if zotero_key:
        return f"zotero:{zotero_key.casefold()}"
    citation_key = str(evidence.get("citation_key") or "").strip().casefold()
    title = str(evidence.get("title") or "").strip().casefold()
    return citation_key or title


def _passage_coverage_accounting(summary: str) -> str:
    evidence = re.search(
        r"Evidence from (\d+) distinct? papers? across (\d+) assessed passages?:\s*"
        r"(.*?)(?:\. Stated scope|\. This is|$)",
        summary,
    )
    if not evidence:
        evidence = re.search(
            r"Evidence from (\d+) papers? across (\d+) assessed passages?:\s*(.*?)(?:\.|$)",
            summary,
        )
    if not evidence:
        return summary
    relation_counts = {
        label: re.search(rf"(\d+) (?:paper\(s\) with |with )?{label}\b", evidence.group(3))
        for label in ("supporting", "partial", "contradicting", "mixed", "unresolved")
    }
    counts = [
        f"{match.group(1) if match else '0'} {label}"
        for label, match in relation_counts.items()
    ]
    paper_count = int(evidence.group(1))
    paper_label = "paper" if paper_count == 1 else "papers"
    return (
        f"{paper_count} {paper_label} · {evidence.group(2)} passages"
        + (" · " + " · ".join(counts) if counts else "")
    )


def _colorize_passage_counts(summary: str) -> str:
    """Mark relation counts for terminal color while preserving plain Markdown."""
    styles = {
        "supporting": "support",
        "partial": "partial",
        "contradicting": "contradict",
        "mixed": "mixed",
        "unresolved": "unresolved",
    }
    for label, marker in styles.items():
        summary = re.sub(
            rf"\b\d+ {label}\b",
            lambda match, kind=marker: f"⟦{kind}⟧{match.group()}⟦/{kind}⟧",
            summary,
            count=1,
        )
    return summary


def _group_by_paper(items: Iterable[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        paper_id = item["chunk"].paper.zotero_key
        groups.setdefault(paper_id, []).append(item)
    return list(groups.values())


def _source_attribution(source: dict) -> str:
    title = str(source.get("title") or "Source")
    page = f"p. {source['page']}" if source.get("page") else None
    citation_key = source.get("citation_key")
    key_label = f"**{citation_key}**" if citation_key else None
    details = [detail for detail in (page, key_label) if detail]
    return f"*{title}*" + (f" · {' · '.join(details)}" if details else "")


def _render_paper_card(number: int, items: list[dict], *, verbose: bool) -> list[str]:
    paper = items[0]["chunk"].paper
    title = _display_title(paper.title)
    lines = [f"### Source {number}: {title}", ""]
    lines.extend(
        [
            "| Source details | Value |",
            "| :--- | :--- |",
            f"| **Title** | {_table_cell(title)} |",
            f"| **BibTeX key** | {_table_cell(paper.citation_key or 'unknown')} |",
            f"| **DOI** | {_table_cell(paper.doi or 'unknown')} |",
            "",
        ]
    )
    for number, item in enumerate(items, start=1):
        lines.extend(_render_passage(number, item, verbose=verbose))
    return lines


def _render_passage(number: int, item: dict, *, verbose: bool) -> list[str]:
    chunk = item["chunk"]
    page = f"p. {chunk.page}" if chunk.page is not None else "page unknown"
    role = item.get("quote_role") or chunk.evidence_role
    heading = f"#### Passage {number} · {page}"
    if role:
        heading += f" · {role}"
    lines = [heading, "", "| Passage details | Value |", "| :--- | :--- |"]
    detail_rows = [("Section", chunk.section or "unknown")]
    if verbose:
        detail_rows[:0] = [
            ("Retrieval relevance", item.get("retrieval_score", item["score"])),
            ("Vector similarity", item.get("vector_score", item["score"])),
            ("Lexical closeness", item.get("lexical_score", 0.0)),
        ]
    for label, value in detail_rows:
        if isinstance(value, (int, float)):
            value = f"{value:.2f}"
        lines.append(f"| **{label}** | {_table_cell(str(value))} |")
    lines.extend(
        [
            "",
            "**Source context (matched excerpt highlighted):**",
            "",
            _highlight_matched_excerpt(
                item.get("evidence", chunk.text), item.get("matched_excerpt")
            ),
            "",
            f"**Passage-level assessment:** {item.get('evidence_relation', 'unassessed')} — "
            f"{item.get('rationale', '')}",
        ]
    )
    scope = item.get("evidence_scope", {})
    if isinstance(scope, dict):
        stated = [
            f"{label}: {_table_cell(str(scope[key]))}"
            for key, label in (
                ("population", "Population"),
                ("unit", "Unit"),
                ("outcome", "Outcome"),
                ("geography", "Geography"),
                ("time", "Time"),
            )
            if scope.get(key)
        ]
        if stated:
            lines.extend(["", "**Passage scope:** " + "; ".join(stated)])
    lines.append("")
    return lines


def _render_source_card(
    number: int, item: dict, *, include_claim: bool = False, verbose: bool = False
) -> list[str]:
    chunk = item["chunk"]
    paper = chunk.paper
    display_title = _display_title(paper.title)
    source_label = citation_label(paper)
    if not paper.authors:
        source_label = display_title
    lines = [
        f"## {number}. {source_label}",
        "",
        "| Detail | Value |",
        "| :--- | :--- |",
        f"| **Title** | {_table_cell(display_title)} |",
        f"| **BibTeX key** | {_table_cell(paper.citation_key or 'unknown')} |",
        f"| **DOI** | {_table_cell(paper.doi or 'unknown')} |",
        f"| **Page** | {chunk.page or 'unknown'} |",
    ]
    if verbose:
        lines[6:6] = [
            f"| **Retrieval relevance** | {item.get('retrieval_score', item['score']):.2f} |",
            f"| **Vector similarity** | {item.get('vector_score', item['score']):.2f} |",
            f"| **Lexical closeness** | {item.get('lexical_score', 0.0):.2f} |",
        ]
    insertion_index = len(lines)
    for label, value in (
        ("Evidence role", chunk.evidence_role),
        ("Matched quote type", item.get("quote_role")),
        ("Evidence assessment", item.get("evidence_status")),
    ):
        if value:
            lines.insert(insertion_index, f"| **{label}** | {_table_cell(str(value))} |")
            insertion_index += 1

    if include_claim:
        lines.extend(
            [
                "",
                "### Draft claim",
                f"> {item['claim']['text']}",
            ]
        )
    lines.extend(
        [
            "",
            f"### Source context · {chunk.section or 'section unknown'}",
            _highlight_matched_excerpt(item["evidence"], item.get("matched_excerpt")),
        ]
    )

    if item.get("evidence_relation"):
        lines.extend(
            [
                "",
                f"**Assessment:** {item['evidence_relation']} — {item['rationale']}",
            ]
        )
        scope = item.get("evidence_scope", {})
        if isinstance(scope, dict):
            stated_scope = [
                f"{label}: {_table_cell(str(scope[key]))}"
                for key, label in (
                    ("population", "Population"),
                    ("unit", "Unit"),
                    ("outcome", "Outcome"),
                    ("geography", "Geography"),
                    ("time", "Time"),
                )
                if scope.get(key)
            ]
            if stated_scope:
                lines.extend(["", "**Stated scope:** " + "; ".join(stated_scope)])
        lines.append("")
    elif item.get("rationale"):
        lines.extend(["", f"**Assessment:** {item['rationale']}", ""])
    return lines


def _mark_coverage_relations(summary: str) -> str:
    relation_patterns = (
        ("support", r"\d+ supporting"),
        ("partial", r"\d+ partial"),
        ("contradict", r"\d+ contradicting"),
        ("mixed", r"\d+ mixed"),
    )
    for relation, pattern in relation_patterns:
        summary = re.sub(
            pattern,
            lambda match, relation=relation: f"⟦{relation}⟧{match.group()}⟦/{relation}⟧",
            summary,
        )
    return summary


def _highlight_matched_excerpt(context: str, matched_excerpt: str | None) -> str:
    if not matched_excerpt:
        return context
    span = find_text_span(context, matched_excerpt)
    if span is None:
        return context
    start, end = span
    return f"{context[:start]}⟦highlight⟧{context[start:end]}⟦/highlight⟧{context[end:]}"


def _display_title(value: str) -> str:
    if "_" in value and value.casefold() == value:
        return value.replace("_", " ").title()
    return value


def _table_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _sorted_items(items: Iterable[dict]) -> list[dict]:
    enumerated = list(enumerate(items))
    relation_order = {
        "supports": 0,
        "contradicts": 0,
        "partial": 1,
        "mixed": 2,
        "insufficient": 3,
    }
    ordered = sorted(
        enumerated,
        key=lambda pair: (
            relation_order.get(pair[1].get("evidence_relation") or "unassessed", 4),
            -pair[1]["score"],
            pair[0],
        ),
    )
    return [item for _index, item in ordered]


def render_status(status: dict) -> str:
    lines = ["# Bibliograph status", ""]
    for key in (
        "path",
        "schema_version",
        "embedding_id",
        "dimension",
        "extraction_chunking_fingerprint",
        "last_successful_sync",
        "papers",
        "documents",
        "chunks",
        "vectors",
        "pending_failures",
        "size_bytes",
    ):
        if key in status:
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {status[key]}")
    for probe in status.get("probes", []):
        detail = probe.get("error") or probe.get("model") or "ok"
        lines.append(f"- **{probe.get('capability', 'provider')}:** {probe.get('ok')} ({detail})")
    if status.get("rebuild_required"):
        lines.extend(
            [
                "",
                "## Rebuild required",
                "This index cannot be used safely. Run `bibliograph sync --rebuild` to replace it.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_sync(summary: dict) -> str:
    lines = [
        "# Synchronization complete",
        "",
        f"- **Operation:** {'Rebuild' if summary.get('rebuild') else 'Regular sync'}",
        f"- **Collection selected:** {summary.get('collection_input', summary['collection'])}",
        f"- **Resolved Zotero key:** {summary['collection']}",
        f"- **Discovered:** {summary['discovered']}",
        f"- **Indexed:** {len(summary['indexed'])}",
        f"- **Unchanged:** {len(summary['unchanged'])}",
        f"- **Empty:** {len(summary['empty'])}",
        f"- **Missing PDFs:** {len(summary['missing'])}",
        f"- **Downloaded:** {len(summary['downloaded'])}",
    ]
    index = summary.get("index")
    if index:
        lines.extend(
            [
                f"- **Indexed documents:** {index['documents']}",
                f"- **Indexed chunks:** {index['chunks']}",
                f"- **Indexed vectors:** {index['vectors']}",
            ]
        )
    if summary["missing"]:
        lines.extend(["", "## Missing PDFs"])
        for missing in summary["missing"]:
            lines.append(f"- {missing['title']} ({missing['source_key']})")
    return "\n".join(lines).rstrip() + "\n"
