"""Pure Markdown renderers for workflow result dictionaries."""

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
    claim = items[0]["claim"]["text"] if items else (claim or "")
    assessed = [
        item
        for item in items
        if item.get("evidence_relation") in {"supports", "partial", "contradicts", "mixed"}
    ]
    shown = items if verbose else assessed[:3]
    lines = ["# Local sources", "", f"> Claim: {claim}", ""]
    if synthesis and synthesis.get("sources"):
        lines.extend(
            [
                "## Claim-level synthesis",
                "",
                "**Conclusion:** " + str(synthesis.get("conclusion", "")),
                "",
                "**Evidence coverage:** " + str(synthesis.get("summary", "")),
                "",
            ]
        )
    if not shown:
        lines.extend(
            [
                "No assessed supporting or contradicting evidence was found.",
                "Retrieved passages were not sufficient to verify a relation to the claim.",
            ]
        )
    paper_groups = _group_by_paper(shown)
    if paper_groups:
        lines.extend(["## Source-level evidence", ""])
    for index, group in enumerate(paper_groups, start=1):
        lines.extend(_render_paper_card(index, group))
    if not verbose and len(assessed) > len(shown):
        lines.extend(
            [
                f"Showing {len(shown)} of {len(assessed)} assessed passages. "
                "Use --verbose to see all.",
                "",
            ]
        )
    if not verbose and items and not assessed:
        lines.extend([f"{len(items)} retrieved passage(s) were not assessed as evidence.", ""])
    return "\n".join(lines).rstrip() + "\n"


def render_check(items: Iterable[dict]) -> str:
    items = _sorted_items(items)
    lines = ["# Citation suggestions", "", f"Found {len(items)} suggestion(s).", ""]
    for number, item in enumerate(items, start=1):
        lines.extend(_render_source_card(number, item, include_claim=True))
        lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


def _group_by_paper(items: Iterable[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        paper_id = item["chunk"].paper.zotero_key
        groups.setdefault(paper_id, []).append(item)
    return list(groups.values())


def _render_paper_card(number: int, items: list[dict]) -> list[str]:
    paper = items[0]["chunk"].paper
    title = _display_title(paper.title)
    lines = [f"### Source {number}: {title}", ""]
    lines.extend(
        [
            "| Source details | Value |",
            "| :--- | :--- |",
            f"| **Title** | {_table_cell(title)} |",
            f"| **DOI** | {_table_cell(paper.doi or 'unknown')} |",
            "",
        ]
    )
    for number, item in enumerate(items, start=1):
        lines.extend(_render_passage(number, item))
    return lines


def _render_passage(number: int, item: dict) -> list[str]:
    chunk = item["chunk"]
    page = f"p. {chunk.page}" if chunk.page is not None else "page unknown"
    role = item.get("quote_role") or chunk.evidence_role
    heading = f"#### Passage {number} · {page}"
    if role:
        heading += f" · {role}"
    lines = [heading, "", "| Passage details | Value |", "| :--- | :--- |"]
    for label, value in (
        ("Retrieval relevance", item.get("retrieval_score", item["score"])),
        ("Vector similarity", item.get("vector_score", item["score"])),
        ("Lexical closeness", item.get("lexical_score", 0.0)),
        ("Section", chunk.section or "unknown"),
    ):
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


def _render_source_card(number: int, item: dict, *, include_claim: bool = False) -> list[str]:
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
        f"| **Retrieval relevance** | {item.get('retrieval_score', item['score']):.2f} |",
        f"| **Vector similarity** | {item.get('vector_score', item['score']):.2f} |",
        f"| **Lexical closeness** | {item.get('lexical_score', 0.0):.2f} |",
        f"| **DOI** | {_table_cell(paper.doi or 'unknown')} |",
        f"| **Page** | {chunk.page or 'unknown'} |",
    ]
    if chunk.evidence_role:
        lines.insert(9, f"| **Evidence role** | {_table_cell(chunk.evidence_role)} |")
    if item.get("quote_role"):
        lines.insert(9, f"| **Matched quote type** | {_table_cell(item['quote_role'])} |")
    if item.get("evidence_status"):
        lines.insert(9, f"| **Evidence assessment** | {_table_cell(item['evidence_status'])} |")

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
        f"- **Collection:** {summary['collection']}",
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
