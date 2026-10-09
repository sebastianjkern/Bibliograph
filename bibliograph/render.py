"""Pure Markdown renderers for workflow result dictionaries."""

from collections.abc import Iterable

from .domain import citation_label


def render_search(items: Iterable[dict], *, claim: str | None = None) -> str:
    items = _sorted_items(items)
    claim = items[0]["claim"]["text"] if items else (claim or "")
    lines = ["# Local sources", "", f"> Claim: {claim}", ""]
    if not items:
        lines.extend(
            [
                "No matching evidence was found in the indexed PDFs.",
                "Try a broader claim or check that the PDFs have been indexed.",
            ]
        )
    for index, item in enumerate(items, start=1):
        lines.extend(_render_source_card(index, item))
    return "\n".join(lines).rstrip() + "\n"


def render_check(items: Iterable[dict]) -> str:
    items = _sorted_items(items)
    lines = ["# Citation suggestions", "", f"Found {len(items)} suggestion(s).", ""]
    for number, item in enumerate(items, start=1):
        lines.extend(_render_source_card(number, item, include_claim=True))
        lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


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
        f"| **Vector similarity** | {item.get('vector_score', item['score']):.2f} |",
        f"| **Lexical closeness** | {item.get('lexical_score', 0.0):.2f} |",
        f"| **Rerank support** | {item.get('rerank_score', item['score']):.2f} |",
        f"| **DOI** | {_table_cell(paper.doi or 'unknown')} |",
        f"| **Page** | {chunk.page or 'unknown'} |",
    ]
    if chunk.evidence_role:
        lines.insert(9, f"| **Evidence role** | {_table_cell(chunk.evidence_role)} |")

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
            "### Excerpt",
            f"> {item['evidence']}",
            "",
            "### Explanation",
            item["rationale"],
            "",
        ]
    )
    return lines


def _display_title(value: str) -> str:
    if "_" in value and value.casefold() == value:
        return value.replace("_", " ").title()
    return value


def _table_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _sorted_items(items: Iterable[dict]) -> list[dict]:
    enumerated = list(enumerate(items))
    ordered = sorted(enumerated, key=lambda pair: (-pair[1]["score"], pair[0]))
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
