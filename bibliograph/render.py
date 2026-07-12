"""Pure Markdown renderers for workflow result dictionaries."""

from collections.abc import Iterable

from .domain import citation_label


def render_search(items: Iterable[dict]) -> str:
    items = list(items)
    claim = items[0]["claim"]["text"] if items else ""
    lines = ["# Local sources", "", f"> Claim: {claim}", ""]
    for index, item in enumerate(items, start=1):
        lines.extend(_render_source_card(index, item))
    return "\n".join(lines).rstrip() + "\n"


def render_check(items: Iterable[dict]) -> str:
    items = list(items)
    lines = ["# Citation suggestions", "", f"Found {len(items)} suggestion(s).", ""]
    for number, item in enumerate(items, start=1):
        lines.extend(_render_source_card(number, item, include_claim=True))
        lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_source_card(number: int, item: dict, *, include_claim: bool = False) -> list[str]:
    chunk = item["chunk"]
    paper = chunk.paper
    lines = [
        f"## {number}. {citation_label(paper)}",
        "",
        f"- **Title:** {paper.title}",
        f"- **Support score:** `{item['score']:.2f}`",
        f"- **DOI:** {paper.doi or 'unknown'}",
        f"- **Page:** {chunk.page or 'unknown'}",
    ]
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
            "### Why this fits",
            item["rationale"],
            "",
        ]
    )
    return lines


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
