"""Read-only index and optional capability status workflow."""

from collections.abc import Callable, Iterable
from pathlib import Path

from ..render import render_status


def status(
    store=None,
    *,
    path: str | Path | None = None,
    probes: Iterable[Callable[[], dict]] = (),
) -> dict:
    summary = (
        dict(store.stats())
        if store is not None
        else {
            "path": str(path or "bibliograph.db"),
            "schema_version": 0,
            "papers": 0,
            "documents": 0,
            "chunks": 0,
            "vectors": 0,
            "embedding_id": None,
            "dimension": None,
            "extraction_chunking_fingerprint": None,
            "last_successful_sync": None,
            "pending_failures": 0,
            "rebuild_required": False,
        }
    )
    path = Path(summary["path"])
    summary["size_bytes"] = path.stat().st_size if path.is_file() else 0
    results = []
    for probe in probes:
        try:
            results.append(probe())
        except Exception as error:
            results.append({"ok": False, "capability": "provider", "error": str(error)})
    if results:
        summary["probes"] = results
    summary["markdown"] = render_status(summary)
    return summary
