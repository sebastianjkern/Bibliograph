"""Small thread-safe stage timer shared by retrieval entry points."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class TimingRecorder:
    """Accumulate wall-clock durations by stage, including repeated stages."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._durations: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._active: dict[str, tuple[int, float]] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = perf_counter()
        with self._lock:
            active = self._active.get(stage)
            if active is None:
                self._active[stage] = (1, started)
            else:
                self._active[stage] = (active[0] + 1, active[1])
        try:
            yield
        finally:
            finished = perf_counter()
            with self._lock:
                active_count, active_started = self._active[stage]
                if active_count == 1:
                    self._durations[stage] += finished - active_started
                    del self._active[stage]
                else:
                    self._active[stage] = (active_count - 1, active_started)
                self._counts[stage] += 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        """Return a stable, JSON-serializable snapshot."""
        with self._lock:
            return {
                stage: {"seconds": seconds, "calls": self._counts[stage]}
                for stage, seconds in sorted(self._durations.items())
            }


def format_timings(timings: dict[str, dict[str, float | int]]) -> str:
    """Render compact stage totals for CLI output."""
    return " · ".join(
        f"{stage} {entry['seconds']:.2f}s ({entry['calls']}x)"
        for stage, entry in timings.items()
    )


def format_workflow_timings(
    timings: dict[str, dict[str, float | int]], *, verbose: bool = False
) -> str:
    """Render the user-facing workflow breakdown without double-counting nested timers."""
    total = float(timings.get("total", {}).get("seconds", 0.0))
    stages = (
        ("Evidence assessment", "evidence_assessment"),
        ("Reranking", "reranking"),
        ("Query expansion", "query_expansion"),
        ("Query refinement", "query_refinement"),
    )
    measured = []
    for label, key in stages:
        sample = timings.get(key, {})
        seconds = float(sample.get("seconds", 0.0))
        if seconds:
            measured.append((label, key, seconds, int(sample.get("calls", 0))))
    measured.sort(key=lambda value: value[2], reverse=True)
    other = max(0.0, total - sum(seconds for _label, _key, seconds, _calls in measured))
    lines = ["Time breakdown"]
    for label, key, seconds, calls in measured:
        share = seconds / total * 100 if total else 0.0
        detail = f"{seconds:>7.2f}s · {calls} call(s)"
        if key == "evidence_assessment" and calls:
            detail += f" · {seconds / calls:.2f}s avg"
        lines.append(f"  {label:<23} {detail} · {share:.0f}%")
    lines.append(f"  {'Other operations':<23} {other:>7.2f}s")
    other_details = _other_operation_breakdown(timings, other)
    for depth, label, seconds in other_details:
        if seconds >= 0.01:
            display_label = "  " * depth + label
            lines.append(f"    {display_label:<27} {seconds:>7.2f}s")
    if verbose:
        lines.append("Detailed timings: " + format_timings(timings))
    return "\n".join(lines)


def _other_operation_breakdown(
    timings: dict[str, dict[str, float | int]], other: float
) -> list[tuple[int, str, float]]:
    """Break the remainder into non-overlapping leaf timings where possible."""

    def seconds(stage: str) -> float:
        return float(timings.get(stage, {}).get("seconds", 0.0))

    retrieval_children = sum(
        seconds(stage)
        for stage in ("embedding", "vector_search", "lexical_search", "graph_expansion")
    )
    top_level = [
        ("Search setup", max(0.0, seconds("plan") - seconds("query_expansion"))),
        ("Retrieval", seconds("retrieval")),
        ("Context preparation", seconds("context")),
        (
            "Evidence workflow overhead",
            max(
                0.0,
                seconds("assessment_and_rerank")
                - seconds("evidence_assessment")
                - seconds("reranking"),
            ),
        ),
        ("Context extension", seconds("context_extension")),
        ("Result finalization", seconds("finish")),
    ]
    details = [(1, "Search setup", top_level[0][1]), (1, "Retrieval", seconds("retrieval"))]
    details.extend(
        [
            (2, "Query embedding", seconds("embedding")),
            (2, "Vector search", seconds("vector_search")),
            (2, "Lexical search", seconds("lexical_search")),
            (2, "Graph expansion", seconds("graph_expansion")),
            (2, "Other retrieval work", max(0.0, seconds("retrieval") - retrieval_children)),
        ]
    )
    details.extend((1, label, value) for label, value in top_level[2:])
    measured = sum(value for _label, value in top_level)
    details.append((1, "Unattributed / coordination", max(0.0, other - measured)))
    return details
