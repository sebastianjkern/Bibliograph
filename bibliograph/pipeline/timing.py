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

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            with self._lock:
                self._durations[stage] += elapsed
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
            measured.append((label, seconds, int(sample.get("calls", 0))))
    measured.sort(key=lambda value: value[1], reverse=True)
    other = max(0.0, total - sum(seconds for _label, seconds, _calls in measured))
    lines = ["Time breakdown"]
    for label, seconds, calls in measured:
        share = seconds / total * 100 if total else 0.0
        lines.append(f"  {label:<23} {seconds:>7.2f}s · {calls} call(s) · {share:.0f}%")
    lines.append(f"  {'Other operations':<23} {other:>7.2f}s")
    if verbose:
        lines.append("Detailed timings: " + format_timings(timings))
    return "\n".join(lines)
