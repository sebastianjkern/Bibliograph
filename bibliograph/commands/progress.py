"""Agent-style progress helpers for long-running CLI workflows."""

from __future__ import annotations

import re
from typing import Any

from rich.console import Console
from rich.progress import Progress


class StageProgress:
    """Keep completed workflow stages visible above the current progress task."""

    def __init__(self, progress: Progress, task_id: Any) -> None:
        self.progress = progress
        self.task_id = task_id
        self.console = progress.console or Console(stderr=True)
        self._stage: str | None = None
        self._description: str | None = None

    def update(self, description: str) -> None:
        stage = _stage_name(description)
        if self._stage is not None and stage != self._stage:
            self.console.print(f"[green]✓[/green] {self._description}")
        self._stage = stage
        self._description = description
        self.progress.update(self.task_id, description=description, refresh=True)

    def complete(self, description: str) -> None:
        if self._description is not None:
            self.console.print(f"[green]✓[/green] {self._description}")
        self._stage = _stage_name(description)
        self._description = description
        self.progress.update(self.task_id, description=description, refresh=True)


def _stage_name(description: str) -> str:
    normalized = re.sub(r"^\[\d+/\d+\]\s*", "", description)
    return normalized.split(" · ", 1)[0]
