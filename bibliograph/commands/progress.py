"""Agent-style progress helpers for long-running CLI workflows."""

from __future__ import annotations

import re
from typing import Any

from rich.progress import Progress

from ..logging_utils import stderr_console


class StageProgress:
    """Keep completed workflow stages visible above the current progress task."""

    def __init__(self, progress: Progress, task_id: Any) -> None:
        self.progress = progress
        self.task_id = task_id
        self.console = progress.console or stderr_console()
        self._stage: str | None = None
        self._full_description: str | None = None
        self._stage_description: str | None = None

    def update(self, description: str) -> None:
        stage = _stage_name(description)
        # Print full description as a completed item when stage changes
        if self._stage is not None and stage != self._stage:
            self.console.print(f"[green]✓[/green] {self._full_description}")
        self._stage = stage
        self._full_description = description
        # Use just the stage name for progress bar to prevent jumping
        self._stage_description = stage
        self.progress.update(self.task_id, description=self._stage_description, refresh=True)

    def complete(self, description: str) -> None:
        if self._full_description is not None:
            self.console.print(f"[green]✓[/green] {self._full_description}")
        self._stage = _stage_name(description)
        self._full_description = description
        self._stage_description = description
        self.progress.update(self.task_id, description=self._stage_description, refresh=True)


def _stage_name(description: str) -> str:
    normalized = re.sub(r"^\[\d+/\d+\]\s*", "", description)
    return normalized.split(" · ", 1)[0]
