"""Persistent SQLite caches for evidence assessment and reranking requests."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def assessment_cache_key(payload: dict[str, Any]) -> str:
    """Hash the complete assessment input without storing source text in the key."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SQLiteAssessmentCache:
    """Small sidecar cache that remains usable with a read-only index connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.path), timeout=10) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evidence_assessments ("
                "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS rerankings ("
                "cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            with sqlite3.connect(str(self.path), timeout=10) as connection:
                row = connection.execute(
                    "SELECT payload FROM evidence_assessments WHERE cache_key = ?", (key,)
                ).fetchone()
            if row is None:
                return None
            value = json.loads(row[0])
            return value if isinstance(value, dict) else None
        except (OSError, sqlite3.Error, json.JSONDecodeError) as error:
            logger.warning("Evidence assessment cache read failed: %s", error)
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
            with sqlite3.connect(str(self.path), timeout=10) as connection:
                connection.execute(
                    "INSERT INTO evidence_assessments(cache_key, payload) VALUES (?, ?) "
                    "ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (key, payload),
                )
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            logger.warning("Evidence assessment cache write failed: %s", error)

    def get_reranking(self, key: str) -> list[dict[str, Any]] | None:
        try:
            with sqlite3.connect(str(self.path), timeout=10) as connection:
                row = connection.execute(
                    "SELECT payload FROM rerankings WHERE cache_key = ?", (key,)
                ).fetchone()
            if row is None:
                return None
            value = json.loads(row[0])
            return value if isinstance(value, list) and all(
                isinstance(item, dict) for item in value
            ) else None
        except (OSError, sqlite3.Error, json.JSONDecodeError) as error:
            logger.warning("Reranking cache read failed: %s", error)
            return None

    def set_reranking(self, key: str, value: list[dict[str, Any]]) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
            with sqlite3.connect(str(self.path), timeout=10) as connection:
                connection.execute(
                    "INSERT INTO rerankings(cache_key, payload) VALUES (?, ?) "
                    "ON CONFLICT(cache_key) DO UPDATE SET payload=excluded.payload, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (key, payload),
                )
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            logger.warning("Reranking cache write failed: %s", error)
