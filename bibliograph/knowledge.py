"""Neutral contracts between Bibliograph and a knowledge-engine backend.

Bibliograph owns Zotero and citation-domain objects.  Knowledge engines such as
Ikarus own ingestion, indexing, query planning, graph expansion, and retrieval.
This module deliberately contains no Ikarus imports so the application layer is
not coupled to one engine's concrete types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .domain import Chunk, Paper


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """A Bibliograph document translated into a knowledge-engine input."""

    source_key: str
    paper: Paper
    path: Path
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryHypothesis:
    """A typed alternative query produced by Bibliograph's query policy."""

    text: str
    kind: str = "alternative"
    weight: float = 1.0
    entities: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryRequest:
    """Domain-neutral query plan supplied to a knowledge-engine retrieval operation."""

    text: str
    hypotheses: tuple[QueryHypothesis, ...] = ()
    filters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_alternatives(
        cls,
        text: str,
        alternatives: Sequence[str] = (),
    ) -> QueryRequest:
        return cls(
            text=text,
            hypotheses=tuple(QueryHypothesis(value) for value in alternatives),
        )


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Optional provenance returned by a knowledge engine for UI/debugging."""

    nodes: tuple[Mapping[str, Any], ...] = ()
    edges: tuple[Mapping[str, Any], ...] = ()
    paths: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Engine result before Bibliograph-specific evidence presentation."""

    hits: tuple[tuple[Chunk, float], ...]
    score_details: Mapping[str, Mapping[str, float]]
    trace: RetrievalTrace | None = None


class KnowledgeBackend(Protocol):
    """The application-facing seam for an Ikarus-like knowledge engine."""

    def index_document(
        self,
        document: KnowledgeDocument,
        *,
        extract_pages: Any,
    ) -> Mapping[str, Any]:
        ...

    def retrieve_request(
        self,
        request: QueryRequest,
        *,
        limit: int = 10,
        include_trace: bool = False,
    ) -> RetrievalResult:
        ...
