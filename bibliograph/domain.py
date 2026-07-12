"""Small, durable domain shapes shared by indexing and retrieval.

The project deliberately keeps the domain layer boring: these are the two
records that are persisted by an index.  Everything produced while a command
is running is represented by tuples or typed dictionaries instead of an
ever-growing family of transient classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict


@dataclass(frozen=True, slots=True)
class Paper:
    """The bibliographic identity shared by one or more indexed documents."""

    zotero_key: str
    title: str
    authors: tuple[str, ...] = ()
    year: str | None = None
    doi: str | None = None
    collections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Chunk:
    """One durable, searchable piece of extracted document content."""

    chunk_id: str
    paper: Paper
    text: str
    page: int | None = None
    section: str | None = None
    ordinal: int = 0
    content_kind: str = "text"

    @property
    def chunk_index(self) -> int:
        """Compatibility spelling for adapters that expose chunk positions."""

        return self.ordinal


type ScoredChunk = tuple[Chunk, float]


class Claim(TypedDict):
    """A draft claim parsed from a supported source format."""

    text: str
    line_start: NotRequired[int | None]
    line_end: NotRequired[int | None]
    citation_keys: NotRequired[tuple[str, ...]]
    source_format: NotRequired[str | None]


def citation_label(paper: Paper) -> str:
    """Return a compact, display-only citation label for ``paper``."""

    author = paper.authors[0] if paper.authors else paper.title
    return f"{author} ({paper.year})" if paper.year else author
