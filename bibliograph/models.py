from dataclasses import dataclass, field


@dataclass(frozen=True)
class Paper:
    zotero_key: str
    title: str
    authors: tuple[str, ...] = ()
    year: str | None = None
    doi: str | None = None
    collections: tuple[str, ...] = ()

    @property
    def citation_label(self) -> str:
        author = self.authors[0] if self.authors else self.title
        suffix = f" ({self.year})" if self.year else ""
        return f"{author}{suffix}"


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    paper: Paper
    text: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0


@dataclass(frozen=True)
class CitationSource:
    chunk: TextChunk
    score: float


@dataclass(frozen=True)
class DraftMatch:
    draft_text: str
    sources: tuple[CitationSource, ...] = field(default_factory=tuple)

