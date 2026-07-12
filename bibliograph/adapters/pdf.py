"""PDF extraction, chunking, and local-file discovery.

The adapter has no knowledge of databases or embedding providers.  It turns a
PDF into durable :class:`~bibliograph.domain.Chunk` values and leaves callers
free to choose how those chunks are embedded and stored.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from math import ceil
from pathlib import Path

from bibliograph.domain import Chunk, Paper


def is_pdf_file(path: str | Path) -> bool:
    """Return whether *path* exists and starts with a PDF file signature."""
    try:
        with Path(path).open("rb") as file:
            return file.read(5) == b"%PDF-"
    except OSError:
        return False


def locate_local_pdf(
    pdf_dir: str | Path,
    source_key: str = "",
    *,
    paper_key: str | None = None,
    doi: str | None = None,
    candidates: Sequence[str | Path] = (),
) -> Path | None:
    """Find a valid cached PDF using explicit paths and stable identifiers.

    ``candidates`` is checked first.  Relative candidates are interpreted
    relative to ``pdf_dir``.  The identifier lookup supports both the new
    ``<source-key>.pdf`` cache convention and historical
    ``<title>-<source-key>.pdf`` files.
    """
    directory = Path(pdf_dir)
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = directory / candidate_path
        if is_pdf_file(candidate_path):
            return candidate_path

    keys = [key for key in (source_key, paper_key) if key]
    if doi:
        keys.append(_safe_identifier(doi))
    for key in dict.fromkeys(keys):
        exact = directory / f"{key}.pdf"
        if is_pdf_file(exact):
            return exact
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.pdf")):
            if candidate.name.endswith(f"-{key}.pdf") and is_pdf_file(candidate):
                return candidate
    return None


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str, str | None]]:
    """Extract body text as ``(page, text, section)`` triples.

    Repeated margin text, page numbers, headings, and reference lists are
    stripped so retrieval is driven by the paper body rather than boilerplate.
    PyMuPDF remains an optional runtime dependency and is imported only when
    extraction is requested.
    """
    try:
        import fitz
    except ImportError as error:  # pragma: no cover - dependency availability
        raise RuntimeError("PDF extraction requires the 'pdf' optional dependency") from error

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with fitz.open(path) as document:
        page_blocks: list[tuple[float, list[tuple[object, ...]]]] = []
        margin_lines: list[str] = []
        for page in document:
            blocks = [block for block in page.get_text("blocks") if block[4].strip()]
            page_blocks.append((page.rect.height, blocks))
            for _x0, y0, _x1, y1, block_text, *_ in blocks:
                if y0 <= page.rect.height * 0.15 or y1 >= page.rect.height * 0.85:
                    margin_lines.extend(
                        normalized
                        for line in block_text.splitlines()
                        if (normalized := _normalized_line(line))
                    )

    repeated = {
        line
        for line, count in Counter(margin_lines).items()
        if count >= max(2, ceil(len(page_blocks) * 0.5))
    }
    pages: list[tuple[int, str, str | None]] = []
    current_section: str | None = None
    references_started = False
    for page_number, (_height, blocks) in enumerate(page_blocks, start=1):
        page_text: list[str] = []
        for _x0, _y0, _x1, _y1, block_text, *_ in blocks:
            lines = [
                line.strip()
                for line in block_text.splitlines()
                if _normalized_line(line) not in repeated and not _is_page_number(line)
            ]
            text = " ".join(lines).strip()
            if not text:
                continue
            if _is_heading(text):
                if _is_reference_heading(text):
                    references_started = True
                elif not references_started:
                    current_section = text
                continue
            if not references_started:
                page_text.append(text)
        if page_text:
            pages.append((page_number, "\n\n".join(page_text), current_section))
    return pages


def chunk_document(
    paper: Paper,
    pages: Iterable[tuple[int, str] | tuple[int, str, str | None]],
    *,
    max_words: int = 120,
    overlap_words: int = 20,
    content_kind: str = "text",
) -> list[Chunk]:
    """Turn extracted pages into ordered, citation-ready durable chunks."""
    chunks: list[Chunk] = []
    ordinal = 0
    for page_data in pages:
        page, text, *section_data = page_data
        section = section_data[0] if section_data else None
        for text_chunk in _split_text(text, max_words=max_words, overlap_words=overlap_words):
            prefix = f"Section: {section}\n\n" if section else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.zotero_key}:{page}:{ordinal}",
                    paper=paper,
                    text=f"{prefix}{text_chunk}",
                    page=page,
                    section=section,
                    ordinal=ordinal,
                    content_kind=content_kind,
                )
            )
            ordinal += 1
    return chunks


def _split_text(text: str, *, max_words: int, overlap_words: int) -> list[str]:
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("max_words must be positive and overlap_words must be smaller")
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def emit() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(" ".join(current).strip())
            overlap = re.findall(r"\S+", " ".join(current))[-overlap_words:]
            current = [" ".join(overlap)] if overlap else []
            current_words = len(overlap)

    for sentence in sentences:
        sentence_words = re.findall(r"\S+", sentence)
        if len(sentence_words) > max_words:
            emit()
            for start in range(0, len(sentence_words), max_words - overlap_words):
                part = sentence_words[start : start + max_words]
                if part:
                    chunks.append(" ".join(part))
                if start + max_words >= len(sentence_words):
                    break
            current = []
            current_words = 0
        elif current and current_words + len(sentence_words) > max_words:
            emit()
            current.extend(sentence_words)
            current_words += len(sentence_words)
        else:
            current.extend(sentence_words)
            current_words += len(sentence_words)
    emit()
    return chunks


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.lower()).strip("_")


def _normalized_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().casefold()


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", line.strip()))


def _is_heading(text: str) -> bool:
    normalized = text.strip()
    words = normalized.split()
    known_headings = {
        "abstract",
        "introduction",
        "background",
        "methods",
        "methodology",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "literature cited",
        "works cited",
    }
    lower = normalized.casefold().rstrip(":")
    return (
        lower in known_headings
        or bool(re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", normalized))
        or (normalized.isupper() and 1 < len(words) <= 12)
    )


def _is_reference_heading(text: str) -> bool:
    return bool(
        re.match(r"^(references|bibliography|literature cited|works cited)\b", text.casefold())
    )
