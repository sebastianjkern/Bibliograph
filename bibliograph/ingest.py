import re
from collections import Counter
from collections.abc import Iterable
from math import ceil
from pathlib import Path

from .chunking import chunk_pages
from .embeddings import Embedder
from .logging_utils import get_logger
from .models import Paper
from .store import SQLiteIndex

logger = get_logger("ingest")


def extract_pdf_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract cleaned page-numbered text from a PDF using PyMuPDF."""
    return [(page, text) for page, text, _section in extract_pdf_sections(pdf_path)]


def extract_pdf_sections(pdf_path: str | Path) -> list[tuple[int, str, str | None]]:
    """Extract body text with section labels and repeated margin text removed."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires the optional 'pdf' dependencies") from exc

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with fitz.open(path) as document:
        logger.debug("Extracting %d PDF pages from %s", len(document), path)
        page_blocks = []
        margin_lines: list[str] = []
        for page in document:
            blocks = [block for block in page.get_text("blocks") if block[4].strip()]
            page_blocks.append((page.rect.height, blocks))
            for _x0, y0, _x1, y1, block_text, *_ in blocks:
                if y0 <= page.rect.height * 0.15 or y1 >= page.rect.height * 0.85:
                    margin_lines.extend(
                        _normalized_line(line)
                        for line in block_text.splitlines()
                        if _normalized_line(line)
                    )

        repeated = {
            line
            for line, count in Counter(margin_lines).items()
            if count >= max(2, ceil(len(page_blocks) * 0.5))
        }
        sections: list[tuple[int, str, str | None]] = []
        current_section: str | None = None
        references_started = False
        for page_number, (_height, blocks) in enumerate(page_blocks, start=1):
            page_text: list[str] = []
            for _x0, _y0, _x1, _y1, block_text, *_ in blocks:
                lines = [
                    line.strip()
                    for line in block_text.splitlines()
                    if _normalized_line(line)
                    not in repeated
                    and not _is_page_number(line)
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
                if references_started:
                    continue
                page_text.append(text)
            if page_text:
                sections.append((page_number, "\n\n".join(page_text), current_section))
        return sections


def index_pdf(
    index: SQLiteIndex,
    embedder: Embedder,
    paper: Paper,
    pdf_path: str | Path,
    max_words: int = 120,
    overlap_words: int = 20,
) -> int:
    pages = extract_pdf_sections(pdf_path)
    return index_chunks(index, embedder, paper, pages, max_words, overlap_words)


def index_chunks(
    index: SQLiteIndex,
    embedder: Embedder,
    paper: Paper,
    pages: Iterable[tuple[int, str] | tuple[int, str, str | None]],
    max_words: int = 120,
    overlap_words: int = 20,
) -> int:
    chunks = chunk_pages(paper, pages, max_words, overlap_words)
    if not chunks:
        logger.warning("No text chunks extracted from %s", paper.title)
        return 0
    logger.debug("Embedding %d chunks for %s", len(chunks), paper.title)
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    return index.upsert(chunks, embeddings)


def paper_from_zotero_item(item: dict, collections: Iterable[str] = ()) -> Paper:
    """Convert a Zotero item response into stable citation metadata."""
    data = item.get("data", item)
    authors = tuple(
        creator.get("lastName") or creator.get("name") or creator.get("firstName", "")
        for creator in data.get("creators", [])
        if creator.get("lastName") or creator.get("name") or creator.get("firstName")
    )
    year = data.get("date", "")[:4] or None
    return Paper(
        zotero_key=data.get("key", ""),
        title=data.get("title", "Untitled"),
        authors=authors,
        year=year,
        doi=data.get("DOI") or None,
        collections=tuple(collections),
    )


def _normalized_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().casefold()


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"[-–—\s]*\d+[-–—\s]*", line.strip()))


def _is_heading(text: str) -> bool:
    normalized = text.strip()
    words = normalized.split()
    known = {
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
        lower in known
        or bool(re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", normalized))
        or (normalized.isupper() and 1 < len(words) <= 12)
    )


def _is_reference_heading(text: str) -> bool:
    return bool(
        re.match(r"^(references|bibliography|literature cited|works cited)\b", text.casefold())
    )
