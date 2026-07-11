from collections.abc import Iterable
from pathlib import Path

from .chunking import chunk_pages
from .embeddings import Embedder
from .logging_utils import get_logger
from .models import Paper
from .store import SQLiteIndex

logger = get_logger("ingest")


def extract_pdf_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Extract page-numbered text from a PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires the optional 'pdf' dependencies") from exc

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with fitz.open(path) as document:
        logger.debug("Extracting %d PDF pages from %s", len(document), path)
        return [
            (page_number + 1, page.get_text("text").strip())
            for page_number, page in enumerate(document)
        ]


def index_pdf(
    index: SQLiteIndex,
    embedder: Embedder,
    paper: Paper,
    pdf_path: str | Path,
    max_words: int = 180,
    overlap_words: int = 30,
) -> int:
    pages = extract_pdf_pages(pdf_path)
    return index_chunks(index, embedder, paper, pages, max_words, overlap_words)


def index_chunks(
    index: SQLiteIndex,
    embedder: Embedder,
    paper: Paper,
    pages: Iterable[tuple[int, str]],
    max_words: int = 180,
    overlap_words: int = 30,
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
