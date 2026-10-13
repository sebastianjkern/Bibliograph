"""LangGraph workflow for hybrid PDF extraction and layout-aware cleanup."""

from __future__ import annotations

import base64
from io import BytesIO
import operator
import re
from collections.abc import Callable
from threading import Lock
from typing import Annotated
from pathlib import Path
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..adapters.ollama_ocr import _has_usable_text, _recognize_page
from ..adapters.pdf import _is_heading, _is_page_number, _is_reference_heading

_DROP_CATEGORIES = {
    "page_header",
    "page_footer",
    "page_number",
    "reference",
    "other",
    "furniture",
}
_DOCLING_LABELS = {
    "caption": "caption",
    "footnote": "footnote",
    "formula": "formula",
    "page_footer": "page_footer",
    "page_header": "page_header",
    "reference": "reference",
    "section_header": "section_header",
    "table": "table",
    "picture": "figure",
    "title": "title",
    "text": "text",
    "paragraph": "text",
    "list_item": "text",
    "code": "text",
}


class Block(TypedDict):
    id: str
    text: str
    bbox: tuple[float, float, float, float] | None
    source_kind: str


class PageState(TypedDict, total=False):
    page: Any
    page_number: int
    page_count: int
    progress: Callable[[str], None] | None
    mode: str
    model: str
    base_url: str
    timeout: float
    min_text_chars: int
    blocks: list[Block]
    image: str
    native_text: str
    needs_ocr: bool
    categories: dict[str, str]
    result: tuple[int, str, str | None] | None
    section: str | None
    references_started: bool
    verbose: bool
    prepared: bool


class OcrBatchState(TypedDict, total=False):
    ocr_tasks: list[dict[str, Any]]
    ocr_results: Annotated[list[dict[str, Any]], operator.add]


class IngestionState(TypedDict, total=False):
    root: str
    recursive: bool
    pdf_paths: list[str]
    summary: dict[str, Any]


_PYMUPDF_LOCK = Lock()


def make_hybrid_page_extractor(
    *,
    model: str,
    base_url: str = "http://localhost:11434",
    mode: str = "hybrid",
    min_text_chars: int = 120,
    timeout: float = 300,
    filesystem=None,
    verbose: bool = False,
) -> Callable[[str | Path], list[tuple[int, str, str | None]]]:
    """Build the page subgraph and adapt it to Bibliograph's extractor seam."""
    if mode not in {"hybrid", "all"}:
        raise ValueError("PDF OCR mode must be 'hybrid' or 'all'")
    if not model.strip():
        raise ValueError("Ollama OCR model must be a non-empty model name")
    if min_text_chars < 1:
        raise ValueError("min_text_chars must be positive")

    if filesystem is None:
        from adapters.providers.filesystems.fsspec import LocalFilesystem

        filesystem = LocalFilesystem()

    page_graph = _build_page_graph()

    def extract(
        pdf_path: str | Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> list[tuple[int, str, str | None]]:
        try:
            import fitz
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("PDF ingestion requires the 'pdf' optional dependency") from error

        with filesystem.open(str(pdf_path), "rb") as stream:
            pdf_bytes = stream.read()
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF file: {pdf_path}")

        page_inputs: list[dict[str, Any]] = []
        ocr_tasks: list[dict[str, Any]] = []
        with _PYMUPDF_LOCK:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                page_count = len(document)
                for page_number, page in enumerate(document, start=1):
                    native_text = page.get_text("text").strip()
                    blocks = _native_blocks(page)
                    needs_ocr = mode == "all" or not _has_usable_text(
                        native_text, min_text_chars
                    )
                    image = ""
                    if needs_ocr:
                        image_bytes = page.get_pixmap(
                            matrix=fitz.Matrix(2, 2), alpha=False
                        ).tobytes("png")
                        image = base64.b64encode(image_bytes).decode("ascii")
                    page_input = {
                        "page_number": page_number,
                        "page_count": page_count,
                        "native_text": native_text,
                        "image": image,
                        "blocks": blocks,
                        "needs_ocr": needs_ocr,
                    }
                    page_inputs.append(page_input)
                    if needs_ocr:
                        ocr_tasks.append(
                            {
                                **page_input,
                                "model": model,
                                "base_url": base_url,
                                "timeout": timeout,
                                "progress": progress,
                            }
                        )

        ocr_by_page: dict[int, dict[str, Any]] = {}
        if ocr_tasks:
            ocr_results = _build_ocr_batch_graph().invoke(
                {"ocr_tasks": ocr_tasks},
                config={"max_concurrency": len(ocr_tasks)},
            )
            ocr_by_page = {
                item["page_number"]: item
                for item in ocr_results.get("ocr_results", [])
            }

        results = []
        section: str | None = None
        references_started = False
        for page_input in page_inputs:
            page_number = page_input["page_number"]
            prepared = dict(page_input)
            prepared["prepared"] = True
            prepared["needs_ocr"] = False
            ocr_result = ocr_by_page.get(page_number)
            if ocr_result is not None:
                prepared["blocks"] = ocr_result["blocks"]
                prepared["categories"] = ocr_result["categories"]
            result = page_graph.invoke(
                {
                    **prepared,
                    "progress": progress,
                    "mode": mode,
                    "model": model,
                    "base_url": base_url,
                    "timeout": timeout,
                    "min_text_chars": min_text_chars,
                    "section": section,
                    "references_started": references_started,
                    "verbose": verbose,
                }
            )
            page_result = result.get("result")
            section = result.get("section") or section
            references_started = result.get("references_started", references_started)
            if page_result is not None:
                results.append(page_result)
        return results

    extract.indexing_fingerprint = (
        f"langgraph-pdf-doctags-v1:{model}:{mode}:"
        f"min-text={min_text_chars}"
    )
    return extract


def _build_page_graph():
    def inspect(state: PageState) -> dict[str, Any]:
        if state.get("prepared"):
            return {
                "native_text": state.get("native_text", ""),
                "image": state.get("image", ""),
                "blocks": state.get("blocks", []),
                "needs_ocr": state.get("needs_ocr", False),
                "categories": state.get("categories"),
            }
        import fitz

        page = state["page"]
        native_text = page.get_text("text").strip()
        image_bytes = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        blocks = _native_blocks(page)
        needs_ocr = state["mode"] == "all" or not _has_usable_text(
            native_text, state["min_text_chars"]
        )
        return {
            "native_text": native_text,
            "image": base64.b64encode(image_bytes).decode("ascii"),
            "blocks": blocks,
            "needs_ocr": needs_ocr,
        }

    def route_after_inspect(state: PageState) -> str:
        if state["needs_ocr"]:
            reason = "no embedded text" if not state.get("native_text", "").strip() else "damaged text"
            _report_page_task(state, f"OCR needed: {reason}")
            return "ocr"
        return "classify"

    def ocr(state: PageState) -> dict[str, Any]:
        _report_page_task(state, f"Running OCR with {state['model']}")
        prompt = (
            "Convert this page to docling."
            if "granite-docling" in state["model"].casefold()
            else "OCR:"
        )
        content = _recognize_page(
            state["image"],
            model=state["model"],
            prompt=prompt,
            url=f"{state['base_url'].rstrip('/')}/api/chat",
            timeout=state["timeout"],
            page_number=state["page_number"],
        )
        blocks, categories, used_docling_labels = _parse_docling_page(
            content,
            base64.b64decode(state["image"]),
            page_number=state["page_number"],
        )
        if not used_docling_labels:
            _report_page_task(state, "DocTags had no semantic labels; using layout rules")
        if not blocks and content.strip():
            blocks = [
                {"id": "b0", "text": content.strip(), "bbox": None, "source_kind": "ocr"}
            ]
            categories = _heuristic_categories(blocks)
        return {"blocks": blocks, "categories": categories}

    def classify(state: PageState) -> dict[str, Any]:
        blocks = state.get("blocks", [])
        verbose = state.get("verbose", False)
        categories = state.get("categories")
        if categories is None:
            _report_page_task(state, f"Applying layout rules to {len(blocks)} blocks")
            categories = _heuristic_categories(blocks)
        else:
            _report_page_task(state, f"Using Docling labels for {len(blocks)} blocks")
        category_counts: dict[str, int] = {}
        for category in categories.values():
            category_counts[category] = category_counts.get(category, 0) + 1
        # Filter out zero counts and format nicely with descriptions
        kept_categories = {k: v for k, v in category_counts.items() if v > 0}
        if kept_categories:
            counts_str = ", ".join(f"{v} {k}" for k, v in sorted(kept_categories.items()))
            _report_page_task(state, f"Content types: {counts_str}")
        
        # In verbose mode, show actual content with classifications
        if verbose and blocks:
            from rich.text import Text
            from ..logging_utils import stderr_console
            console = stderr_console()
            page_num = state.get('page_number', 0)
            console.print(f"[bold]Page {page_num} blocks and classifications:[/bold]")
            for block in blocks:
                block_id = block.get("id", "")
                text = block.get("text", "")
                if len(text) > 100:
                    text = text[:100] + "..."
                category = categories.get(block_id, "unknown")
                # Color code by category for better visibility
                category_colors = {
                    "title": "bold cyan",
                    "text": "white",
                    "table": "magenta",
                    "figure": "green",
                    "caption": "dim",
                    "formula": "yellow",
                    "footnote": "italic",
                    "reference": "red",
                    "page_header": "dim",
                    "page_footer": "dim",
                    "page_number": "dim",
                    "other": "dim",
                    "unknown": "red",
                }
                color = category_colors.get(category, "white")
                # Create styled text with category label and content
                line = Text()
                line.append("  [", style="")
                line.append(f"{category}", style="bold")
                line.append("] ", style="")
                line.append(text, style=color)
                console.print(line)
            console.print()
        
        return {"categories": categories}

    def filter_content(state: PageState) -> dict[str, Any]:
        kept: list[str] = []
        section = state.get("section")
        references_started = state.get("references_started", False)
        categories = state.get("categories", {})
        for block in state.get("blocks", []):
            text = block["text"].strip()
            category = categories.get(block["id"], "text")
            if _is_page_number(text):
                continue
            if category == "reference" or _is_reference_heading(text):
                references_started = True
            if references_started or category in _DROP_CATEGORIES or not text:
                continue
            if category in {"title", "section_header"} or _is_heading(text):
                section = text
                continue
            kept.append(text)
        page_text = "\n\n".join(kept).strip()
        page_number = state["page_number"]
        page_result = (page_number, page_text, section) if page_text else None
        return {
            "result": page_result,
            "section": section,
            "references_started": references_started,
        }

    graph = StateGraph(PageState)
    graph.add_node("inspect", inspect)
    graph.add_node("ocr", ocr)
    graph.add_node("classify", classify)
    graph.add_node("filter", filter_content)
    graph.add_edge(START, "inspect")
    graph.add_conditional_edges(
        "inspect",
        route_after_inspect,
        {"ocr": "ocr", "classify": "classify"},
    )
    graph.add_edge("ocr", "classify")
    graph.add_edge("classify", "filter")
    graph.add_edge("filter", END)
    return graph.compile()


def _report_page_task(state: PageState, task: str) -> None:
    progress = state.get("progress")
    if progress is not None:
        page = state.get("page_number", 0)
        total = state.get("page_count", 0)
        progress(f"Page {page}/{total}: {task}")


def _build_ocr_batch_graph():
    def dispatch(state: OcrBatchState):
        return [
            Send("ocr_page", {"ocr_task": task})
            for task in state.get("ocr_tasks", [])
        ]

    def ocr_page(state: dict[str, Any]) -> dict[str, Any]:
        task = state["ocr_task"]
        progress = task.get("progress")
        page_number = task["page_number"]
        page_count = task["page_count"]
        if progress is not None:
            progress(f"Page {page_number}/{page_count}: Running OCR with {task['model']}")
        prompt = (
            "Convert this page to docling."
            if "granite-docling" in task["model"].casefold()
            else "OCR:"
        )
        content = _recognize_page(
            task["image"],
            model=task["model"],
            prompt=prompt,
            url=f"{task['base_url'].rstrip('/')}/api/chat",
            timeout=task["timeout"],
            page_number=page_number,
        )
        blocks, categories, used_docling_labels = _parse_docling_page(
            content,
            base64.b64decode(task["image"]),
            page_number=page_number,
        )
        if progress is not None and not used_docling_labels:
            progress(f"Page {page_number}/{page_count}: DocTags had no semantic labels; using layout rules")
        if not blocks and content.strip():
            blocks = [
                {"id": "b0", "text": content.strip(), "bbox": None, "source_kind": "ocr"}
            ]
            categories = _heuristic_categories(blocks)
        return {"ocr_results": [{"page_number": page_number, "blocks": blocks, "categories": categories}]}

    graph = StateGraph(OcrBatchState)
    graph.add_node("ocr_page", ocr_page)
    graph.add_conditional_edges(START, dispatch)
    graph.add_edge("ocr_page", END)
    return graph.compile()


def _native_blocks(page) -> list[Block]:
    width, height = page.rect.width, page.rect.height
    blocks: list[Block] = []
    for block in page.get_text("dict").get("blocks", []):
        block_type = block.get("type")
        bbox_raw = block.get("bbox")
        bbox = None
        if bbox_raw and width and height:
            x0, y0, x1, y1 = bbox_raw
            bbox = (
                round(x0 / width * 500, 1),
                round(y0 / height * 500, 1),
                round(x1 / width * 500, 1),
                round(y1 / height * 500, 1),
            )
        if block_type == 0:
            lines = []
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                if line_text.strip():
                    lines.append(line_text.strip())
            text = "\n".join(lines).strip()
            if text:
                blocks.append(
                    {"id": f"b{len(blocks)}", "text": text, "bbox": bbox, "source_kind": "text"}
                )
        elif block_type == 1:
            blocks.append(
                {"id": f"b{len(blocks)}", "text": "", "bbox": bbox, "source_kind": "image"}
            )
    return blocks


def _docling_blocks(content: str) -> list[Block]:
    """Parse Granite Docling location-prefixed segments into candidate blocks."""
    loc_pattern = re.compile(r"<loc_(\d+)>")
    matches = list(loc_pattern.finditer(content))
    blocks: list[Block] = []
    cursor = 0
    block_number = 0
    while cursor < len(matches):
        group = matches[cursor : cursor + 4]
        if len(group) < 4:
            break
        values = [int(match.group(1)) for match in group]
        text_start = group[-1].end()
        next_cursor = cursor + 4
        next_start = matches[next_cursor].start() if next_cursor < len(matches) else len(content)
        text = _clean_docling_markup(content[text_start:next_start])
        if text:
            blocks.append(
                {
                    "id": f"b{block_number}",
                    "text": text,
                    "bbox": tuple(float(value) for value in values),
                    "source_kind": "ocr",
                }
            )
            block_number += 1
        cursor += 4
    if not blocks and content.strip():
        for paragraph in re.split(r"\n\s*\n", content.strip()):
            paragraph = _clean_docling_markup(paragraph)
            if paragraph:
                blocks.append(
                    {
                        "id": f"b{len(blocks)}",
                        "text": paragraph,
                        "bbox": None,
                        "source_kind": "ocr",
                    }
                )
    return blocks


def _clean_docling_markup(text: str) -> str:
    """Remove DocTags control tokens while preserving recognized page text."""
    text = re.sub(r"<loc_\d+>", "", text)
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9_]*(?:\s+[^<>]*)?>", "", text)
    return text.strip()


def _parse_docling_page(
    content: str,
    image_bytes: bytes,
    *,
    page_number: int,
) -> tuple[list[Block], dict[str, str], bool]:
    """Convert semantic DocTags into blocks; fall back when only raw OCR is present."""
    semantic_tags = re.search(
        r"<(?:title|section_header_level_\d+|text|paragraph|table|picture|caption|"
        r"formula|footnote|page_header|page_footer|reference|list_item|code)(?:\s|>)",
        content,
    )
    if semantic_tags is None:
        blocks = _docling_blocks(content)
        return blocks, _heuristic_categories(blocks), False

    try:
        from PIL import Image
        from docling_core.types.doc import ContentLayer, DoclingDocument
        from docling_core.types.doc.document import DocTagsDocument

        with Image.open(BytesIO(image_bytes)) as page_image:
            doctags = DocTagsDocument.from_doctags_and_image_pairs(
                [content], [page_image.convert("RGB")]
            )
        document = DoclingDocument.load_from_doctags(
            doctags, document_name=f"page-{page_number}"
        )
        blocks: list[Block] = []
        categories: dict[str, str] = {}
        for item, _level in document.iterate_items(
            traverse_pictures=True,
            included_content_layers=set(ContentLayer),
        ):
            label = getattr(item, "label", None)
            label = getattr(label, "value", label)
            if label is None:
                continue
            category = _DOCLING_LABELS.get(str(label), "text")
            content_layer = getattr(item, "content_layer", None)
            content_layer = getattr(content_layer, "value", content_layer)
            if content_layer == "furniture" and category not in {
                "page_header",
                "page_footer",
            }:
                category = "furniture"

            text = getattr(item, "text", "") or ""
            if category == "table":
                text = item.export_to_markdown(doc=document)
            block_id = f"b{len(blocks)}"
            blocks.append(
                {
                    "id": block_id,
                    "text": str(text).strip(),
                    "bbox": None,
                    "source_kind": "docling",
                }
            )
            categories[block_id] = category
        if blocks:
            return blocks, categories, True
    except Exception:
        # Keep ingestion usable if a model response contains partial or
        # unsupported DocTags; the plain-text fallback never drops content.
        pass

    blocks = _docling_blocks(content)
    return blocks, _heuristic_categories(blocks), False


def _heuristic_categories(blocks: list[Block]) -> dict[str, str]:
    """Conservative cleanup for native-text pages and unstructured OCR output."""
    categories: dict[str, str] = {}
    for block in blocks:
        text = block["text"].strip()
        bbox = block.get("bbox")
        if block["source_kind"] == "image" and not text:
            category = "figure"
        elif _is_page_number(text):
            category = "page_number"
        elif _is_reference_heading(text):
            category = "reference"
        elif _is_heading(text):
            category = "section_header"
        elif bbox is not None and len(text) <= 140 and bbox[1] <= 18:
            category = "page_header"
        elif bbox is not None and len(text) <= 140 and bbox[3] >= 482:
            category = "page_footer"
        else:
            category = "text"
        categories[block["id"]] = category
    return categories


def discover_pdf_paths(filesystem, root: str | Path, *, recursive: bool = True) -> list[str]:
    """Discover PDF objects through Ikarus's fsspec filesystem interface."""
    root_text = str(root)
    if recursive:
        paths = filesystem.find(root_text)
    else:
        paths = filesystem.ls(root_text, detail=False)
    return sorted(
        (str(path) for path in paths if str(path).casefold().endswith(".pdf")),
        key=str.casefold,
    )


def build_ingestion_graph(filesystem, process_files: Callable[[list[str]], dict[str, Any]]):
    """Create the discovery and processing graph, with Ikarus fsspec at its edge."""

    def discover(state: IngestionState) -> dict[str, Any]:
        return {
            "pdf_paths": discover_pdf_paths(
                filesystem, state["root"], recursive=state.get("recursive", True)
            )
        }

    def process(state: IngestionState) -> dict[str, Any]:
        return {"summary": process_files(state.get("pdf_paths", []))}

    graph = StateGraph(IngestionState)
    graph.add_node("discover_pdfs", discover)
    graph.add_node("process_pdfs", process)
    graph.add_edge(START, "discover_pdfs")
    graph.add_edge("discover_pdfs", "process_pdfs")
    graph.add_edge("process_pdfs", END)
    return graph.compile()
