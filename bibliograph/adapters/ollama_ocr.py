"""Page-level PDF OCR through a local Ollama vision model."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .pdf import _is_heading, _is_page_number, _is_reference_heading, extract_pages


def configured_page_extractor(settings: Mapping[str, object], verbose: bool = False):
    """Choose the configured PDF page extractor for the Ikarus ingestion seam."""
    options = settings.get("pdf_extraction", {})
    if not isinstance(options, Mapping) or options.get("provider", "ollama") == "pymupdf":
        return extract_pages
    if options.get("provider") != "ollama":
        raise ValueError(f"Unsupported PDF extraction provider: {options.get('provider')}")
    from ..pipeline.ingestion_workflow import make_hybrid_page_extractor

    return make_hybrid_page_extractor(
        model=str(options["ollama_model"]),
        base_url=str(options["ollama_base_url"]),
        mode=str(options["ollama_mode"]),
        min_text_chars=int(options["min_text_chars"]),
        verbose=verbose,
    )


def make_ollama_page_extractor(
    *,
    model: str,
    prompt: str = "OCR:",
    base_url: str = "http://localhost:11434",
    mode: str = "hybrid",
    min_text_chars: int = 120,
    timeout: float = 300,
    on_page: Callable[[int, str], None] | None = None,
) -> Callable[[str | Path], list[tuple[int, str, str | None]]]:
    """Return an Ikarus-compatible extractor using native text and Ollama OCR.

    In ``hybrid`` mode, pages with a usable embedded text layer stay on the
    PyMuPDF path; pages with little or clearly damaged text are rendered and
    sent individually to Ollama. ``all`` sends every page to the model.
    """
    if mode not in {"hybrid", "all"}:
        raise ValueError("PDF OCR mode must be 'hybrid' or 'all'")
    if not model.strip():
        raise ValueError("Ollama OCR model must be a non-empty model name")
    if min_text_chars < 1:
        raise ValueError("min_text_chars must be positive")

    def extract(pdf_path: str | Path) -> list[tuple[int, str, str | None]]:
        try:
            import fitz
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Ollama PDF OCR requires the 'pdf' optional dependency") from error

        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        native_pages = (
            {page: (text, section) for page, text, section in extract_pages(path)}
            if mode == "hybrid"
            else {}
        )
        base_url_clean = base_url.rstrip("/")
        results: list[tuple[int, str, str | None]] = []
        current_section: str | None = None
        references_started = False

        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                native_text = page.get_text("text").strip()
                if mode == "hybrid" and _has_usable_text(native_text, min_text_chars):
                    native = native_pages.get(page_number)
                    if native is not None:
                        text, section = native
                        current_section = section or current_section
                        results.append((page_number, text, current_section))
                        if on_page is not None:
                            on_page(page_number, text)
                    if any(_is_reference_heading(line) for line in native_text.splitlines()):
                        references_started = True
                    continue

                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
                content = _recognize_page(
                    image,
                    model=model,
                    prompt=prompt,
                    url=f"{base_url_clean}/api/chat",
                    timeout=timeout,
                    page_number=page_number,
                )
                cleaned_lines: list[str] = []
                for line in content.splitlines():
                    line = line.strip()
                    if not line or _is_page_number(line):
                        continue
                    if _is_heading(line):
                        if _is_reference_heading(line):
                            references_started = True
                        elif not references_started:
                            current_section = line
                        continue
                    if not references_started:
                        cleaned_lines.append(line)
                text = "\n".join(cleaned_lines).strip()
                if text:
                    results.append((page_number, text, current_section))
                    if on_page is not None:
                        on_page(page_number, text)

        return results

    return extract


def _has_usable_text(text: str, minimum: int) -> bool:
    if len(text) < minimum:
        return False
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return False
    damaged = sum(char == "\ufffd" for char in visible)
    return damaged / len(visible) <= 0.01


def _recognize_page(
    image: str,
    *,
    model: str,
    prompt: str,
    url: str,
    timeout: float,
    page_number: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [image]},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except HTTPError as error:
        try:
            response_body = error.read().decode("utf-8", errors="replace").strip()
        except OSError:
            response_body = ""
        detail = f"HTTP {error.code} {error.reason}"
        if response_body:
            detail += f": {response_body}"
        raise RuntimeError(
            f"Ollama OCR failed on PDF page {page_number} using {model} "
            f"at {url}: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Ollama OCR failed on PDF page {page_number} using {model} "
            f"at {url}: {error}"
        ) from error
    try:
        return str(body["message"]["content"])
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"Ollama returned no OCR text for PDF page {page_number} using {model}"
        ) from error
