"""Stream page-by-page PDF extraction through Bibliograph's Ollama OCR adapter.

Despite the historical filename, this exercises the same Ikarus-compatible
page extractor used by Bibliograph ingestion. By default every page is sent to
the local PaddleOCR-VL model; use ``--mode hybrid`` to keep clean embedded text
and OCR only pages where native extraction looks incomplete.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

DEFAULT_PDF = (
    Path(__file__).resolve().parents[1]
    / "pdfs"
    / "spatial_spillover_of_conflict.pdf"
)
DEFAULT_MODEL = "ibm/granite-docling:258m"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to extract (default: {DEFAULT_PDF})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama vision model")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "hybrid"),
        default="all",
        help="OCR every page, or OCR only pages without usable embedded text",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=120,
        help="Minimum embedded text length for hybrid mode (default: 120)",
    )
    args = parser.parse_args()

    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        parser.error(f"PDF not found: {pdf_path}")

    try:
        from bibliograph.adapters.ollama_ocr import make_ollama_page_extractor
    except ImportError as error:
        parser.error(
            "Bibliograph's PDF extraction dependencies are unavailable. "
            "Run this script in the Bibliograph environment with the 'all' extras installed."
        )
        raise error  # pragma: no cover - parser.error exits

    console = Console(file=sys.stdout)
    printed_page = False

    def render_page(page_number: int, text: str) -> None:
        nonlocal printed_page
        if printed_page:
            console.print()
        text_without_locations = re.sub(r"<loc_\d+>", "", text)
        console.print(Markdown(html.escape(text_without_locations)))
        printed_page = True

    extractor = make_ollama_page_extractor(
        model=args.model,
        prompt="Convert this page to docling.",
        base_url=args.ollama_url,
        mode=args.mode,
        min_text_chars=args.min_text_chars,
        on_page=render_page,
    )

    print(
        f"Extracting {pdf_path} page by page (Ollama: {args.model}, mode: {args.mode})...",
        file=sys.stderr,
        flush=True,
    )
    try:
        extractor(pdf_path)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1

    print("Extraction complete.", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
