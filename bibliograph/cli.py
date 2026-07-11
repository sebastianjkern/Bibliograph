import argparse
import os
from pathlib import Path

from .embeddings import HashEmbedder, OpenAICompatibleEmbedder, SentenceTransformerEmbedder
from .export import to_markdown
from .ingest import index_pdf
from .models import Paper
from .retrieval import find_citations
from .store import SQLiteIndex
from .suggestions import OpenAISuggestionGenerator, suggest_citations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find citation evidence in a local paper library.")
    parser.add_argument("--db", default="bibliograph.db", help="SQLite index path")
    parser.add_argument(
        "--embedding-provider",
        choices=("hash", "sentence-transformers", "openai"),
        default="hash",
        help="Embedding backend (default: hash)",
    )
    parser.add_argument(
        "--model", help="Embedding model name for sentence-transformers or OpenAI-compatible APIs"
    )
    parser.add_argument("--embedding-base-url", help="Base URL for an OpenAI-compatible API")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index-pdf", help="Index one PDF with its citation metadata")
    index.add_argument("pdf", type=Path)
    index.add_argument("--key", required=True, help="Zotero item key")
    index.add_argument("--title", required=True)
    index.add_argument("--author", action="append", default=[])
    index.add_argument("--year")
    index.add_argument("--doi")

    download = subparsers.add_parser(
        "download-pdf", help="Download one missing PDF attachment from Zotero"
    )
    download.add_argument("item_key", help="Zotero parent item or attachment key")
    download.add_argument("--attachment-key", help="Select one PDF when the item has several")
    download.add_argument("--output-dir", type=Path, default=Path("pdfs"))

    suggest = subparsers.add_parser("suggest", help="Suggest citations for a draft file")
    suggest.add_argument("draft", type=Path)
    suggest.add_argument("--limit", type=int, default=5)
    suggest.add_argument("--min-score", type=float, default=0.0)
    suggest.add_argument("--output", type=Path)
    suggest.add_argument("--llm-model", help="Optional chat model for grounded rationales")
    return parser


def _embedder(provider: str, model: str | None, base_url: str | None):
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
    if provider == "openai":
        return OpenAICompatibleEmbedder.from_environment(model, base_url)
    return HashEmbedder()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download-pdf":
        from connections.reload_embeddings import download_pdf_for_item

        result = download_pdf_for_item(
            args.item_key,
            str(args.output_dir),
            args.attachment_key,
        )
        status = "Downloaded" if result.downloaded else "Already present"
        print(f"{status}: {result.path}")
        return 0

    embedder = _embedder(args.embedding_provider, args.model, args.embedding_base_url)
    index = SQLiteIndex(args.db)
    try:
        if args.command == "index-pdf":
            paper = Paper(args.key, args.title, tuple(args.author), args.year, args.doi)
            count = index_pdf(index, embedder, paper, args.pdf)
            print(f"Indexed {count} chunks from {args.pdf}")
            return 0

        matches = find_citations(
            args.draft.read_text(encoding="utf-8"),
            index,
            embedder,
            limit=args.limit,
            min_score=args.min_score,
        )
        generator = None
        if args.llm_model:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required when --llm-model is used")
            generator = OpenAISuggestionGenerator(
                args.llm_model,
                api_key,
                os.getenv("OPENAI_BASE_URL"),
            )
        output = to_markdown(suggest_citations(matches, generator, args.min_score))
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0
    finally:
        index.close()
