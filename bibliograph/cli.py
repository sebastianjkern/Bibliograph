import argparse
import os
from pathlib import Path

from .drafts import parse_draft_file
from .embeddings import HashEmbedder, OpenAICompatibleEmbedder, SentenceTransformerEmbedder
from .export import to_markdown
from .ingest import index_pdf
from .models import Paper
from .reranker import HeuristicReranker, OpenAIReranker, rerank_matches
from .retrieval import find_citations, find_claim_citations
from .store import SQLiteIndex
from .suggestions import OpenAISuggestionGenerator, suggest_citations
from .zotero_sync import MissingPaper, sync_collection


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
    parser.add_argument(
        "--embedding-cache-dir", help="Persistent cache directory for SentenceTransformers models"
    )
    parser.add_argument(
        "--embedding-offline",
        action="store_true",
        default=None,
        help="Load SentenceTransformers only from the local cache",
    )
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
    download.add_argument(
        "item_key", nargs="?", help="Optional Zotero parent item or attachment key"
    )
    identifier = download.add_mutually_exclusive_group()
    identifier.add_argument("--doi", help="Resolve the Zotero item by DOI")
    identifier.add_argument("--title", help="Resolve the Zotero item by exact title")
    download.add_argument("--attachment-key", help="Select one PDF when the item has several")
    download.add_argument("--output-dir", type=Path, default=Path("pdfs"))

    suggest = subparsers.add_parser("suggest", help="Suggest citations for a draft file")
    suggest.add_argument("draft", type=Path)
    suggest.add_argument("--limit", type=int, default=5)
    suggest.add_argument("--min-score", type=float, default=0.0)
    suggest.add_argument("--output", type=Path)
    suggest.add_argument("--llm-model", help="Optional chat model for grounded rationales")

    check = subparsers.add_parser(
        "check", help="Sync one Zotero collection and check a Typst/LaTeX draft"
    )
    check.add_argument("draft", type=Path)
    check.add_argument("collection", help="Zotero collection key or exact name")
    check.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    check.add_argument("--limit", type=int, default=5)
    check.add_argument("--min-score", type=float, default=0.0)
    check.add_argument("--output", type=Path)
    check.add_argument("--llm-model", help="Optional chat model for evidence reranking")
    check.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing PDFs through the single-item Zotero backend",
    )
    return parser


def _embedder(
    provider: str,
    model: str | None,
    base_url: str | None,
    cache_dir: str | None,
    offline: bool | None,
):
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder.from_environment(model, cache_dir, offline)
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
            args.doi,
            args.title,
        )
        status = "Downloaded" if result.downloaded else "Already present"
        print(f"{status}: {result.path}")
        return 0

    embedder = _embedder(
        args.embedding_provider,
        args.model,
        args.embedding_base_url,
        args.embedding_cache_dir,
        args.embedding_offline,
    )
    index = SQLiteIndex(args.db)
    try:
        if args.command == "index-pdf":
            paper = Paper(args.key, args.title, tuple(args.author), args.year, args.doi)
            count = index_pdf(index, embedder, paper, args.pdf)
            print(f"Indexed {count} chunks from {args.pdf}")
            return 0

        if args.command == "check":
            from connections.reload_embeddings import find_collection_key, load_zotero_client

            zotero = load_zotero_client()
            collection_key = find_collection_key(zotero, args.collection)
            report = sync_collection(
                zotero, collection_key, args.pdf_dir, index, embedder
            )
            downloaded: list[str] = []
            if args.download_missing:
                from connections.reload_embeddings import download_pdf_for_item

                for missing in report.missing_papers:
                    try:
                        result = download_pdf_for_item(
                            missing.item_key,
                            str(args.pdf_dir),
                            missing.attachment_keys[0] if missing.attachment_keys else None,
                        )
                    except ValueError:
                        continue
                    downloaded.append(result.path)
                if downloaded:
                    report = sync_collection(
                        zotero, collection_key, args.pdf_dir, index, embedder
                    )
            matches = find_claim_citations(
                parse_draft_file(args.draft),
                index,
                embedder,
                limit=args.limit,
                min_score=args.min_score,
            )
            if args.llm_model:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is required when --llm-model is used")
                matches = rerank_matches(
                    matches,
                    OpenAIReranker(args.llm_model, api_key, os.getenv("OPENAI_BASE_URL")),
                )
            else:
                matches = rerank_matches(matches, HeuristicReranker())
            output = to_markdown(suggest_citations(matches))
            summary = (
                f"Indexed: {len(report.indexed)} | Unchanged: {len(report.unchanged)} | "
                f"Missing papers: {len(report.missing_papers)} | Downloaded: {len(downloaded)}\n"
            )
            if report.missing_papers:
                summary += "Missing papers:\n" + "\n".join(
                    _format_missing_paper(paper) for paper in report.missing_papers
                ) + "\n"
            output = summary + "\n" + output
            if args.output:
                args.output.write_text(output, encoding="utf-8")
            else:
                print(output)
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


def _format_missing_paper(paper: MissingPaper) -> str:
    doi = f" — DOI: {paper.doi}" if paper.doi else ""
    keys = f" — attachments: {', '.join(paper.attachment_keys)}" if paper.attachment_keys else ""
    return f"- {paper.title}{doi}{keys}"
