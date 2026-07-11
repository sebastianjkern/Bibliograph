import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .config import DEFAULT_LLM_MODEL, openai_settings
from .drafts import parse_draft_file
from .embeddings import HashEmbedder, OpenAICompatibleEmbedder, SentenceTransformerEmbedder
from .export import sources_to_markdown, to_markdown
from .ingest import index_pdf
from .logging_utils import configure_logging, get_logger
from .models import Paper
from .reranker import HeuristicReranker, OpenAIReranker, rerank_matches
from .retrieval import find_citations, find_claim_citations, find_claim_sources
from .store import SQLiteIndex
from .suggestions import OpenAISuggestionGenerator, suggest_citations
from .zotero_sync import MissingPaper, is_remote_downloadable_item, sync_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find citation evidence in a local paper library.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level (default: INFO)",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable console progress logging")
    parser.add_argument("--db", default="bibliograph.db", help="SQLite index path")
    parser.add_argument(
        "--zotero-storage-dir",
        default=None,
        help="Local Zotero storage directory (usually .../storage)",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("hash", "sentence-transformers", "openai"),
        default="sentence-transformers",
        help="Embedding backend (default: sentence-transformers)",
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
        "--source", choices=("zotero", "remote"), default="remote",
        help="PDF source (default: remote open-access resolver chain)",
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
    suggest.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    suggest.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic rationales instead of the local LLM",
    )

    find_sources = subparsers.add_parser(
        "find-sources",
        aliases=("find",),
        help="Find sources for one claim in the local vector index",
    )
    find_sources.add_argument("claim", help="Claim or question to search for")
    find_sources.add_argument("--limit", type=int, default=5)
    find_sources.add_argument("--min-score", type=float, default=0.0)
    find_sources.add_argument("--output", type=Path)
    find_sources.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    find_sources.add_argument("--no-rerank", action="store_true")

    check = subparsers.add_parser(
        "check", help="Sync one Zotero collection and check a Typst/LaTeX draft"
    )
    check.add_argument("draft", type=Path)
    check.add_argument("collection", help="Zotero collection key or exact name")
    check.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    check.add_argument("--limit", type=int, default=5)
    check.add_argument("--min-score", type=float, default=0.0)
    check.add_argument("--output", type=Path)
    check.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    check.add_argument("--no-rerank", action="store_true", help="Skip LLM reranking")
    download_group = check.add_mutually_exclusive_group()
    download_group.add_argument(
        "--download-missing",
        action="store_true",
        dest="download_missing",
        help="Download missing PDFs (enabled by default)",
    )
    download_group.add_argument(
        "--no-download-missing",
        action="store_false",
        dest="download_missing",
        help="Only report missing PDFs",
    )
    check.set_defaults(download_missing=True)
    check.add_argument(
        "--download-source",
        choices=("zotero", "remote"),
        default="remote",
        help="Source used with --download-missing",
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
    load_dotenv()
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.quiet)
    logger = get_logger("cli")
    logger.info("Command: %s", args.command)
    if args.command == "download-pdf":
        logger.info("Downloading one PDF from %s", args.source)
        if args.source == "remote":
            if not args.doi:
                raise ValueError("Remote downloads require --doi")
            from .remote import RemoteDownloadError, download_remote_pdf

            try:
                result = download_remote_pdf(
                    args.doi,
                    args.output_dir,
                    email=os.getenv("UNPAYWALL_EMAIL"),
                    openalex_api_key=os.getenv("OPENALEX_API_KEY"),
                    title=args.title,
                    item_key=args.item_key,
                )
            except RemoteDownloadError as error:
                logger.error("Automatic download failed: %s", error)
                for hint in error.hints:
                    logger.error("Manual download hint (%s): %s", hint.source, hint.url)
                return 1
        else:
            from connections.reload_embeddings import download_pdf_for_item

            result = download_pdf_for_item(
                args.item_key,
                str(args.output_dir),
                args.attachment_key,
                args.doi,
                args.title,
                args.zotero_storage_dir,
            )
        status = "Downloaded" if result.downloaded else "Already present"
        logger.info("%s PDF: %s", status, result.path)
        print(f"{status}: {result.path}")
        return 0

    logger.info("Loading %s embedding backend", args.embedding_provider)
    embedder = _embedder(
        args.embedding_provider,
        args.model,
        args.embedding_base_url,
        args.embedding_cache_dir,
        args.embedding_offline,
    )
    index = SQLiteIndex(args.db, embedding_id=getattr(embedder, "identity", None))
    logger.info("Using persistent vector index: %s", args.db)
    if index.reindexed:
        logger.info("Index cleared; local PDFs will be reindexed with the selected model")
    try:
        if args.command == "index-pdf":
            paper = Paper(args.key, args.title, tuple(args.author), args.year, args.doi)
            logger.info("Extracting and indexing PDF: %s", args.pdf)
            count = index_pdf(index, embedder, paper, args.pdf)
            logger.info("Indexed %d chunks from %s", count, args.pdf)
            print(f"Indexed {count} chunks from {args.pdf}")
            return 0

        if args.command in {"find-sources", "find"}:
            logger.info("Searching local sources for claim")
            matches = find_claim_sources(
                args.claim,
                index,
                embedder,
                limit=args.limit,
                min_score=args.min_score,
            )
            if not args.no_rerank:
                logger.info("Reranking local sources with LLM model: %s", args.llm_model)
                try:
                    api_key, base_url = openai_settings()
                    matches = rerank_matches(
                        matches,
                        OpenAIReranker(args.llm_model, api_key, base_url),
                    )
                except Exception as error:
                    logger.warning("LLM reranking unavailable; using retrieval order: %s", error)
                    matches = rerank_matches(matches, HeuristicReranker())
            output = sources_to_markdown(matches)
            if args.output:
                args.output.write_text(output, encoding="utf-8")
                logger.info("Wrote report: %s", args.output)
            else:
                print(output)
            return 0

        if args.command == "check":
            from connections.reload_embeddings import find_collection_key, load_zotero_client

            zotero = load_zotero_client()
            logger.info("Resolving Zotero collection: %s", args.collection)
            collection_key = find_collection_key(zotero, args.collection)
            logger.info("Synchronizing collection %s", collection_key)
            report = sync_collection(
                zotero,
                collection_key,
                args.pdf_dir,
                index,
                embedder,
                zotero_storage_dir=args.zotero_storage_dir,
            )
            logger.info(
                "Sync complete: %d indexed, %d unchanged, %d missing",
                len(report.indexed),
                len(report.unchanged),
                len(report.missing_papers),
            )
            if report.imported:
                logger.info("Imported %d PDFs from local Zotero storage", len(report.imported))
            downloaded: list[str] = []
            if args.download_missing:
                logger.info(
                    "Downloading missing PDFs through the %s backend",
                    args.download_source,
                )
                from connections.reload_embeddings import download_pdf_for_item

                from .remote import download_remote_pdf

                for missing in report.missing_papers:
                    if args.download_source == "remote" and not is_remote_downloadable_item(
                        missing.item_type
                    ):
                        logger.info(
                            "Skipping remote PDF download for Zotero item type %s: %s",
                            missing.item_type,
                            missing.title,
                        )
                        continue
                    try:
                        if args.download_source == "remote":
                            if not missing.doi:
                                continue
                            result = download_remote_pdf(
                                missing.doi,
                                args.pdf_dir,
                                email=os.getenv("UNPAYWALL_EMAIL"),
                                openalex_api_key=os.getenv("OPENALEX_API_KEY"),
                                title=missing.title,
                                item_key=missing.item_key,
                            )
                        else:
                            result = download_pdf_for_item(
                                missing.item_key,
                                str(args.pdf_dir),
                                missing.attachment_keys[0]
                                if missing.attachment_keys
                                else None,
                                zotero_storage_dir=args.zotero_storage_dir,
                            )
                    except (FileNotFoundError, ValueError) as error:
                        logger.warning("Could not download missing paper: %s", missing.title)
                        for hint in getattr(error, "hints", ()):
                            logger.warning("Manual download hint (%s): %s", hint.source, hint.url)
                        continue
                    downloaded.append(result.path)
                    logger.info("Downloaded: %s", result.path)
                if downloaded:
                    logger.info("Re-synchronizing downloaded PDFs")
                    report = sync_collection(
                        zotero,
                        collection_key,
                        args.pdf_dir,
                        index,
                        embedder,
                        zotero_storage_dir=args.zotero_storage_dir,
                    )
            logger.info("Parsing draft and retrieving claim evidence: %s", args.draft)
            matches = find_claim_citations(
                parse_draft_file(args.draft),
                index,
                embedder,
                limit=args.limit,
                min_score=args.min_score,
            )
            if not args.no_rerank:
                logger.info("Reranking evidence with LLM model: %s", args.llm_model)
                try:
                    api_key, base_url = openai_settings()
                    matches = rerank_matches(
                        matches,
                        OpenAIReranker(
                            args.llm_model,
                            api_key,
                            base_url,
                        ),
                    )
                except Exception as error:
                    logger.warning("LLM reranking unavailable; using retrieval order: %s", error)
                    matches = rerank_matches(matches, HeuristicReranker())
            else:
                matches = rerank_matches(matches, HeuristicReranker())
            output = to_markdown(suggest_citations(matches))
            summary = (
                f"Indexed: {len(report.indexed)} | Unchanged: {len(report.unchanged)} | "
                f"Imported: {len(report.imported)} | "
                f"Missing papers: {len(report.missing_papers)} | "
                f"Downloaded: {len(downloaded)}\n"
            )
            if report.missing_papers:
                summary += "Missing papers:\n" + "\n".join(
                    _format_missing_paper(paper) for paper in report.missing_papers
                ) + "\n"
            output = summary + "\n" + output
            if args.output:
                args.output.write_text(output, encoding="utf-8")
                logger.info("Wrote report: %s", args.output)
            else:
                logger.info("Writing report to console")
                print(output)
            return 0

        logger.info("Retrieving citations from draft: %s", args.draft)
        matches = find_citations(
            args.draft.read_text(encoding="utf-8"),
            index,
            embedder,
            limit=args.limit,
            min_score=args.min_score,
        )
        generator = None
        if not args.no_llm:
            try:
                api_key, base_url = openai_settings()
                generator = OpenAISuggestionGenerator(
                    args.llm_model,
                    api_key,
                    base_url,
                )
            except Exception as error:
                logger.warning(
                    "LLM suggestions unavailable; using deterministic rationales: %s", error
                )
        output = to_markdown(suggest_citations(matches, generator, args.min_score))
        if args.output:
            args.output.write_text(output, encoding="utf-8")
            logger.info("Wrote report: %s", args.output)
        else:
            logger.info("Writing report to console")
            print(output)
        return 0
    finally:
        index.close()


def _format_missing_paper(paper: MissingPaper) -> str:
    doi = f" — DOI: {paper.doi}" if paper.doi else ""
    keys = f" — attachments: {', '.join(paper.attachment_keys)}" if paper.attachment_keys else ""
    return f"- {paper.title}{doi}{keys}"
