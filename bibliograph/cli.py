"""Small CLI parser and dispatcher for Bibliograph workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from .bootstrap import run_check, run_search, run_status, run_sync
from .logging_utils import configure_logging, get_logger
from .providers.registry import chat_names, embedding_names
from .render import render_sync
from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find and verify citation evidence in a local Zotero-backed library."
    )
    parser.add_argument("--config", type=Path, help="Path to bibliograph.toml")
    parser.add_argument(
        "--profile",
        default="default",
        help="Configuration profile (default: default)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable console progress logging")
    _add_runtime_overrides(parser)
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser(
        "sync",
        help="Synchronize a Zotero collection into the index",
    )
    sync.add_argument("collection", nargs="?", help="Collection key or exact name")
    sync.add_argument(
        "--rebuild",
        action="store_true",
        help="Build a replacement index and swap it in",
    )
    _add_legacy_rebuild_argument(sync)
    sync.add_argument("--output", type=Path, help="Write the synchronization summary to a file")

    search = subparsers.add_parser("search", help="Search indexed sources for one claim")
    search.add_argument("claim")
    _add_query_options(search)

    check = subparsers.add_parser(
        "check",
        help="Check a LaTeX or Typst draft using the current index",
    )
    check.add_argument("draft", type=Path)
    # A second positional preserves the former `check DRAFT COLLECTION` command
    # for one release.  The handler explicitly runs sync before the read-only check.
    check.add_argument("legacy_collection", nargs="?", help=argparse.SUPPRESS)
    _add_legacy_rebuild_argument(check)
    _add_query_options(check)

    status = subparsers.add_parser(
        "status",
        help="Show index status and optional provider diagnostics",
    )
    status.add_argument(
        "--probe",
        action="store_true",
        help="Test configured provider capabilities",
    )
    status.add_argument("--output", type=Path, help="Write the status report to a file")

    # One-release compatibility shims. They intentionally live only at the
    # command boundary and delegate to the new read-only handlers.
    legacy_search = subparsers.add_parser(
        "find-sources",
        aliases=("find",),
        help="Deprecated; use search",
    )
    legacy_search.add_argument("claim")
    _add_query_options(legacy_search)
    legacy_search.set_defaults(command="legacy-search")

    legacy_suggest = subparsers.add_parser("suggest", help="Deprecated; use check")
    legacy_suggest.add_argument("draft", type=Path)
    _add_query_options(legacy_suggest)
    legacy_suggest.set_defaults(command="legacy-suggest")
    return parser


def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", help="Override the configured index path")
    parser.add_argument("--pdf-dir", help="Override the configured PDF cache directory")
    parser.add_argument(
        "--collection",
        dest="configured_collection",
        help="Override the configured collection",
    )
    parser.add_argument("--zotero-storage-dir", help="Override the local Zotero storage directory")
    parser.add_argument("--embedding-provider", choices=embedding_names())
    parser.add_argument("--embedding-model", "--model", dest="embedding_model")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--embedding-cache-dir")
    parser.add_argument("--embedding-offline", action="store_true", default=None)
    parser.add_argument("--llm-provider", choices=chat_names())
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-mode", choices=("off", "optional", "required"))


def _add_legacy_rebuild_argument(parser: argparse.ArgumentParser) -> None:
    """Accept the old flag after the command without broadening new command syntax."""

    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        dest="subcommand_rebuild_db",
        help=argparse.SUPPRESS,
    )


def _add_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic ranking and rationale",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable only the LLM reranking stage",
    )
    parser.add_argument(
        "--no-evidence-extraction",
        action="store_true",
        help="Disable only the LLM evidence-selection stage",
    )
    parser.add_argument(
        "--no-enrichment",
        action="store_true",
        help="Disable LLM evidence selection and rationale generation",
    )
    parser.add_argument("--output", type=Path)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.quiet)
    logger = get_logger("cli")
    try:
        settings = load_settings(args.config, args.profile, _settings_overrides(args))
        logger.info("Command: %s", args.command)
        legacy_rebuild = args.rebuild_db or getattr(args, "subcommand_rebuild_db", False)
        if legacy_rebuild:
            logger.warning("`--rebuild-db` is deprecated; use `sync --rebuild` instead")
        if args.command == "sync":
            result = run_sync(
                settings,
                collection=args.collection,
                rebuild=args.rebuild or legacy_rebuild,
                show_progress=not args.quiet,
            )
            _emit(render_sync(result), args.output)
            return 0

        if args.command in {"search", "legacy-search"}:
            if args.command == "legacy-search":
                logger.warning("`find-sources`/`find` is deprecated; use `search` instead")
            if legacy_rebuild:
                raise ValueError("`--rebuild-db` is only supported by `sync --rebuild`")
            result = run_search(
                settings,
                args.claim,
                limit=args.limit,
                min_score=args.min_score,
                no_llm=args.no_llm,
                disabled_stages=_disabled_llm_stages(args),
                show_progress=not args.quiet,
                enrich=not args.no_enrichment,
            )
            _emit(result["markdown"], args.output)
            return 0

        if args.command in {"check", "legacy-suggest"}:
            if args.command == "legacy-suggest":
                logger.warning("`suggest` is deprecated; use `check` instead")
            if args.command == "check" and args.legacy_collection:
                logger.warning(
                    "`check DRAFT COLLECTION` is deprecated; run `sync COLLECTION` "
                    "then `check DRAFT`"
                )
                sync_result = run_sync(
                    settings,
                    collection=args.legacy_collection,
                    rebuild=legacy_rebuild,
                    show_progress=not args.quiet,
                )
                logger.info("Legacy sync indexed %d documents", len(sync_result["indexed"]))
            elif legacy_rebuild:
                raise ValueError("`--rebuild-db` is only supported by `sync --rebuild`")
            result = run_check(
                settings,
                args.draft,
                limit=args.limit,
                min_score=args.min_score,
                no_llm=args.no_llm,
                disabled_stages=_disabled_llm_stages(args),
                show_progress=not args.quiet,
                enrich=not args.no_enrichment,
            )
            _emit(result["markdown"], args.output)
            return 0

        if args.command == "status":
            if legacy_rebuild:
                raise ValueError("`--rebuild-db` is only supported by `sync --rebuild`")
            result = run_status(settings, probe=args.probe)
            _emit(result["markdown"], args.output)
            return 0
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as error:
        logger.error("%s", error)
        return 1


def _settings_overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "db": args.db,
        "pdf_dir": args.pdf_dir,
        "collection": args.configured_collection,
        "zotero_storage_dir": args.zotero_storage_dir,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_cache_dir": args.embedding_cache_dir,
        "embedding_offline": args.embedding_offline,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "llm_mode": args.llm_mode,
    }
    return {key: value for key, value in values.items() if value is not None}


def _disabled_llm_stages(args: argparse.Namespace) -> tuple[str, ...]:
    stages: list[str] = []
    if args.no_rerank:
        stages.append("rerank")
    if args.no_evidence_extraction:
        stages.append("evidence")
    if args.no_enrichment:
        stages.extend(("evidence", "rationale"))
    return tuple(dict.fromkeys(stages))


def _emit(markdown: str, output: Path | None) -> None:
    if output is not None:
        output.write_text(markdown, encoding="utf-8")
        get_logger("cli").info("Wrote report: %s", output)
    else:
        Console().print(Markdown(markdown))
