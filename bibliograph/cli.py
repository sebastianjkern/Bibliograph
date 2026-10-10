"""Small CLI parser and dispatcher for Bibliograph workflows."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from adapters.providers.model_runtimes.registry import chat_names, embedding_names
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .bootstrap import run_check, run_ingest_pdfs, run_search, run_status, run_sync
from .logging_utils import configure_logging, get_logger
from .pipeline.retrieval import DEFAULT_SEARCH_LIMIT
from .render import render_sync
from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find and verify citation evidence in a local Zotero-backed library.",
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="show_help",
        help="Show this help message and the active configuration summary",
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

    subparsers = parser.add_subparsers(dest="command")

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
    sync.add_argument(
        "--reset-pdf-index",
        "--reset",
        dest="reset_pdf_index",
        action="store_true",
        help="Force reindex PDFs in the selected Zotero collection",
    )
    sync.add_argument("--output", type=Path, help="Write the synchronization summary to a file")

    ingest = subparsers.add_parser(
        "ingest-pdfs",
        aliases=("ingest",),
        help="Index PDFs from a local folder without Zotero",
    )
    ingest.add_argument("directory", type=Path, help="Folder containing PDF files")
    ingest.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only inspect PDFs directly inside the folder",
    )
    ingest.add_argument(
        "--reset-pdf-index",
        "--reset",
        dest="reset_pdf_index",
        action="store_true",
        help="Force reindex PDFs found in this directory",
    )
    ingest.add_argument("--output", type=Path, help="Write the ingestion summary to a file")

    search = subparsers.add_parser("search", help="Search indexed sources for one claim")
    search.add_argument("claim")
    _add_query_options(search, one_per_paper=True)
    search.add_argument("--verbose", action="store_true", help="Show all retrieved passages")

    check = subparsers.add_parser(
        "check",
        help="Check a LaTeX or Typst draft using the current index",
    )
    check.add_argument("draft", type=Path)
    check.add_argument("--verbose", action="store_true", help="Show retrieval diagnostics")
    # A second positional preserves the former `check DRAFT COLLECTION` command
    # for one release.  The handler explicitly runs sync before the read-only check.
    check.add_argument("legacy_collection", nargs="?", help=argparse.SUPPRESS)
    _add_legacy_rebuild_argument(check)
    _add_query_options(check)
    check.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel claim retrieval processes (default: 2; use 1 for sequential)",
    )

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
    _add_query_options(legacy_search, one_per_paper=True)
    legacy_search.add_argument("--verbose", action="store_true", help=argparse.SUPPRESS)
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


def _add_query_options(
    parser: argparse.ArgumentParser, *, one_per_paper: bool = False
) -> None:
    parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)
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
    if one_per_paper:
        parser.add_argument(
            "--one-per-paper",
            action="store_true",
            help="Keep at most one excerpt from each paper",
        )
    parser.add_argument("--output", type=Path)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.show_help:
        parser.print_help()
        _print_configuration_summary(args.config, args.profile)
        return 0
    if args.command is None:
        parser.error("a command is required (use --help for usage and configuration)")
    configure_logging(args.log_level, args.quiet)
    logger = get_logger("cli")
    try:
        settings = load_settings(args.config, args.profile, _settings_overrides(args))
        logger.info("Command: %s", args.command)
        _announce_command(args, settings)
        legacy_rebuild = args.rebuild_db or getattr(args, "subcommand_rebuild_db", False)
        if legacy_rebuild:
            logger.warning("`--rebuild-db` is deprecated; use `sync --rebuild` instead")
        if args.command == "sync":
            result = run_sync(
                settings,
                collection=args.collection,
                rebuild=args.rebuild or legacy_rebuild,
                show_progress=not args.quiet,
                force_reindex=args.reset_pdf_index,
            )
            _emit(render_sync(result), args.output)
            return 0

        if args.command in {"ingest-pdfs", "ingest"}:
            if legacy_rebuild:
                raise ValueError("`--rebuild-db` is only supported by `sync --rebuild`")
            result = run_ingest_pdfs(
                settings,
                args.directory,
                recursive=not args.non_recursive,
                show_progress=not args.quiet,
                force_reindex=args.reset_pdf_index,
            )
            _emit(_render_ingest_summary(result), args.output)
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
                one_per_paper=args.one_per_paper,
                verbose=args.verbose,
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
                workers=getattr(args, "workers", 1),
                verbose=getattr(args, "verbose", False),
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


def _announce_command(args: argparse.Namespace, settings: dict[str, Any]) -> None:
    """Show the active run plan in a compact agent-style banner."""
    if args.quiet:
        return
    embedding = settings.get("embedding", {})
    llm = settings.get("llm", {})
    details = Table.grid(padding=(0, 1))
    details.add_column(style="bright_black")
    details.add_column(style="#767676")
    details.add_row("[bold]Run[/bold]", "")
    details.add_row("Profile", str(args.profile))
    if args.command in {"search", "legacy-search", "check", "legacy-suggest"}:
        details.add_row("Sources", str(settings.get("db")))
    details.add_row("[bold]Models[/bold]", "")
    details.add_row("Embeddings", f"{embedding.get('provider')} · {embedding.get('model')}")
    details.add_row("LLM default", f"{llm.get('provider')} · {llm.get('model')}")
    for stage, provider, model, enabled in _llm_stage_overview(llm):
        status = "[green]active[/green]" if enabled else "[bright_black]inactive[/bright_black]"
        details.add_row(f"{stage}", f"{provider} · {model} · {status}")
    if args.command not in {"search", "legacy-search", "check", "legacy-suggest"}:
        details.add_row("Index", str(settings.get("db")))
    if args.command in {"search", "legacy-search"}:
        details.add_row("Task", "Find and rank evidence for one claim")
    elif args.command in {"check", "legacy-suggest"}:
        details.add_row("Task", "Check draft claims and suggest citations")
        if args.command == "check":
            details.add_row("Claim workers", str(args.workers))
    elif args.command == "sync":
        details.add_row("Task", "Sync collection PDFs")
    elif args.command in {"ingest-pdfs", "ingest"}:
        details.add_row("Task", "Index local PDFs")
    elif args.command == "status":
        details.add_row("Task", "Inspect index and provider readiness")
    title = {
        "search": "Bibliograph · Claim check",
        "legacy-search": "Bibliograph · Claim check",
        "check": "Bibliograph · Draft check",
        "legacy-suggest": "Bibliograph · Draft check",
    }.get(args.command, "Bibliograph")
    Console(stderr=True).print(
        Panel(
            details,
            title=f"[bold cyan]{title}[/bold cyan]",
            subtitle=f"[dim]{args.command}[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _llm_stage_overview(llm: dict[str, Any]) -> list[tuple[str, str, str, bool]]:
    overrides = llm.get("models", {})
    enabled_stages = set(llm.get("stages", ()))
    active_mode = llm.get("mode") != "off"
    stages = (
        ("expand", "Query expansion"),
        ("rerank", "Reranking"),
        ("evidence", "Evidence selection"),
        ("rationale", "Rationale"),
    )
    overview = []
    for key, label in stages:
        override = overrides.get(key, {})
        if isinstance(override, str):
            override = {"model": override}
        overview.append(
            (
                label,
                str(override.get("provider", llm.get("provider"))),
                str(override.get("model", llm.get("model"))),
                active_mode and key in enabled_stages,
            )
        )
    return overview


def _print_configuration_summary(config_path: Path | None, profile: str) -> None:
    """Print resolved non-secret settings without probing configured services."""
    selected_path = Path("bibliograph.toml") if config_path is None else config_path
    try:
        settings = load_settings(config_path, profile)
    except Exception as error:
        Console().print(
            f"\nConfiguration\n  path: {selected_path}\n  profile: {profile}\n"
            f"  error: {error}",
            markup=False,
        )
        return

    embedding = settings.get("embedding", {})
    llm = settings.get("llm", {})
    stages = ", ".join(str(stage) for stage in llm.get("stages", ())) or "none"
    lines = [
        "",
        "Configuration",
        f"  path: {selected_path}",
        f"  profile: {profile}",
        f"  database: {settings.get('db')}",
        f"  PDF directory: {settings.get('pdf_dir')}",
        f"  collection: {settings.get('collection')}",
        "",
        "Providers",
        f"  embeddings: {embedding.get('provider')} / {embedding.get('model')}",
        f"    endpoint: {embedding.get('base_url')}",
        f"  LLM: {llm.get('provider')} / {llm.get('model')}",
        f"    endpoint: {llm.get('base_url')}",
        f"    mode: {llm.get('mode')}",
        f"    stages: {stages}",
    ]
    Console().print("\n".join(lines), markup=False)


def _render_ingest_summary(result: dict[str, Any]) -> str:
    lines = [
        f"# PDF ingestion: {result['directory']}",
        "",
        f"- PDFs found: {result['discovered']}",
        f"- Indexed: {len(result['indexed'])}",
        f"- Unchanged: {len(result['unchanged'])}",
        f"- Invalid PDFs skipped: {len(result['invalid'])}",
    ]
    for item in result["indexed"]:
        lines.append(f"- Indexed `{item['path']}` ({item['chunks']} chunks)")
    for path in result["invalid"]:
        lines.append(f"- Skipped invalid PDF `{path}`")
    return "\n".join(lines)


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
    marker_pattern = re.compile(
        r"⟦(?P<kind>highlight|support|partial|contradict|mixed)⟧"
        r"(?P<text>.*?)⟦/(?P=kind)⟧",
        re.DOTALL,
    )
    if output is not None:
        markdown = marker_pattern.sub(lambda match: match.group("text"), markdown)
        output.write_text(markdown, encoding="utf-8")
        get_logger("cli").info("Wrote report: %s", output)
        return

    styles = {
        "highlight": "bold black on yellow",
        "support": "bold green",
        "partial": "bold magenta",
        "contradict": "bold red",
        "mixed": "bold yellow",
    }
    console = Console()
    position = 0
    for match in marker_pattern.finditer(markdown):
        if match.start() > position:
            console.print(Markdown(markdown[position : match.start()]), end="")
        console.print(Text(match.group("text"), style=styles[match.group("kind")]), end="")
        # Rich's Markdown parser strips leading whitespace from the next chunk.
        # Preserve a separator that belongs between this styled phrase and the
        # following plain text before passing that text back through Markdown.
        if markdown[match.end() :].startswith(" "):
            console.print(" ", end="")
        position = match.end()
    if position < len(markdown):
        console.print(Markdown(markdown[position:]), end="")
