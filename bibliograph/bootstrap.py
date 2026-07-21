"""Composition root for Bibliograph's command handlers.

This is the only module that knows concrete providers, source adapters, and
the command dependency graph.  Command and pipeline modules receive bound
callables or small stateful resources instead of selecting implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from typing import Any

from .adapters.acquisition import (
    cache_strategy,
    download_remote_pdf,
    remote_strategy,
    zotero_api_strategy,
    zotero_storage_strategy,
)
from .adapters.pdf import extract_pages
from .adapters.ragtime import (
    RagtimeBackend,
    inspect_ragtime_index,
    is_ragtime_index,
    open_ragtime_staging,
)
from .adapters.sqlite import SQLiteStore
from .adapters.zotero import build_zotero, zotero_storage_dirs
from .commands.check import check
from .commands.search import search
from .commands.status import status
from .commands.sync import sync_library
from .logging_utils import get_logger
from .pipeline.llm_tasks import (
    expand_query,
    explain,
    heuristic_expand,
    heuristic_rerank,
    rerank,
    select_evidence,
    template_rationale,
)
from .providers.registry import build_chat, build_embedding
from .settings import Settings

logger = get_logger("bootstrap")


def run_sync(
    settings: Settings,
    *,
    collection: str | None = None,
    rebuild: bool = False,
    show_progress: bool = True,
) -> dict:
    embedding = build_embedding(settings)
    # A successful HTTP listener is insufficient: this performs a capability
    # request against the exact model before a staging rebuild is opened.
    embedding["probe"]()
    zotero = build_zotero(settings.get("zotero", {}))
    strategies = _acquisition_strategies(settings, zotero)
    storage_dirs = zotero_storage_dirs(settings.get("zotero_storage_dir"))
    database = settings["db"]

    if rebuild:
        with open_ragtime_staging(database, embedding=embedding) as store:
            return sync_library(
                settings,
                store=store,
                zotero=zotero,
                strategies=strategies,
                extract_pages=extract_pages,
                storage_dirs=storage_dirs,
                collection=collection,
                show_progress=show_progress,
            )

    with RagtimeBackend(database, mode="write", embedding=embedding) as store:
        return sync_library(
            settings,
            store=store,
            zotero=zotero,
            strategies=strategies,
            extract_pages=extract_pages,
            storage_dirs=storage_dirs,
            collection=collection,
            show_progress=show_progress,
        )


def run_search(
    settings: Settings,
    claim: str,
    *,
    limit: int = 5,
    min_score: float = 0.0,
    no_llm: bool = False,
    disabled_stages: Iterable[str] = (),
    show_progress: bool = True,
    enrich: bool = True,
) -> dict:
    embedding = build_embedding(settings)
    with RagtimeBackend(settings["db"], mode="read", embedding=embedding) as store:
        return search(
            claim,
            backend=store,
            llm_tools=_llm_tools(
                settings,
                disabled=no_llm,
                disabled_stages=disabled_stages,
            ),
            limit=limit,
            min_score=min_score,
            show_progress=show_progress,
            enrich=enrich,
        )


def run_check(
    settings: Settings,
    draft: str | Path,
    *,
    limit: int = 5,
    min_score: float = 0.0,
    no_llm: bool = False,
    disabled_stages: Iterable[str] = (),
    show_progress: bool = True,
    enrich: bool = True,
) -> dict:
    embedding = build_embedding(settings)
    with RagtimeBackend(settings["db"], mode="read", embedding=embedding) as store:
        return check(
            draft,
            backend=store,
            llm_tools=_llm_tools(
                settings,
                disabled=no_llm,
                disabled_stages=disabled_stages,
            ),
            limit=limit,
            min_score=min_score,
            show_progress=show_progress,
            enrich=enrich,
        )


def run_status(settings: Settings, *, probe: bool = False) -> dict:
    database = Path(settings["db"])
    probes: list[Callable[[], dict]] = []
    if probe:
        probes.append(_provider_probe("embedding", lambda: build_embedding(settings)))
        if settings["llm"]["mode"] != "off":
            probes.append(_provider_probe("LLM", lambda: build_chat(settings)))
        probes.append(_zotero_probe(settings))
    if not database.is_file():
        return status(path=database, probes=probes)
    if is_ragtime_index(database):
        summary = inspect_ragtime_index(database)
        return status(_StaticStats(summary), probes=probes)
    # Legacy indexes remain inspectable, but no search or sync path uses their RAG backend.
    with SQLiteStore(database, mode="read") as store:
        summary = dict(store.stats())
        summary["rebuild_required"] = True
        return status(_StaticStats(summary), probes=probes)


class _StaticStats:
    def __init__(self, value: dict) -> None:
        self.value = value

    def stats(self) -> dict:
        return self.value


def _llm_tools(
    settings: Settings,
    *,
    disabled: bool = False,
    disabled_stages: Iterable[str] = (),
) -> dict[str, Callable]:
    llm_settings = settings["llm"]
    mode = "off" if disabled else llm_settings["mode"]
    stages = set(llm_settings.get("stages", ())) - set(disabled_stages)
    if mode == "off" or not stages:
        return {}

    fallbacks = _deterministic_llm_tools(stages)
    try:
        chat = build_chat(settings)
    except Exception as error:
        if mode == "required":
            raise
        logger.warning("LLM initialization unavailable; using configured fallbacks: %s", error)
        return fallbacks
    complete = chat["complete"]

    def optional(stage: str, primary: Callable, fallback: Callable) -> Callable:
        if mode == "required":
            return primary

        @wraps(primary)
        def run(*args, **kwargs):
            try:
                return primary(*args, **kwargs)
            except Exception as error:
                logger.warning("LLM %s unavailable; using fallback: %s", stage, error)
                return fallback(*args, **kwargs)

        return run

    tools: dict[str, Callable] = {}
    if "rerank" in stages:
        tools["rerank"] = optional(
            "reranking",
            lambda claim, hits, progress=None: rerank(
                claim,
                hits,
                complete=complete,
                progress=progress,
            ),
            fallbacks["rerank"],
        )
    if "expand" in stages:
        tools["expand"] = optional(
            "query expansion",
            lambda claim, progress=None: expand_query(
                claim,
                complete=complete,
                progress=progress,
            ),
            fallbacks["expand"],
        )
    if "evidence" in stages:
        tools["select_evidence"] = optional(
            "evidence extraction",
            lambda claim, hit, context: select_evidence(claim, hit, context, complete=complete),
            fallbacks["select_evidence"],
        )
    if "rationale" in stages:
        tools["explain"] = optional(
            "rationale generation",
            lambda claim, evidence: explain(claim, evidence, complete=complete),
            fallbacks["explain"],
        )
    return tools


def _deterministic_llm_tools(stages: set[str]) -> dict[str, Callable]:
    """Return only the deterministic fallbacks selected by configured stages."""

    tools: dict[str, Callable] = {}
    if "expand" in stages:
        tools["expand"] = heuristic_expand
    if "rerank" in stages:
        tools["rerank"] = heuristic_rerank
    if "evidence" in stages:
        tools["select_evidence"] = lambda _claim, _hit, _context: None
    if "rationale" in stages:
        tools["explain"] = template_rationale
    return tools


def _acquisition_strategies(settings: Settings, zotero: Any) -> list[Callable]:
    strategies: list[Callable] = []
    storage_dirs = zotero_storage_dirs(settings.get("zotero_storage_dir"))
    for name in settings["acquisition"].get("order", ()):
        if name == "cache":
            strategies.append(cache_strategy(settings["pdf_dir"]))
        elif name == "zotero-storage":
            strategies.append(zotero_storage_strategy(settings["pdf_dir"], storage_dirs))
        elif name == "zotero-api":
            strategies.append(zotero_api_strategy(zotero, settings["pdf_dir"]))
        elif name == "remote":
            strategies.append(remote_strategy(_remote_downloader(settings), settings["pdf_dir"]))
        else:
            raise ValueError(f"Unknown acquisition strategy: {name}")
    return strategies


def _remote_downloader(settings: Settings) -> Callable:
    remote = settings.get("remote", {})

    def download(doi: str, output_dir: str | Path, **kwargs):
        return download_remote_pdf(
            doi,
            output_dir,
            email=remote.get("email"),
            openalex_api_key=remote.get("openalex_api_key"),
            playwright_profile=remote.get("playwright_profile"),
            playwright_headless=remote.get("playwright_headless", True),
            **kwargs,
        )

    return download


def _provider_probe(capability: str, build_runtime: Callable[[], dict]) -> Callable[[], dict]:
    def run() -> dict:
        try:
            runtime = build_runtime()
            result = dict(runtime["probe"]())
            result["capability"] = capability
            return result
        except Exception as error:
            logger.warning("%s probe failed: %s", capability, error)
            return {"ok": False, "capability": capability, "error": str(error)}

    return run


def _zotero_probe(settings: Settings) -> Callable[[], dict]:
    """Check configured Zotero access lazily so status always keeps DB output."""

    def run() -> dict:
        try:
            zotero = build_zotero(settings.get("zotero", {}))
            collections = zotero.collections()
            everything = getattr(zotero, "everything", None)
            if callable(everything):
                everything(collections)
            return {"ok": True, "capability": "Zotero"}
        except Exception as error:
            logger.warning("Zotero probe failed: %s", error)
            return {"ok": False, "capability": "Zotero", "error": str(error)}

    return run
