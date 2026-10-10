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

from adapters.providers.model_runtimes.registry import build_chat, build_embedding

from .adapters.acquisition import (
    cache_strategy,
    download_remote_pdf,
    remote_strategy,
    zotero_api_strategy,
    zotero_storage_strategy,
)
from .adapters.ikarus import (
    IkarusBackend,
    inspect_ikarus_index,
    is_ikarus_index,
    open_ikarus_staging,
)
from .adapters.pdf import extract_pages
from .adapters.sqlite import SQLiteStore
from .adapters.zotero import build_zotero, zotero_storage_dirs
from .commands.check import check
from .commands.ingest_pdfs import ingest_pdfs
from .commands.search import search
from .commands.status import status
from .commands.sync import sync_library
from .logging_utils import get_logger
from .pipeline.classification import paper_evidence_strategy
from .pipeline.llm_tasks import (
    expand_query,
    explain,
    heuristic_expand,
    heuristic_rerank,
    refine_queries,
    rerank,
    rerank_with_systemone,
    select_evidence,
    template_rationale,
)
from .pipeline.retrieval import DEFAULT_SEARCH_LIMIT
from .settings import Settings

logger = get_logger("bootstrap")


def run_sync(
    settings: Settings,
    *,
    collection: str | None = None,
    rebuild: bool = False,
    show_progress: bool = True,
    force_reindex: bool = False,
) -> dict:
    embedding = build_embedding(settings)
    # A successful HTTP listener is insufficient: this performs a capability
    # request against the exact model before a staging rebuild is opened.
    embedding["probe"]()
    classification = _paper_evidence_strategy(settings)
    zotero = build_zotero(settings.get("zotero", {}))
    strategies = _acquisition_strategies(settings, zotero)
    storage_dirs = zotero_storage_dirs(settings.get("zotero_storage_dir"))
    database = settings["db"]

    if rebuild:
        with open_ikarus_staging(
            database,
            embedding=embedding,
            ingestion_strategy=classification,
            classification_enabled=settings.get("classification", {}).get("enabled", False),
        ) as store:
            return sync_library(
                settings,
                store=store,
                zotero=zotero,
                strategies=strategies,
                extract_pages=extract_pages,
                storage_dirs=storage_dirs,
                collection=collection,
                show_progress=show_progress,
                force_reindex=force_reindex,
            )

    with IkarusBackend(
        database,
        mode="write",
        embedding=embedding,
        ingestion_strategy=classification,
        classification_enabled=settings.get("classification", {}).get("enabled", False),
    ) as store:
        return sync_library(
            settings,
            store=store,
            zotero=zotero,
            strategies=strategies,
            extract_pages=extract_pages,
            storage_dirs=storage_dirs,
            collection=collection,
            show_progress=show_progress,
            force_reindex=force_reindex,
        )


def run_ingest_pdfs(
    settings: Settings,
    directory: str | Path,
    *,
    recursive: bool = True,
    show_progress: bool = True,
    force_reindex: bool = False,
) -> dict:
    embedding = build_embedding(settings)
    embedding["probe"]()
    classification = _paper_evidence_strategy(settings)
    with IkarusBackend(
        settings["db"],
        mode="write",
        embedding=embedding,
        ingestion_strategy=classification,
        classification_enabled=settings.get("classification", {}).get("enabled", False),
    ) as store:
        return ingest_pdfs(
            directory,
            store=store,
            extract_pages=extract_pages,
            recursive=recursive,
            show_progress=show_progress,
            force_reindex=force_reindex,
        )


def run_search(
    settings: Settings,
    claim: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
    min_score: float = 0.0,
    no_llm: bool = False,
    disabled_stages: Iterable[str] = (),
    show_progress: bool = True,
    enrich: bool = True,
    one_per_paper: bool = False,
    verbose: bool = False,
) -> dict:
    embedding = build_embedding(settings)
    with IkarusBackend(
        settings["db"],
        mode="read",
        embedding=embedding,
        classification_enabled=settings.get("classification", {}).get("enabled", False),
    ) as store:
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
            one_per_paper=one_per_paper,
            verbose=verbose,
        )


def run_check(
    settings: Settings,
    draft: str | Path,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
    min_score: float = 0.0,
    no_llm: bool = False,
    disabled_stages: Iterable[str] = (),
    show_progress: bool = True,
    enrich: bool = True,
    workers: int = 1,
    verbose: bool = False,
) -> dict:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers > 1:
        return check(
            draft,
            backend=None,
            llm_tools={},
            limit=limit,
            min_score=min_score,
            show_progress=show_progress,
            enrich=enrich,
            workers=workers,
            verbose=verbose,
            worker_settings=settings,
            no_llm=no_llm,
            disabled_stages=tuple(disabled_stages),
        )

    embedding = build_embedding(settings)
    with IkarusBackend(
        settings["db"],
        mode="read",
        embedding=embedding,
        classification_enabled=settings.get("classification", {}).get("enabled", False),
    ) as store:
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
            verbose=verbose,
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
    if is_ikarus_index(database):
        summary = inspect_ikarus_index(database)
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


def _paper_evidence_strategy(settings: Settings):
    """Build optional ingest-time paper-role classification from configured chat."""
    config = settings.get("classification", {})
    if not config.get("enabled", False):
        return None
    llm_settings = dict(settings["llm"])
    llm_settings["model"] = config.get("model") or llm_settings["model"]
    try:
        complete = build_chat({**settings, "llm": llm_settings})["complete"]
        strategy = paper_evidence_strategy(complete)
    except Exception as error:
        if settings["llm"]["mode"] == "required":
            raise
        logger.warning(
            "Paper evidence classifier unavailable; indexing without labels: %s", error
        )
        return None
    if settings["llm"]["mode"] == "required":
        return strategy

    def classify(text: str):
        try:
            return strategy(text)
        except Exception as error:
            logger.warning(
                "Paper evidence classification failed; keeping chunk unlabelled: %s", error
            )
            return {"text": text}

    return classify


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
    model_overrides = llm_settings.get("models", {})
    chats: dict[tuple[Any, ...], Any] = {}

    def stage_tool(
        stage: str, label: str, factory: Callable, fallback: Callable | None = None
    ) -> Callable:
        fallback_keys = {
            "expand": "expand",
            "rerank": "rerank",
            "evidence": "select_evidence",
            "rationale": "explain",
        }
        fallback_tool = fallback or fallbacks.get(fallback_keys[stage])
        override = model_overrides.get(stage, {})
        if isinstance(override, str):
            override = {"model": override}
        stage_settings = dict(llm_settings)
        stage_settings.update(override)
        cache_key = (
            stage_settings.get("provider"),
            stage_settings.get("model"),
            stage_settings.get("base_url"),
            stage_settings.get("api_key"),
        )
        try:
            if cache_key not in chats:
                chats[cache_key] = build_chat({**settings, "llm": stage_settings})
            primary = factory(chats[cache_key]["complete"])
        except Exception as error:
            if mode == "required":
                raise
            logger.warning("LLM %s initialization unavailable; using fallback: %s", label, error)
            return fallback_tool
        if mode == "required":
            return primary

        @wraps(primary)
        def run(*args, **kwargs):
            try:
                return primary(*args, **kwargs)
            except Exception as error:
                logger.warning("LLM %s unavailable; using fallback: %s", label, error)
                return fallback_tool(*args, **kwargs)

        return run

    tools: dict[str, Callable] = {}
    if "rerank" in stages:
        if llm_settings.get("rerank_strategy", "systemone") == "systemone":
            def run_systemone_reranker(claim, hits, progress=None):
                from adapters.providers.decisions import build_decider

                decision_config = {"provider": "ollama"}
                if llm_settings.get("systemone_model"):
                    decision_config["model"] = llm_settings["systemone_model"]
                decider = build_decider(decision_config)
                return rerank_with_systemone(
                    claim,
                    hits,
                    decider=decider,
                    workers=llm_settings.get("systemone_workers", 4),
                    progress=progress,
                )

            if mode == "required":
                tools["rerank"] = run_systemone_reranker
            else:
                @wraps(run_systemone_reranker)
                def optional_systemone_reranker(*args, **kwargs):
                    try:
                        return run_systemone_reranker(*args, **kwargs)
                    except Exception as error:
                        logger.warning(
                            "Ollama reranking unavailable; using retrieval order: %s", error
                        )
                        return heuristic_rerank(args[0], args[1])

                tools["rerank"] = optional_systemone_reranker
        else:
            tools["rerank"] = stage_tool(
                "rerank", "reranking", lambda complete: lambda claim, hits, progress=None: rerank(
                    claim, hits, complete=complete, progress=progress
                )
            )
    if "expand" in stages:
        tools["expand"] = stage_tool(
            "expand", "query expansion", lambda complete: lambda claim, progress=None: expand_query(
                claim, complete=complete, progress=progress
            )
        )
        tools["refine"] = stage_tool(
            "expand",
            "query refinement",
            lambda complete: lambda claim, excerpts: refine_queries(
                claim, excerpts, complete=complete
            ),
            fallback=lambda _claim, _excerpts: [],
        )
    if "evidence" in stages:
        tools["select_evidence"] = stage_tool(
            "evidence",
            "evidence extraction",
            lambda complete: lambda claim, hit, context, components=None: select_evidence(
                claim, hit, context, complete=complete, components=components
            ),
        )
    if "rationale" in stages:
        tools["explain"] = stage_tool(
            "rationale", "rationale generation", lambda complete: lambda claim, evidence:
            explain(claim, evidence, complete=complete)
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
        tools["select_evidence"] = lambda _claim, _hit, _context, components=None: None
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
