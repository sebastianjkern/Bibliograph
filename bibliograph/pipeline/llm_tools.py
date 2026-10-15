"""Construction and graceful fallback behavior for configured LLM tools."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from functools import wraps
from threading import Lock
from time import sleep
from typing import Any

from adapters.providers.model_runtimes.registry import build_chat

from ..logging_utils import get_logger
from ..pipeline.llm_tasks import (
    EVIDENCE_ASSESSMENT_CACHE_VERSION,
    RERANK_CACHE_VERSION,
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
from ..settings import Settings

logger = get_logger("bootstrap")


class _UncacheableList(list):
    """Mark optional fallback results so they are never persisted as model output."""

    cacheable = False


def _llm_tools(
    settings: Settings,
    *,
    disabled: bool = False,
    disabled_stages: Iterable[str] = (),
) -> dict[str, Callable]:
    """Create configured tools; keep stage setup separate from runtime policy."""
    llm_settings = settings["llm"]
    mode = "off" if disabled else llm_settings["mode"]
    stages = set(llm_settings.get("stages", ())) - set(disabled_stages)
    if mode == "off" or not stages:
        return {}

    fallbacks = _deterministic_llm_tools(stages)
    chats: dict[tuple[Any, ...], Any] = {}
    tools: dict[str, Callable] = {}

    if "rerank" in stages:
        if llm_settings.get("rerank_strategy", "systemone") == "systemone":
            tools["rerank"] = _build_systemone_reranker(llm_settings, mode)
        else:
            tools["rerank"] = _build_stage_tool(
                settings,
                llm_settings,
                mode,
                fallbacks,
                chats,
                "rerank",
                "reranking",
                _rerank_factory,
            )

    if "expand" in stages:
        tools["expand"] = _build_stage_tool(
            settings,
            llm_settings,
            mode,
            fallbacks,
            chats,
            "expand",
            "query expansion",
            _expand_factory,
        )
        tools["refine"] = _build_stage_tool(
            settings,
            llm_settings,
            mode,
            fallbacks,
            chats,
            "expand",
            "query refinement",
            _refine_factory,
            fallback=lambda _claim, _excerpts: [],
        )

    if "evidence" in stages:
        tools["select_evidence"] = _build_stage_tool(
            settings,
            llm_settings,
            mode,
            fallbacks,
            chats,
            "evidence",
            "evidence extraction",
            _evidence_factory,
        )

    if "rationale" in stages:
        tools["explain"] = _build_stage_tool(
            settings,
            llm_settings,
            mode,
            fallbacks,
            chats,
            "rationale",
            "rationale generation",
            _rationale_factory,
        )
    return tools


def _build_stage_tool(
    settings: Settings,
    llm_settings: Mapping[str, Any],
    mode: str,
    fallbacks: Mapping[str, Callable],
    chats: dict[tuple[Any, ...], Any],
    stage: str,
    label: str,
    factory: Callable[[Callable], Callable],
    fallback: Callable | None = None,
) -> Callable:
    fallback_key = {
        "expand": "expand",
        "rerank": "rerank",
        "evidence": "select_evidence",
        "rationale": "explain",
    }[stage]
    fallback_tool = fallback or fallbacks.get(fallback_key)
    if fallback_tool is None:
        raise ValueError(f"No deterministic fallback configured for LLM stage {stage!r}")

    stage_settings = dict(llm_settings)
    override = llm_settings.get("models", {}).get(stage, {})
    if isinstance(override, str):
        override = {"model": override}
    stage_settings.update(override)
    evidence_failure = _evidence_provider_failure(label) if stage == "evidence" else None

    cache_key = tuple(
        stage_settings.get(key) for key in ("provider", "model", "base_url", "api_key")
    )
    try:
        if cache_key not in chats:
            chats[cache_key] = build_chat({**settings, "llm": stage_settings})
        complete = chats[cache_key]["complete"]
        if str(stage_settings.get("provider", "")).casefold() == "codex":
            provider_complete = complete

            def complete(messages, **kwargs):
                return provider_complete(_replace_null_bytes(messages), **kwargs)

        primary = factory(complete)
    except Exception as error:
        if mode == "required":
            raise
        if evidence_failure is not None:
            return lambda *_args, _error=error, **_kwargs: evidence_failure(_error)
        logger.warning("LLM %s initialization unavailable; using fallback: %s", label, error)
        return fallback_tool

    cache_identity = _cache_identity(stage, stage_settings)
    if cache_identity is not None:
        setattr(primary, f"{stage}_cache_identity", cache_identity)
    if mode == "required":
        return primary

    @wraps(primary)
    def run(*args, **kwargs):
        try:
            return primary(*args, **kwargs)
        except Exception as error:
            if stage == "evidence" and _is_transient_provider_error(error):
                sleep(0.25)
                try:
                    return primary(*args, **kwargs)
                except Exception as retry_error:
                    error = retry_error
            if evidence_failure is not None:
                return evidence_failure(error)
            logger.warning("LLM %s unavailable; using fallback: %s", label, error)
            result = fallback_tool(*args, **kwargs)
            if stage == "rerank" and isinstance(result, list):
                return _UncacheableList(result)
            return result

    if cache_identity is not None:
        setattr(run, f"{stage}_cache_identity", cache_identity)
    return run


def _evidence_provider_failure(label: str) -> Callable[[Exception], dict]:
    warning_lock = Lock()
    warning_emitted = False

    def unavailable_evidence(*_args, **_kwargs):
        return {
            "relation": "unavailable",
            "assessment_status": "provider_unavailable",
            "matched_quote": "",
            "reason": "Evidence assessment could not be completed because the model provider "
            "was unavailable.",
            "scope": {},
            "components": [],
            "__cacheable__": False,
        }

    def handle(error: Exception) -> dict:
        nonlocal warning_emitted
        with warning_lock:
            if not warning_emitted:
                logger.warning(
                    "LLM %s unavailable; affected passages will be marked not assessed: %s",
                    label,
                    error,
                )
                warning_emitted = True
        return unavailable_evidence()

    return handle


def _cache_identity(stage: str, settings: Mapping[str, Any]) -> str | None:
    if stage not in {"evidence", "rerank"}:
        return None
    version = EVIDENCE_ASSESSMENT_CACHE_VERSION if stage == "evidence" else RERANK_CACHE_VERSION
    safe_settings = {
        key: value for key, value in settings.items() if key not in {"api_key", "token", "password"}
    }
    return json.dumps({"version": version, "settings": safe_settings}, sort_keys=True, default=str)


def _build_systemone_reranker(llm_settings: Mapping[str, Any], mode: str) -> Callable:
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

    cache_identity = json.dumps(
        {
            "version": RERANK_CACHE_VERSION,
            "strategy": "systemone",
            "provider": "ollama",
            "model": llm_settings.get("systemone_model") or "nimble",
            "workers": llm_settings.get("systemone_workers", 4),
        },
        sort_keys=True,
    )
    setattr(run_systemone_reranker, "rerank_cache_identity", cache_identity)  # noqa: B010
    if mode == "required":
        return run_systemone_reranker

    @wraps(run_systemone_reranker)
    def optional_reranker(*args, **kwargs):
        try:
            return run_systemone_reranker(*args, **kwargs)
        except Exception as error:
            logger.warning("Ollama reranking unavailable; using retrieval order: %s", error)
            return _UncacheableList(heuristic_rerank(args[0], args[1]))

    setattr(optional_reranker, "rerank_cache_identity", cache_identity)  # noqa: B010
    return optional_reranker


def _rerank_factory(complete):
    return lambda claim, hits, progress=None: rerank(
        claim, hits, complete=complete, progress=progress
    )


def _expand_factory(complete):
    return lambda claim, progress=None: expand_query(claim, complete=complete, progress=progress)


def _refine_factory(complete):
    return lambda claim, excerpts: refine_queries(claim, excerpts, complete=complete)


def _evidence_factory(complete):
    return lambda claim, hit, context, components=None: select_evidence(
        claim, hit, context, complete=complete, components=components
    )


def _rationale_factory(complete):
    return lambda claim, evidence: explain(claim, evidence, complete=complete)


def _is_transient_provider_error(error: Exception) -> bool:
    message = str(error).casefold()
    return any(
        marker in message
        for marker in (
            "timed out",
            "timeout",
            "cannot connect",
            "connection refused",
            "connection reset",
            "temporarily unavailable",
            "http 429",
            "http 502",
            "http 503",
            "http 504",
        )
    )


def _replace_null_bytes(value: Any) -> Any:
    """Remove NULs from Codex prompt payloads before they reach subprocess argv."""
    if isinstance(value, str):
        return value.replace("\x00", " ")
    if isinstance(value, Mapping):
        return {key: _replace_null_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_null_bytes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_null_bytes(item) for item in value)
    return value


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
