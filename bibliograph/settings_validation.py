"""Validation rules for resolved Bibliograph settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from adapters.providers.model_runtimes.registry import chat_names, embedding_names, resolve_name

_LLM_STAGES = frozenset({"expand", "rerank", "evidence", "rationale"})
_ACQUISITION_STRATEGIES = frozenset({"cache", "zotero-storage", "zotero-api", "remote"})


def _validate(settings: Mapping[str, Any]) -> None:
    embedding = settings.get("embedding")
    llm = settings.get("llm")
    if not isinstance(embedding, Mapping) or not isinstance(llm, Mapping):
        raise ValueError("embedding and llm settings must be tables")

    _validate_pdf_extraction(settings.get("pdf_extraction"))
    _validate_embedding(embedding)
    mode = _validate_llm(llm)
    _validate_acquisition(settings.get("acquisition"))
    _validate_classification(settings.get("classification"), mode)
    _validate_remote(settings.get("remote"))


def _validate_pdf_extraction(config: Any) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("pdf_extraction settings must be a table")
    if config.get("provider") not in {"pymupdf", "ollama"}:
        raise ValueError("pdf_extraction.provider must be 'pymupdf' or 'ollama'")
    if config.get("ollama_mode") not in {"hybrid", "all"}:
        raise ValueError("pdf_extraction.ollama_mode must be 'hybrid' or 'all'")
    model = config.get("ollama_model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("pdf_extraction.ollama_model must be a non-empty model name")
    base_url = config.get("ollama_base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("pdf_extraction.ollama_base_url must be a non-empty URL")
    min_text_chars = config.get("min_text_chars")
    if (
        not isinstance(min_text_chars, int)
        or isinstance(min_text_chars, bool)
        or min_text_chars < 1
    ):
        raise ValueError("pdf_extraction.min_text_chars must be a positive integer")


def _validate_embedding(config: Mapping[str, Any]) -> None:
    _validate_provider(config, section="embedding", choices=embedding_names())
    batch_size = config.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("embedding.batch_size must be a positive integer")


def _validate_llm(config: Mapping[str, Any]) -> str:
    _validate_provider(config, section="llm", choices=chat_names())
    mode = config.get("mode")
    if mode not in {"off", "optional", "required"}:
        raise ValueError("llm.mode must be one of: off, optional, required")
    if config.get("rerank_strategy") not in {"llm", "systemone"}:
        raise ValueError("llm.rerank_strategy must be one of: llm, systemone")
    model = config.get("systemone_model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("llm.systemone_model must be a non-empty model name")
    workers = config.get("systemone_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("llm.systemone_workers must be a positive integer")

    stages = config.get("stages")
    if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
        raise ValueError("llm.stages must be a list of strings")
    _validate_known_names(stages, _LLM_STAGES, label="llm.stages", unknown_description="stage(s)")
    _validate_model_overrides(config.get("models", {}))
    return mode


def _validate_model_overrides(models: Any) -> None:
    if not isinstance(models, Mapping):
        raise ValueError("llm.models must be a table mapping stages to model settings")
    _validate_known_names(models, _LLM_STAGES, label="llm.models", unknown_description="stage(s)")
    for stage, override in models.items():
        if isinstance(override, str):
            if not override.strip():
                raise ValueError(f"llm.models.{stage} must be a non-empty model name")
            continue
        if not isinstance(override, Mapping):
            raise ValueError(
                f"llm.models.{stage} must be a model name or a table of provider settings"
            )
        unknown_options = sorted(set(override) - {"provider", "model", "base_url", "api_key"})
        if unknown_options:
            raise ValueError(
                f"llm.models.{stage} contains unsupported setting(s): {', '.join(unknown_options)}"
            )
        if "provider" in override and resolve_name(override["provider"], kind="chat") is None:
            raise ValueError(
                f"llm.models.{stage}.provider must be one of: {', '.join(chat_names())}"
            )
        if "model" in override and (
            not isinstance(override["model"], str) or not override["model"].strip()
        ):
            raise ValueError(f"llm.models.{stage}.model must be a non-empty model name")
        for option in ("base_url", "api_key"):
            if (
                option in override
                and override[option] is not None
                and not isinstance(override[option], str)
            ):
                raise ValueError(f"llm.models.{stage}.{option} must be a string")


def _validate_known_names(
    values: Any,
    allowed: frozenset[str],
    *,
    label: str,
    unknown_description: str,
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"{label} contains unknown {unknown_description}: {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(allowed))}"
        )


def _validate_acquisition(config: Any) -> None:
    if (
        not isinstance(config, Mapping)
        or not isinstance(config.get("order"), list)
        or not all(isinstance(name, str) for name in config["order"])
    ):
        raise ValueError("acquisition.order must be a list")
    _validate_known_names(
        config["order"],
        _ACQUISITION_STRATEGIES,
        label="acquisition.order",
        unknown_description="strategy/strategies",
    )


def _validate_classification(config: Any, llm_mode: str) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("classification settings must be a table")
    if not isinstance(config.get("enabled"), bool):
        raise ValueError("classification.enabled must be a boolean")
    model = config.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("classification.model must be a non-empty model name")
    if config["enabled"] and llm_mode == "off":
        raise ValueError("classification.enabled requires llm.mode to be optional or required")


def _validate_remote(config: Any) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("remote settings must be a table")
    if not isinstance(config.get("playwright_headless"), bool):
        raise ValueError("remote.playwright_headless must be a boolean")


def _validate_provider(
    config: Mapping[str, Any], *, section: str, choices: tuple[str, ...]
) -> None:
    provider = config.get("provider")
    kind = "embedding" if section == "embedding" else "chat"
    if resolve_name(provider, kind=kind) is None:
        raise ValueError(f"{section}.provider must be one of: {', '.join(choices)}")
