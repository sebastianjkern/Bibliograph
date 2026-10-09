"""Project settings resolution for Bibliograph's semi-static pipelines.

Provider adapters receive the resulting dictionaries directly.  Keeping all
environment handling here means provider modules are deterministic and are
easy to exercise with fake clients in tests.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from adapters.providers.model_runtimes.registry import chat_names, embedding_names, resolve_name

type Settings = dict[str, Any]

_LLM_STAGES = frozenset({"expand", "rerank", "evidence", "rationale"})
_ACQUISITION_STRATEGIES = frozenset({"cache", "zotero-storage", "zotero-api", "remote"})


DEFAULT_SETTINGS: Settings = {
    "db": "bibliograph.db",
    "pdf_dir": "pdfs",
    "collection": "My Collection",
    "zotero_storage_dir": None,
    "zotero": {
        "library_id": None,
        "library_type": "user",
        "api_key": None,
        "storage_dir": None,
    },
    "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
        "api_key": None,
        "batch_size": 32,
        "cache_dir": None,
        "local_files_only": False,
        "document_prefix": "",
        "query_prefix": "",
    },
    "llm": {
        "provider": "ollama",
        "model": "rnj-1",
        "base_url": "http://localhost:11434",
        "api_key": None,
        "mode": "optional",
        "stages": ["rerank", "evidence", "rationale"],
        "models": {},
    },
    "classification": {"enabled": False, "model": None},
    "acquisition": {
        "order": ["cache", "zotero-storage", "zotero-api", "remote"],
    },
    "remote": {
        "email": None,
        "openalex_api_key": None,
        "playwright_profile": None,
        "playwright_headless": True,
    },
}


def load_settings(
    config_path: str | Path | None = None,
    profile: str = "default",
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Resolve settings using CLI overrides, environment, TOML, then defaults.

    ``bibliograph.toml`` is optional when ``config_path`` is omitted.  An
    explicitly supplied missing path is treated as a configuration error so a
    typo cannot silently select the defaults.  Profiles use the shape
    ``[profiles.<name>]``.
    """

    profile_settings = _read_profile(config_path, profile)
    normalized_overrides = _normalise_overrides(overrides or {})
    generic_environment = _generic_environment()

    # Legacy provider variables (OPENAI_*, OLLAMA_*, …) are selected based on
    # the final requested provider.  This lets a CLI provider override use the
    # matching credentials while retaining the documented precedence order.
    provider_context = _deep_merge(DEFAULT_SETTINGS, profile_settings)
    provider_context = _deep_merge(provider_context, _provider_only(generic_environment))
    provider_context = _deep_merge(provider_context, _provider_only(normalized_overrides))
    legacy_environment = _legacy_environment(provider_context)

    settings = _deep_merge(DEFAULT_SETTINGS, profile_settings)
    settings = _deep_merge(settings, legacy_environment)
    settings = _deep_merge(settings, generic_environment)
    settings = _deep_merge(settings, normalized_overrides)
    _synchronise_compatibility_aliases(settings)
    _validate(settings)
    return settings


def _read_profile(config_path: str | Path | None, profile: str) -> Settings:
    path = Path("bibliograph.toml") if config_path is None else Path(config_path)
    if not path.is_file():
        if config_path is None:
            return {}
        raise FileNotFoundError(f"Bibliograph configuration file was not found: {path}")

    with path.open("rb") as file:
        document = tomllib.load(file)
    profiles = document.get("profiles")
    if profiles is None:
        if profile != "default":
            raise ValueError(f"Configuration has no profile named {profile!r}")
        return _mapping_copy(document, label="configuration")
    if not isinstance(profiles, Mapping):
        raise ValueError("The TOML [profiles] section must be a table")
    selected = profiles.get(profile)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Configuration has no profile named {profile!r}")
    return _mapping_copy(selected, label=f"profile {profile!r}")


def _generic_environment() -> Settings:
    settings: Settings = {}
    entries: tuple[tuple[str, tuple[str, ...], Callable[[str], Any]], ...] = (
        ("BIBLIOGRAPH_DB", ("db",), str),
        ("BIBLIOGRAPH_PDF_DIR", ("pdf_dir",), str),
        ("BIBLIOGRAPH_COLLECTION", ("collection",), str),
        ("BIBLIOGRAPH_ZOTERO_STORAGE_DIR", ("zotero", "storage_dir"), str),
        ("BIBLIOGRAPH_ZOTERO_LIBRARY_ID", ("zotero", "library_id"), str),
        ("BIBLIOGRAPH_ZOTERO_LIBRARY_TYPE", ("zotero", "library_type"), str),
        ("BIBLIOGRAPH_ZOTERO_API_KEY", ("zotero", "api_key"), str),
        ("BIBLIOGRAPH_EMBEDDING_PROVIDER", ("embedding", "provider"), str),
        ("BIBLIOGRAPH_EMBEDDING_MODEL", ("embedding", "model"), str),
        ("BIBLIOGRAPH_EMBEDDING_BASE_URL", ("embedding", "base_url"), str),
        ("BIBLIOGRAPH_EMBEDDING_API_KEY", ("embedding", "api_key"), str),
        ("BIBLIOGRAPH_EMBEDDING_BATCH_SIZE", ("embedding", "batch_size"), _as_int),
        ("BIBLIOGRAPH_EMBEDDING_CACHE_DIR", ("embedding", "cache_dir"), str),
        (
            "BIBLIOGRAPH_EMBEDDING_OFFLINE",
            ("embedding", "local_files_only"),
            _as_bool,
        ),
        (
            "BIBLIOGRAPH_EMBEDDING_DOCUMENT_PREFIX",
            ("embedding", "document_prefix"),
            str,
        ),
        ("BIBLIOGRAPH_EMBEDDING_QUERY_PREFIX", ("embedding", "query_prefix"), str),
        ("BIBLIOGRAPH_LLM_PROVIDER", ("llm", "provider"), str),
        ("BIBLIOGRAPH_LLM_MODEL", ("llm", "model"), str),
        ("BIBLIOGRAPH_LLM_BASE_URL", ("llm", "base_url"), str),
        ("BIBLIOGRAPH_LLM_API_KEY", ("llm", "api_key"), str),
        ("BIBLIOGRAPH_LLM_MODE", ("llm", "mode"), str),
        ("BIBLIOGRAPH_LLM_STAGES", ("llm", "stages"), _as_csv),
        ("BIBLIOGRAPH_ACQUISITION_ORDER", ("acquisition", "order"), _as_csv),
        ("BIBLIOGRAPH_REMOTE_UNPAYWALL_EMAIL", ("remote", "email"), str),
        ("BIBLIOGRAPH_REMOTE_OPENALEX_API_KEY", ("remote", "openalex_api_key"), str),
        (
            "BIBLIOGRAPH_REMOTE_PLAYWRIGHT_PROFILE",
            ("remote", "playwright_profile"),
            str,
        ),
        (
            "BIBLIOGRAPH_REMOTE_PLAYWRIGHT_HEADLESS",
            ("remote", "playwright_headless"),
            _as_bool,
        ),
    )
    for variable, path, convert in entries:
        value = os.getenv(variable)
        if value is not None:
            _set_nested(settings, path, convert(value))
    return settings


def _legacy_environment(provider_context: Mapping[str, Any]) -> Settings:
    """Read compatibility environment variables without leaking them to adapters."""

    settings: Settings = {}
    for variable, key in (
        ("ZOTERO_STORAGE_DIR", "storage_dir"),
        ("ZOTERO_LIBRARY_ID", "library_id"),
        ("ZOTERO_LIBRARY_TYPE", "library_type"),
        ("ZOTERO_API_KEY", "api_key"),
    ):
        if value := os.getenv(variable):
            _set_nested(settings, ("zotero", key), value)
    if value := os.getenv("UNPAYWALL_EMAIL"):
        _set_nested(settings, ("remote", "email"), value)
    if value := os.getenv("OPENALEX_API_KEY"):
        _set_nested(settings, ("remote", "openalex_api_key"), value)
    if value := os.getenv("BIBLIOGRAPH_PLAYWRIGHT_PROFILE"):
        _set_nested(settings, ("remote", "playwright_profile"), value)
    if value := os.getenv("BIBLIOGRAPH_PLAYWRIGHT_HEADLESS"):
        _set_nested(settings, ("remote", "playwright_headless"), _as_bool(value))

    embedding = provider_context.get("embedding", {})
    if isinstance(embedding, Mapping):
        _apply_provider_environment(settings, "embedding", embedding, role="embedding")

    llm = provider_context.get("llm", {})
    if isinstance(llm, Mapping):
        _apply_provider_environment(settings, "llm", llm, role="llm")
    return settings


def _apply_provider_environment(
    target: Settings, section: str, provider_settings: Mapping[str, Any], *, role: str
) -> None:
    provider = str(provider_settings.get("provider", "")).casefold()
    if provider in {"openai", "openai-compatible"}:
        values = {
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_EMBEDDING_MODEL" if role == "embedding" else "LLM_MODEL"),
        }
    elif provider == "ollama":
        values = {
            "base_url": os.getenv("OLLAMA_HOST"),
            "model": os.getenv(
                "OLLAMA_EMBEDDING_MODEL" if role == "embedding" else "OLLAMA_LLM_MODEL"
            ),
        }
    elif provider == "sentence-transformers" and role == "embedding":
        values = {
            "model": os.getenv("EMBEDDING_MODEL"),
            "cache_dir": os.getenv("SENTENCE_TRANSFORMERS_CACHE") or os.getenv("HF_HOME"),
            "local_files_only": _optional_bool("SENTENCE_TRANSFORMERS_OFFLINE"),
        }
    else:
        return

    for key, value in values.items():
        if value is not None:
            _set_nested(target, (section, key), value)


def _normalise_overrides(overrides: Mapping[str, Any]) -> Settings:
    """Accept nested settings plus common flat argparse-style override names."""

    normalized = _mapping_copy(overrides, label="overrides")
    aliases = {
        "embedding_provider": ("embedding", "provider"),
        "embedding_model": ("embedding", "model"),
        "model": ("embedding", "model"),
        "embedding_base_url": ("embedding", "base_url"),
        "embedding_api_key": ("embedding", "api_key"),
        "embedding_batch_size": ("embedding", "batch_size"),
        "embedding_cache_dir": ("embedding", "cache_dir"),
        "embedding_offline": ("embedding", "local_files_only"),
        "llm_provider": ("llm", "provider"),
        "llm_model": ("llm", "model"),
        "llm_base_url": ("llm", "base_url"),
        "llm_api_key": ("llm", "api_key"),
        "llm_mode": ("llm", "mode"),
        "zotero_storage_dir": ("zotero", "storage_dir"),
        "zotero_library_id": ("zotero", "library_id"),
        "zotero_library_type": ("zotero", "library_type"),
        "zotero_api_key": ("zotero", "api_key"),
    }
    for flat_key, path in aliases.items():
        value = normalized.pop(flat_key, None)
        if value is not None:
            _set_nested(normalized, path, value)
    return normalized


def _provider_only(settings: Mapping[str, Any]) -> Settings:
    result: Settings = {}
    for section in ("embedding", "llm"):
        value = settings.get(section)
        if isinstance(value, Mapping) and value.get("provider") is not None:
            result[section] = {"provider": value["provider"]}
    return result


def _synchronise_compatibility_aliases(settings: Settings) -> None:
    """Keep transitional flat/legacy spellings usable during the CLI migration."""

    zotero = settings.get("zotero")
    if isinstance(zotero, dict):
        storage_dir = zotero.get("storage_dir")
        flat_storage_dir = settings.get("zotero_storage_dir")
        if storage_dir is None and flat_storage_dir is not None:
            zotero["storage_dir"] = flat_storage_dir
        elif storage_dir is not None:
            settings["zotero_storage_dir"] = storage_dir

    remote = settings.get("remote")
    if isinstance(remote, dict) and remote.get("email") is None:
        legacy_email = remote.get("unpaywall_email")
        if legacy_email is not None:
            remote["email"] = legacy_email


def _validate(settings: Mapping[str, Any]) -> None:
    embedding = settings.get("embedding")
    llm = settings.get("llm")
    acquisition = settings.get("acquisition")
    remote = settings.get("remote")
    classification = settings.get("classification")
    if not isinstance(embedding, Mapping) or not isinstance(llm, Mapping):
        raise ValueError("embedding and llm settings must be tables")
    _validate_provider(embedding, section="embedding", choices=embedding_names())
    _validate_provider(llm, section="llm", choices=chat_names())
    batch_size = embedding.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("embedding.batch_size must be a positive integer")
    mode = llm.get("mode")
    if mode not in {"off", "optional", "required"}:
        raise ValueError("llm.mode must be one of: off, optional, required")
    stages = llm.get("stages")
    if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
        raise ValueError("llm.stages must be a list of strings")
    unknown_stages = sorted(set(stages) - _LLM_STAGES)
    if unknown_stages:
        raise ValueError(
            "llm.stages contains unknown stage(s): "
            f"{', '.join(unknown_stages)}; choose from {', '.join(sorted(_LLM_STAGES))}"
        )
    models = llm.get("models", {})
    if not isinstance(models, Mapping):
        raise ValueError("llm.models must be a table mapping stages to model names")
    unknown_model_stages = sorted(set(models) - _LLM_STAGES)
    if unknown_model_stages:
        raise ValueError(
            "llm.models contains unknown stage(s): "
            f"{', '.join(unknown_model_stages)}; choose from {', '.join(sorted(_LLM_STAGES))}"
        )
    if any(not isinstance(model, str) or not model.strip() for model in models.values()):
        raise ValueError("llm.models values must be non-empty model names")
    if (
        not isinstance(acquisition, Mapping)
        or not isinstance(acquisition.get("order"), list)
        or not all(isinstance(name, str) for name in acquisition["order"])
    ):
        raise ValueError("acquisition.order must be a list")
    unknown_strategies = sorted(set(acquisition["order"]) - _ACQUISITION_STRATEGIES)
    if unknown_strategies:
        raise ValueError(
            "acquisition.order contains unknown strategy/strategies: "
            f"{', '.join(unknown_strategies)}; choose from "
            f"{', '.join(sorted(_ACQUISITION_STRATEGIES))}"
        )
    if not isinstance(classification, Mapping):
        raise ValueError("classification settings must be a table")
    if not isinstance(classification.get("enabled"), bool):
        raise ValueError("classification.enabled must be a boolean")
    model = classification.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("classification.model must be a non-empty model name")
    if classification["enabled"] and mode == "off":
        raise ValueError("classification.enabled requires llm.mode to be optional or required")
    if not isinstance(remote, Mapping):
        raise ValueError("remote settings must be a table")
    if not isinstance(remote.get("playwright_headless"), bool):
        raise ValueError("remote.playwright_headless must be a boolean")


def _validate_provider(
    config: Mapping[str, Any], *, section: str, choices: tuple[str, ...]
) -> None:
    provider = config.get("provider")
    kind = "embedding" if section == "embedding" else "chat"
    if resolve_name(provider, kind=kind) is None:
        raise ValueError(f"{section}.provider must be one of: {', '.join(choices)}")


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Settings:
    merged = _mapping_copy(base, label="settings")
    for key, value in overlay.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(previous, value)
        else:
            merged[key] = _copy_value(value)
    return merged


def _mapping_copy(mapping: Mapping[str, Any], *, label: str) -> Settings:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return {str(key): _copy_value(value) for key, value in mapping.items()}


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return deepcopy(value)
    return value


def _set_nested(target: Settings, path: tuple[str, ...], value: Any) -> None:
    current: Settings = target
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[path[-1]] = value


def _as_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean environment value, got {value!r}")


def _optional_bool(variable: str) -> bool | None:
    value = os.getenv(variable)
    return _as_bool(value) if value is not None else None


def _as_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Expected an integer environment value, got {value!r}") from error


def _as_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
