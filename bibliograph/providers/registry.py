"""Explicit provider registry and construction helpers.

Adding a provider is intentionally small: implement its closure factory and
add one entry here.  Commands and pipeline functions never branch on provider
names after this composition boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ._shared import ChatRuntime, EmbeddingRuntime, ProviderError
from .local import hash_embedding, sentence_transformers_embedding
from .ollama import ollama_chat, ollama_embedding
from .openai import openai_chat, openai_embedding

EmbeddingFactory = Callable[[Mapping[str, Any]], EmbeddingRuntime]
ChatFactory = Callable[[Mapping[str, Any]], ChatRuntime]


EMBEDDING_FACTORIES: dict[str, EmbeddingFactory] = {
    "hash": hash_embedding,
    "sentence-transformers": sentence_transformers_embedding,
    "openai": openai_embedding,
    "ollama": ollama_embedding,
}

CHAT_FACTORIES: dict[str, ChatFactory] = {
    "openai": openai_chat,
    "ollama": ollama_chat,
}

_ALIASES = {
    "openai-compatible": "openai",
    "sentence_transformers": "sentence-transformers",
}


def names(kind: str | None = None) -> tuple[str, ...]:
    """Return valid registered names for one capability or both capabilities."""

    if kind is None:
        return tuple(sorted(set(EMBEDDING_FACTORIES) | set(CHAT_FACTORIES)))
    if kind == "embedding":
        return tuple(sorted(EMBEDDING_FACTORIES))
    if kind in {"chat", "llm"}:
        return tuple(sorted(CHAT_FACTORIES))
    raise ValueError("kind must be 'embedding', 'chat', 'llm', or None")


def embedding_names() -> tuple[str, ...]:
    return names("embedding")


def chat_names() -> tuple[str, ...]:
    return names("chat")


def resolve_name(name: object, *, kind: str) -> str | None:
    """Return a registered canonical provider name, or ``None``.

    Settings resolution uses this small registry-facing check so TOML and
    environment configuration receive the same validation as CLI selection.
    Aliases remain accepted during the compatibility window, while factories
    continue to own construction and diagnostics.
    """

    if not isinstance(name, str) or not name.strip():
        return None
    canonical = _ALIASES.get(name.strip().casefold(), name.strip().casefold())
    if kind == "embedding":
        return canonical if canonical in EMBEDDING_FACTORIES else None
    if kind in {"chat", "llm"}:
        return canonical if canonical in CHAT_FACTORIES else None
    raise ValueError("kind must be 'embedding', 'chat', or 'llm'")


def build_embedding(config: Mapping[str, Any]) -> EmbeddingRuntime:
    """Select and build an embedding runtime from a config section or settings map."""

    section = _section(config, "embedding")
    provider = _provider_name(section, capability="embedding")
    factory = EMBEDDING_FACTORIES.get(provider)
    if factory is None:
        raise ProviderError(
            f"unknown provider; choose one of: {', '.join(embedding_names())}",
            provider=provider,
            operation="embedding initialization",
        )
    return factory(section)


def build_chat(config: Mapping[str, Any]) -> ChatRuntime:
    """Select and build a chat runtime from an LLM section or settings map."""

    section = _section(config, "llm")
    provider = _provider_name(section, capability="LLM")
    factory = CHAT_FACTORIES.get(provider)
    if factory is None:
        raise ProviderError(
            f"unknown provider; choose one of: {', '.join(chat_names())}",
            provider=provider,
            operation="LLM initialization",
        )
    return factory(section)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("Provider configuration must be a mapping")
    if "provider" in config:
        return config
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise TypeError(f"Provider configuration needs a [{name}] mapping")
    return section


def _provider_name(config: Mapping[str, Any], *, capability: str) -> str:
    value = config.get("provider")
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            "a provider name is required",
            provider="unknown",
            operation=f"{capability} initialization",
        )
    return _ALIASES.get(value.strip().casefold(), value.strip().casefold())
