"""Internal helpers shared by concrete provider adapters."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from core.embedding import BatchEmbeddingStrategy


class EmbeddingRuntime(TypedDict):
    """The callable capabilities required by the indexing and retrieval paths."""

    id: str
    kinds: tuple[str, ...]
    embed_documents: BatchEmbeddingStrategy
    embed_queries: BatchEmbeddingStrategy
    probe: Callable[[], dict[str, object]]


class ChatRuntime(TypedDict):
    """The callable capability required by provider-independent LLM tasks."""

    id: str
    complete: Callable[..., str]
    probe: Callable[[], dict[str, object]]


logger = logging.getLogger("bibliograph.providers")


class ProviderError(RuntimeError):
    """A provider failure with enough context to give a useful CLI diagnostic."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        operation: str,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.operation = operation
        self.endpoint = endpoint
        self.model = model
        context = f"{provider} {operation} provider"
        if model:
            context += f" (model {model})"
        if endpoint:
            context += f" at {endpoint}"
        super().__init__(f"{context} failed: {message}")


def option(config: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Read an adapter option without letting a false-y value select a default."""

    return config[name] if name in config else default


def response_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def normalized_url(value: Any, default: str) -> str:
    raw = str(value or default).strip()
    if not raw:
        return default.rstrip("/")
    return raw.rstrip("/")


def apply_prefix(texts: Sequence[str], prefix: str) -> list[str]:
    values = [str(text) for text in texts]
    return [f"{prefix}{text}" for text in values] if prefix else values


def embedding_identity(base: str, document_prefix: str, query_prefix: str) -> str:
    """Include transformations in the embedding-space fingerprint when needed."""

    if not document_prefix and not query_prefix:
        return base
    document = json.dumps(document_prefix, ensure_ascii=False, separators=(",", ":"))
    query = json.dumps(query_prefix, ensure_ascii=False, separators=(",", ":"))
    return f"{base}:document-prefix={document}:query-prefix={query}"


def checked_vectors(
    vectors: Sequence[Any],
    *,
    expected_count: int,
    provider: str,
    endpoint: str | None,
    model: str,
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise ProviderError(
            f"returned {len(vectors)} vectors for {expected_count} inputs",
            provider=provider,
            operation="embedding",
            endpoint=endpoint,
            model=model,
        )
    normalized: list[list[float]] = []
    for vector in vectors:
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as error:
            raise ProviderError(
                "returned a non-numeric embedding",
                provider=provider,
                operation="embedding",
                endpoint=endpoint,
                model=model,
            ) from error
        if not values:
            raise ProviderError(
                "returned an empty embedding",
                provider=provider,
                operation="embedding",
                endpoint=endpoint,
                model=model,
            )
        normalized.append(values)
    dimensions = {len(vector) for vector in normalized}
    if len(dimensions) > 1:
        raise ProviderError(
            "returned embeddings with inconsistent dimensions",
            provider=provider,
            operation="embedding",
            endpoint=endpoint,
            model=model,
        )
    return normalized


def report(
    *, provider: str,
    model: str,
    identity: str,
    endpoint: str | None = None,
    **values: Any,
) -> dict[str, object]:
    result: dict[str, object] = {
        "ok": True,
        "provider": provider,
        "model": model,
        "id": identity,
    }
    if endpoint:
        result["endpoint"] = endpoint
    result.update(values)
    return result


def log_failure(error: ProviderError) -> ProviderError:
    logger.error("%s", error)
    return error
