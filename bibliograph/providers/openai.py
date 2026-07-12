"""OpenAI-compatible provider factories.

The factories intentionally only consume a configuration mapping.  A caller
may inject either an OpenAI SDK-shaped ``client`` or a small ``transport``
callable, which keeps tests and alternative compatible servers lightweight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._shared import (
    ChatRuntime,
    EmbeddingRuntime,
    ProviderError,
    apply_prefix,
    checked_vectors,
    embedding_identity,
    log_failure,
    normalized_url,
    option,
    report,
    response_value,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def openai_embedding(config: Mapping[str, Any]) -> EmbeddingRuntime:
    """Build an OpenAI-compatible document/query embedding runtime."""

    model = _required_model(config, role="embedding")
    base_url = normalized_url(option(config, "base_url"), DEFAULT_BASE_URL)
    document_prefix = str(option(config, "document_prefix", ""))
    query_prefix = str(option(config, "query_prefix", ""))
    identity = embedding_identity(
        f"openai-compatible:{base_url}:{model}", document_prefix, query_prefix
    )
    client = _client(config, model=model, base_url=base_url, operation="embedding")
    transport = option(config, "transport")

    def embed(texts: Sequence[str], prefix: str) -> list[list[float]]:
        values = apply_prefix(texts, prefix)
        if not values:
            return []
        payload = {"model": model, "input": values}
        try:
            response = (
                transport("embeddings", payload)
                if callable(transport)
                else client.embeddings.create(**payload)
            )
            return _parse_embedding_response(
                response,
                expected_count=len(values),
                provider="openai",
                base_url=base_url,
                model=model,
            )
        except ProviderError as error:
            log_failure(error)
            raise
        except Exception as error:
            raise log_failure(
                ProviderError(
                    str(error),
                    provider="openai",
                    operation="embedding",
                    endpoint=base_url,
                    model=model,
                )
            ) from error

    def probe() -> dict[str, object]:
        vectors = embed(["Bibliograph provider probe"], document_prefix)
        return report(
            provider="openai",
            model=model,
            identity=identity,
            endpoint=base_url,
            dimension=len(vectors[0]),
        )

    return {
        "id": identity,
        "kinds": ("text",),
        "embed_documents": lambda texts: embed(texts, document_prefix),
        "embed_queries": lambda texts: embed(texts, query_prefix),
        "probe": probe,
    }


def openai_chat(config: Mapping[str, Any]) -> ChatRuntime:
    """Build an OpenAI-compatible chat-completion runtime."""

    model = _required_model(config, role="LLM")
    base_url = normalized_url(option(config, "base_url"), DEFAULT_BASE_URL)
    identity = f"openai-compatible:{base_url}:{model}"
    client = _client(config, model=model, base_url=base_url, operation="LLM")
    transport = option(config, "transport")

    def complete(messages: Sequence[Mapping[str, object]], *, json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": model,
            "messages": [dict(message) for message in messages],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = (
                transport("chat", payload)
                if callable(transport)
                else client.chat.completions.create(**payload)
            )
            content = _parse_chat_response(response)
            if not isinstance(content, str):
                raise ProviderError(
                    "returned a non-text chat completion",
                    provider="openai",
                    operation="LLM",
                    endpoint=base_url,
                    model=model,
                )
            return content
        except ProviderError as error:
            log_failure(error)
            raise
        except Exception as error:
            raise log_failure(
                ProviderError(
                    str(error),
                    provider="openai",
                    operation="LLM",
                    endpoint=base_url,
                    model=model,
                )
            ) from error

    def probe() -> dict[str, object]:
        complete([{"role": "user", "content": "Reply with OK."}], json_mode=False)
        return report(
            provider="openai",
            model=model,
            identity=identity,
            endpoint=base_url,
        )

    return {"id": identity, "complete": complete, "probe": probe}


def _client(
    config: Mapping[str, Any], *, model: str, base_url: str, operation: str
) -> Any | None:
    client = option(config, "client")
    if client is not None or callable(option(config, "transport")):
        return client
    api_key = option(config, "api_key")
    if not api_key:
        raise ProviderError(
            "an api_key is required when no client or transport is injected",
            provider="openai",
            operation=f"{operation} initialization",
            endpoint=base_url,
            model=model,
        )
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key, base_url=base_url)
    except ProviderError:
        raise
    except Exception as error:
        raise log_failure(
            ProviderError(
                str(error),
                provider="openai",
                operation=f"{operation} initialization",
                endpoint=base_url,
                model=model,
            )
        ) from error


def _required_model(config: Mapping[str, Any], *, role: str) -> str:
    value = option(config, "model")
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            "a model is required",
            provider="openai",
            operation=f"{role} initialization",
        )
    return value.strip()


def _parse_embedding_response(
    response: Any,
    *,
    expected_count: int,
    provider: str,
    base_url: str,
    model: str,
) -> list[list[float]]:
    if isinstance(response, Sequence) and not isinstance(response, (str, bytes, bytearray)):
        raw_vectors = response
    else:
        data = response_value(response, "data")
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
            raise ProviderError(
                "returned no embedding data",
                provider=provider,
                operation="embedding",
                endpoint=base_url,
                model=model,
            )
        ordered = sorted(
            enumerate(data),
            key=lambda item: int(response_value(item[1], "index", item[0])),
        )
        raw_vectors = [response_value(item, "embedding") for _index, item in ordered]
    return checked_vectors(
        raw_vectors,
        expected_count=expected_count,
        provider=provider,
        endpoint=base_url,
        model=model,
    )


def _parse_chat_response(response: Any) -> Any:
    if isinstance(response, str):
        return response
    choices = response_value(response, "choices")
    if isinstance(choices, Sequence) and choices:
        message = response_value(choices[0], "message")
        return response_value(message, "content")
    return response_value(response, "content")


# Friendly names for direct adapter use without coupling callers to the registry.
embedding_runtime = openai_embedding
chat_runtime = openai_chat
