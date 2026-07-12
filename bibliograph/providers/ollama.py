"""Native Ollama embedding and chat factories without SDK-shaped emulation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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

DEFAULT_BASE_URL = "http://localhost:11434"


def ollama_embedding(config: Mapping[str, Any]) -> EmbeddingRuntime:
    """Build an embedding runtime using Ollama's native ``/api/embed`` API."""

    model = _required_model(config, role="embedding")
    base_url = normalized_url(option(config, "base_url"), DEFAULT_BASE_URL)
    timeout = _timeout(config, model=model, base_url=base_url, operation="embedding")
    document_prefix = str(option(config, "document_prefix", ""))
    query_prefix = str(option(config, "query_prefix", ""))
    identity = embedding_identity(f"ollama:{base_url}:{model}", document_prefix, query_prefix)
    transport = option(config, "transport", option(config, "post"))

    def embed(texts: Sequence[str], prefix: str) -> list[list[float]]:
        values = apply_prefix(texts, prefix)
        if not values:
            return []
        payload = {"model": model, "input": values}
        try:
            response = _call(
                transport,
                base_url=base_url,
                path="/api/embed",
                payload=payload,
                timeout=timeout,
            )
            raw_vectors = response_value(response, "embeddings")
            if not isinstance(raw_vectors, Sequence) or isinstance(
                raw_vectors, (str, bytes, bytearray)
            ):
                raise ProviderError(
                    "returned no embeddings",
                    provider="ollama",
                    operation="embedding",
                    endpoint=base_url,
                    model=model,
                )
            return checked_vectors(
                raw_vectors,
                expected_count=len(values),
                provider="ollama",
                endpoint=base_url,
                model=model,
            )
        except ProviderError as error:
            log_failure(error)
            raise
        except Exception as error:
            raise log_failure(
                ProviderError(
                    str(error),
                    provider="ollama",
                    operation="embedding",
                    endpoint=base_url,
                    model=model,
                )
            ) from error

    def probe() -> dict[str, object]:
        vectors = embed(["Bibliograph provider probe"], document_prefix)
        return report(
            provider="ollama",
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


def ollama_chat(config: Mapping[str, Any]) -> ChatRuntime:
    """Build a chat runtime using Ollama's native ``/api/chat`` API."""

    model = _required_model(config, role="LLM")
    base_url = normalized_url(option(config, "base_url"), DEFAULT_BASE_URL)
    timeout = _timeout(config, model=model, base_url=base_url, operation="LLM")
    identity = f"ollama:{base_url}:{model}"
    transport = option(config, "transport", option(config, "post"))

    def complete(messages: Sequence[Mapping[str, object]], *, json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = _call(
                transport,
                base_url=base_url,
                path="/api/chat",
                payload=payload,
                timeout=timeout,
            )
            message = response_value(response, "message")
            content = response_value(message, "content", response_value(response, "content"))
            if not isinstance(content, str):
                raise ProviderError(
                    "returned no text chat completion",
                    provider="ollama",
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
                    provider="ollama",
                    operation="LLM",
                    endpoint=base_url,
                    model=model,
                )
            ) from error

    def probe() -> dict[str, object]:
        complete([{"role": "user", "content": "Reply with OK."}], json_mode=False)
        return report(
            provider="ollama",
            model=model,
            identity=identity,
            endpoint=base_url,
        )

    return {"id": identity, "complete": complete, "probe": probe}


def _call(
    transport: Any,
    *,
    base_url: str,
    path: str,
    payload: Mapping[str, object],
    timeout: float,
) -> Any:
    if callable(transport):
        return transport(path, dict(payload))
    return _native_post(base_url, path, payload, timeout)


def _native_post(
    base_url: str, path: str, payload: Mapping[str, object], timeout: float
) -> dict[str, object]:
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _request_json(request, timeout)


def _request_json(request: Request, timeout: float) -> dict[str, object]:
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except (URLError, OSError, ValueError) as error:
        reason = getattr(error, "reason", error)
        raise RuntimeError(f"Cannot connect to Ollama: {reason}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Ollama returned a non-object JSON response")
    if payload.get("error"):
        raise RuntimeError(f"Ollama error: {payload['error']}")
    return payload


def _required_model(config: Mapping[str, Any], *, role: str) -> str:
    value = option(config, "model")
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            "a model is required",
            provider="ollama",
            operation=f"{role} initialization",
        )
    return value.strip()


def _timeout(config: Mapping[str, Any], *, model: str, base_url: str, operation: str) -> float:
    value = option(config, "timeout", 120.0)
    try:
        timeout = float(value)
    except (TypeError, ValueError) as error:
        raise ProviderError(
            "timeout must be numeric",
            provider="ollama",
            operation=f"{operation} initialization",
            endpoint=base_url,
            model=model,
        ) from error
    if timeout <= 0:
        raise ProviderError(
            "timeout must be positive",
            provider="ollama",
            operation=f"{operation} initialization",
            endpoint=base_url,
            model=model,
        )
    return timeout


embedding_runtime = ollama_embedding
chat_runtime = ollama_chat
