"""Provider runtimes and the explicit registry used by Bibliograph."""

from ._shared import ChatRuntime, EmbeddingRuntime, ProviderError
from .registry import (
    CHAT_FACTORIES,
    EMBEDDING_FACTORIES,
    build_chat,
    build_embedding,
    chat_names,
    embedding_names,
    names,
    resolve_name,
)

__all__ = [
    "CHAT_FACTORIES",
    "EMBEDDING_FACTORIES",
    "ChatRuntime",
    "EmbeddingRuntime",
    "ProviderError",
    "build_chat",
    "build_embedding",
    "chat_names",
    "embedding_names",
    "names",
    "resolve_name",
]
