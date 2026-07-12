import hashlib
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .config import (
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
)


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic local fallback for tests and development without a model."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.identity = f"hash:{dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        from sentence_transformers import SentenceTransformer

        self.cache_dir = cache_dir or _sentence_transformer_cache_dir()
        self.local_files_only = local_files_only
        self.model_name = model_name
        self.identity = f"sentence-transformers:{model_name}"
        self.model = SentenceTransformer(
            model_name,
            cache_folder=self.cache_dir,
            local_files_only=local_files_only,
        )
        self.dimension = self.model.get_sentence_embedding_dimension()

    @classmethod
    def from_environment(
        cls,
        model_name: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
    ):
        if local_files_only is None:
            local_files_only = _env_bool("SENTENCE_TRANSFORMERS_OFFLINE") or _env_bool(
                "HF_HUB_OFFLINE"
            )
        return cls(
            model_name or os.getenv("EMBEDDING_MODEL") or DEFAULT_SENTENCE_TRANSFORMER_MODEL,
            cache_dir,
            local_files_only,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()


def _sentence_transformer_cache_dir() -> str:
    return (
        os.getenv("SENTENCE_TRANSFORMERS_CACHE")
        or os.getenv("HF_HOME")
        or str(Path.home() / ".cache" / "huggingface")
    )


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class OpenAICompatibleEmbedder:
    """Embed texts through any OpenAI-compatible ``/embeddings`` endpoint.

    The client is injectable so callers can test this adapter without network access.
    ``base_url`` may point to OpenAI, vLLM, LM Studio, Ollama, or another compatible API.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        client=None,
    ):
        if not model:
            raise ValueError("An embedding model is required")
        if not api_key and client is None:
            raise ValueError("An API key is required for an OpenAI-compatible embedder")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI-compatible embeddings require the optional 'embeddings' dependencies"
                ) from exc
            client = OpenAI(
                api_key=api_key or DEFAULT_OPENAI_API_KEY,
                base_url=base_url or DEFAULT_OPENAI_BASE_URL,
            )
        self.client = client
        self.model = model
        self.base_url = base_url or DEFAULT_OPENAI_BASE_URL
        self.identity = f"openai-compatible:{self.base_url}:{model}"
        self.dimension = 0

    @classmethod
    def from_environment(cls, model: str | None = None, base_url: str | None = None):
        return cls(
            model or os.getenv("OPENAI_EMBEDDING_MODEL") or DEFAULT_OPENAI_EMBEDDING_MODEL,
            os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY,
            base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        data = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in data]
        if len(vectors) != len(texts):
            raise ValueError("Embedding service returned a different number of vectors")
        self.dimension = len(vectors[0])
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("Embedding service returned inconsistent vector dimensions")
        return vectors


class OllamaEmbedder(OpenAICompatibleEmbedder):
    """Embed text through Ollama's native ``/api/embed`` endpoint."""

    def __init__(self, model: str, base_url: str | None = None, client=None):
        if client is None:
            from .ollama import OllamaClient

            client = OllamaClient(base_url)
        super().__init__(model, "ollama", base_url, client)
        self.base_url = client.base_url
        self.identity = f"ollama:{self.base_url}:{model}"

    @classmethod
    def from_environment(cls, model: str | None = None, base_url: str | None = None):
        return cls(
            model or os.getenv("OLLAMA_EMBEDDING_MODEL") or DEFAULT_OLLAMA_EMBEDDING_MODEL, base_url
        )
