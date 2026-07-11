import hashlib
import math
import os
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic local fallback for tests and development without a model."""

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()


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
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.dimension = 0

    @classmethod
    def from_environment(cls, model: str | None = None, base_url: str | None = None):
        return cls(
            model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            os.getenv("OPENAI_API_KEY", ""),
            base_url or os.getenv("OPENAI_BASE_URL"),
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
