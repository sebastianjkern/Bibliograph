"""Local embedding provider factories with no environment coupling."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ._shared import (
    EmbeddingRuntime,
    ProviderError,
    apply_prefix,
    checked_vectors,
    embedding_identity,
    log_failure,
    option,
    report,
)


def hash_embedding(config: Mapping[str, Any]) -> EmbeddingRuntime:
    """Build a deterministic hashing embedder for offline development and tests."""

    dimension = option(config, "dimension", 256)
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ProviderError(
            "dimension must be a positive integer",
            provider="hash",
            operation="embedding initialization",
        )
    document_prefix = str(option(config, "document_prefix", ""))
    query_prefix = str(option(config, "query_prefix", ""))
    identity = embedding_identity(f"hash:{dimension}", document_prefix, query_prefix)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * dimension
            for token in str(text).lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vector[int.from_bytes(digest[:4], "big") % dimension] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors

    def embed_documents(texts: Sequence[str]) -> list[list[float]]:
        return embed(apply_prefix(texts, document_prefix))

    def embed_queries(texts: Sequence[str]) -> list[list[float]]:
        return embed(apply_prefix(texts, query_prefix))

    def probe() -> dict[str, object]:
        vectors = embed_documents(["Bibliograph provider probe"])
        return report(
            provider="hash",
            model=str(dimension),
            identity=identity,
            dimension=len(vectors[0]),
        )

    return {
        "id": identity,
        "kinds": ("text",),
        "embed_documents": embed_documents,
        "embed_queries": embed_queries,
        "probe": probe,
    }


def sentence_transformers_embedding(config: Mapping[str, Any]) -> EmbeddingRuntime:
    """Build a SentenceTransformers embedding runtime around one loaded model."""

    model_name = str(option(config, "model", "all-MiniLM-L6-v2"))
    cache_dir = option(config, "cache_dir")
    local_files_only = bool(option(config, "local_files_only", option(config, "offline", False)))
    document_prefix = str(option(config, "document_prefix", ""))
    query_prefix = str(option(config, "query_prefix", ""))
    identity = embedding_identity(
        f"sentence-transformers:{model_name}", document_prefix, query_prefix
    )
    model = option(config, "model_instance")
    if model is None:
        factory = option(config, "model_factory")
        try:
            if factory is None:
                from sentence_transformers import SentenceTransformer

                factory = SentenceTransformer
            model = factory(
                model_name,
                cache_folder=cache_dir,
                local_files_only=local_files_only,
            )
        except ProviderError:
            raise
        except Exception as error:
            raise log_failure(
                ProviderError(
                    str(error),
                    provider="sentence-transformers",
                    operation="embedding initialization",
                    model=model_name,
                )
            ) from error

    def embed(texts: Sequence[str], prefix: str) -> list[list[float]]:
        values = apply_prefix(texts, prefix)
        if not values:
            return []
        try:
            encoded = model.encode(values, normalize_embeddings=True)
            vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
            return checked_vectors(
                vectors,
                expected_count=len(values),
                provider="sentence-transformers",
                endpoint=None,
                model=model_name,
            )
        except ProviderError:
            raise
        except Exception as error:
            raise log_failure(
                ProviderError(
                    str(error),
                    provider="sentence-transformers",
                    operation="embedding",
                    model=model_name,
                )
            ) from error

    def probe() -> dict[str, object]:
        vectors = embed(["Bibliograph provider probe"], document_prefix)
        dimension = len(vectors[0]) if vectors else None
        values: dict[str, object] = {"local_files_only": local_files_only}
        if dimension is not None:
            values["dimension"] = dimension
        return report(
            provider="sentence-transformers",
            model=model_name,
            identity=identity,
            **values,
        )

    return {
        "id": identity,
        "kinds": ("text",),
        "embed_documents": lambda texts: embed(texts, document_prefix),
        "embed_queries": lambda texts: embed(texts, query_prefix),
        "probe": probe,
    }
