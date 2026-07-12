from dataclasses import FrozenInstanceError

import pytest

from bibliograph.domain import Chunk, Paper, citation_label
from bibliograph.providers import (
    ProviderError,
    build_chat,
    build_embedding,
    chat_names,
    embedding_names,
    names,
)


def test_domain_records_are_frozen_slotted_and_transient_shapes_are_simple():
    paper = Paper("P1", "A paper", ("Ng",), "2025")
    chunk = Chunk("P1:0", paper, "Evidence", ordinal=2)

    assert not hasattr(paper, "__dict__")
    assert not hasattr(chunk, "__dict__")
    assert chunk.chunk_index == 2
    assert citation_label(paper) == "Ng (2025)"
    with pytest.raises(FrozenInstanceError):
        paper.title = "Changed"  # type: ignore[misc]


def test_registry_exposes_capability_specific_factory_names():
    assert set(embedding_names()) == {"hash", "ollama", "openai", "sentence-transformers"}
    assert set(chat_names()) == {"ollama", "openai"}
    assert set(names()) == set(embedding_names())


def test_hash_runtime_keeps_document_and_query_encoders_separate():
    runtime = build_embedding(
        {
            "provider": "hash",
            "dimension": 32,
            "document_prefix": "document: ",
            "query_prefix": "query: ",
        }
    )

    document_vector = runtime["embed_documents"](["same text"])
    query_vector = runtime["embed_queries"](["same text"])

    assert document_vector != query_vector
    assert runtime["kinds"] == ("text",)
    assert runtime["probe"]()["dimension"] == 32
    assert "document-prefix" in runtime["id"]


def test_sentence_transformers_runtime_accepts_an_injected_model():
    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, *, normalize_embeddings):
            self.calls.append((texts, normalize_embeddings))
            return [[float(index), 1.0, 0.0] for index, _text in enumerate(texts)]

        def get_sentence_embedding_dimension(self):
            return 3

    model = FakeModel()
    runtime = build_embedding(
        {"provider": "sentence-transformers", "model": "fake", "model_instance": model}
    )

    assert runtime["embed_documents"](["one", "two"]) == [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert model.calls == [(["one", "two"], True)]
    assert runtime["probe"]()["dimension"] == 3
    assert model.calls[-1] == (["Bibliograph provider probe"], True)


def test_openai_compatible_factories_accept_an_injected_sdk_shaped_client():
    captured = {}

    class Embeddings:
        def create(self, **payload):
            captured["embedding"] = payload
            return {
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            }

    class Completions:
        def create(self, **payload):
            captured["chat"] = payload
            return {"choices": [{"message": {"content": "answer"}}]}

    class Client:
        embeddings = Embeddings()

        class chat:
            completions = Completions()

    embedding = build_embedding(
        {
            "provider": "openai",
            "model": "embed-model",
            "base_url": "http://server.test/v1",
            "client": Client(),
        }
    )
    chat = build_chat(
        {
            "provider": "openai-compatible",
            "model": "chat-model",
            "base_url": "http://server.test/v1",
            "client": Client(),
        }
    )

    assert embedding["embed_documents"](["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["embedding"] == {"model": "embed-model", "input": ["one", "two"]}
    assert chat["complete"]([{"role": "user", "content": "Hi"}], json_mode=True) == "answer"
    assert captured["chat"] == {
        "model": "chat-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    }


def test_openai_chat_sends_strict_response_schema_when_requested():
    captured = {}

    class Completions:
        def create(self, **payload):
            captured["chat"] = payload
            return {"choices": [{"message": {"content": "{}"}}]}

    class Client:
        class chat:
            completions = Completions()

    chat = build_chat(
        {
            "provider": "openai-compatible",
            "model": "chat-model",
            "client": Client(),
        }
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    chat["complete"](
        [{"role": "user", "content": "Return structured data."}],
        json_mode=True,
        response_schema=schema,
    )

    assert captured["chat"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "bibliograph_structured_response",
            "strict": True,
            "schema": schema,
        },
    }


def test_ollama_factories_use_native_paths_and_contextual_errors():
    calls = []

    def transport(path, payload):
        calls.append((path, payload))
        if path == "/api/embed":
            return {"embeddings": [[1, 0], [0, 1]]}
        if path == "/api/chat":
            return {"message": {"content": "native answer"}}
        return {"models": []}

    embedding = build_embedding(
        {"provider": "ollama", "model": "nomic", "transport": transport}
    )
    chat = build_chat({"provider": "ollama", "model": "llama", "transport": transport})

    assert embedding["embed_queries"](["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert (
        chat["complete"]([{"role": "user", "content": "hello"}], json_mode=True)
        == "native answer"
    )
    assert calls[:2] == [
        ("/api/embed", {"model": "nomic", "input": ["one", "two"]}),
        (
            "/api/chat",
            {
                "model": "llama",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "format": "json",
            },
        ),
    ]

    def unavailable(_path, _payload):
        raise OSError("connection refused")

    failing = build_embedding(
        {
            "provider": "ollama",
            "model": "nomic",
            "base_url": "http://ollama.test",
            "transport": unavailable,
        }
    )
    with pytest.raises(ProviderError, match="ollama embedding provider") as error:
        failing["probe"]()
    assert "http://ollama.test" in str(error.value)
    assert "connection refused" in str(error.value)


def test_ollama_chat_sends_schema_format_when_requested():
    calls = []

    def transport(path, payload):
        calls.append((path, payload))
        return {"message": {"content": "{}"}}

    chat = build_chat(
        {"provider": "ollama", "model": "chat-model", "transport": transport}
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    chat["complete"](
        [{"role": "user", "content": "Return structured data."}],
        json_mode=True,
        response_schema=schema,
    )

    assert calls == [
        (
            "/api/chat",
            {
                "model": "chat-model",
                "messages": [{"role": "user", "content": "Return structured data."}],
                "stream": False,
                "format": schema,
            },
        )
    ]


def test_ollama_probes_use_the_configured_embedding_and_chat_models():
    calls = []

    def transport(path, payload):
        calls.append((path, payload))
        if path == "/api/embed":
            return {"embeddings": [[1, 0]]}
        if path == "/api/chat":
            return {"message": {"content": "OK"}}
        raise AssertionError(f"unexpected Ollama request: {path}")

    embedding = build_embedding(
        {"provider": "ollama", "model": "embed-model", "transport": transport}
    )
    chat = build_chat({"provider": "ollama", "model": "chat-model", "transport": transport})

    assert embedding["probe"]()["dimension"] == 2
    assert chat["probe"]()["ok"] is True
    assert calls == [
        (
            "/api/embed",
            {"model": "embed-model", "input": ["Bibliograph provider probe"]},
        ),
        (
            "/api/chat",
            {
                "model": "chat-model",
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "stream": False,
            },
        ),
    ]


def test_openai_compatible_probes_use_the_configured_models():
    calls = []

    def transport(operation, payload):
        calls.append((operation, payload))
        if operation == "embeddings":
            return {"data": [{"index": 0, "embedding": [1, 0, 0]}]}
        if operation == "chat":
            return {"choices": [{"message": {"content": "OK"}}]}
        raise AssertionError(f"unexpected OpenAI-compatible request: {operation}")

    embedding = build_embedding(
        {
            "provider": "openai",
            "model": "embed-model",
            "transport": transport,
        }
    )
    chat = build_chat(
        {
            "provider": "openai-compatible",
            "model": "chat-model",
            "transport": transport,
        }
    )

    assert embedding["probe"]()["dimension"] == 3
    assert chat["probe"]()["ok"] is True
    assert calls == [
        (
            "embeddings",
            {"model": "embed-model", "input": ["Bibliograph provider probe"]},
        ),
        (
            "chat",
            {
                "model": "chat-model",
                "messages": [{"role": "user", "content": "Reply with OK."}],
            },
        ),
    ]
