"""Small native Ollama API client with OpenAI-client-shaped responses."""

import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ollama_base_url


class OllamaClient:
    """Call Ollama's native ``/api/chat`` and ``/api/embed`` endpoints."""

    def __init__(self, base_url: str | None = None, timeout: float = 120.0, json_mode=False):
        self.base_url = (base_url or ollama_base_url()).rstrip("/")
        self.timeout = timeout
        self.json_mode = json_mode
        self.chat = SimpleNamespace(completions=_OllamaChatCompletions(self))
        self.embeddings = _OllamaEmbeddings(self)

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}: {error.reason}"
            ) from error
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"Ollama error: {result['error']}")
        return result


class _OllamaChatCompletions:
    def __init__(self, client: OllamaClient):
        self.client = client

    def create(self, *, model: str, messages: list[dict], temperature=0, **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if self.client.json_mode:
            payload["format"] = "json"
        result = self.client._post("/api/chat", payload)
        content = result.get("message", {}).get("content", "")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _OllamaEmbeddings:
    def __init__(self, client: OllamaClient):
        self.client = client

    def create(self, *, model: str, input: list[str], **kwargs):
        result = self.client._post("/api/embed", {"model": model, "input": input})
        embeddings = result.get("embeddings", [])
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector)
                for index, vector in enumerate(embeddings)
            ]
        )
