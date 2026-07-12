import json
from io import BytesIO

from bibliograph.ollama import OllamaClient


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_ollama_client_calls_native_chat_api(monkeypatch):
    captured = {}

    def open_request(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(b'{"message":{"content":"result"}}')

    monkeypatch.setattr("bibliograph.ollama.urlopen", open_request)
    client = OllamaClient("http://ollama.test/", timeout=3, json_mode=True)

    response = client.chat.completions.create(
        model="qwen3", messages=[{"role": "user", "content": "hello"}]
    )

    assert response.choices[0].message.content == "result"
    assert captured == {
        "url": "http://ollama.test/api/chat",
        "payload": {
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "options": {"temperature": 0},
            "format": "json",
        },
        "timeout": 3,
    }


def test_ollama_client_calls_native_embed_api(monkeypatch):
    def open_request(request, timeout):
        assert request.full_url == "http://ollama.test/api/embed"
        assert json.loads(request.data) == {"model": "nomic", "input": ["one", "two"]}
        return _Response(b'{"embeddings":[[1,0],[0,1]]}')

    monkeypatch.setattr("bibliograph.ollama.urlopen", open_request)

    response = OllamaClient("http://ollama.test").embeddings.create(
        model="nomic", input=["one", "two"]
    )

    assert [item.embedding for item in response.data] == [[1, 0], [0, 1]]
