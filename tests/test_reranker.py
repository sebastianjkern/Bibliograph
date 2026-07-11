from bibliograph.config import openai_settings
from bibliograph.models import CitationSource, DraftMatch, Paper, TextChunk
from bibliograph.reranker import OpenAIReranker, rerank_matches


class _Response:
    def __init__(self, content):
        message = type("Message", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message})()]


class _Client:
    calls = []

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                _Client.calls.append(kwargs)
                return _Response(
                    '{"items": [{"candidate": 1, "support": 0.95}, '
                    '{"candidate": 0, "support": 0.2}]}'
                )


def test_llm_reranker_orders_candidates_and_clamps_scores():
    paper = Paper("P", "Paper", ("Author",))
    sources = (
        CitationSource(TextChunk("1", paper, "weak evidence"), 0.8),
        CitationSource(TextChunk("2", paper, "strong evidence"), 0.7),
    )
    matches = [DraftMatch("A claim", sources)]
    reranker = OpenAIReranker("model", "key", client=_Client())

    result = rerank_matches(matches, reranker)[0]

    assert result.sources[0].chunk.text == "strong evidence"
    assert result.sources[0].score == 0.95
    assert "response_format" not in _Client.calls[-1]


def test_openai_settings_fall_back_when_api_key_is_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    assert openai_settings() == ("lm-studio", "http://localhost:1234/v1")
