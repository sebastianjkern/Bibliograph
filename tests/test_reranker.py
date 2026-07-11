from bibliograph.models import CitationSource, DraftMatch, Paper, TextChunk
from bibliograph.reranker import OpenAIReranker, rerank_matches


class _Response:
    def __init__(self, content):
        message = type("Message", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message})()]


class _Client:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
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
