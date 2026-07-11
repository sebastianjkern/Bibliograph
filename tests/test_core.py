from bibliograph.chunking import chunk_pages, split_text
from bibliograph.embeddings import HashEmbedder, OpenAICompatibleEmbedder
from bibliograph.models import Paper
from bibliograph.store import SQLiteIndex


def test_split_text_overlaps_and_preserves_all_short_text():
    chunks = split_text("one two three four five", max_words=3, overlap_words=1)
    assert chunks == ["one two three", "three four five"]


def test_index_returns_metadata_aware_match():
    paper = Paper("ABC", "A Paper", ("Ada Lovelace",), "1843", "10/example")
    chunks = chunk_pages(paper, [(4, "semantic citation retrieval")], max_words=10, overlap_words=1)
    embedder = HashEmbedder()
    index = SQLiteIndex(":memory:")
    index.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))

    result = index.search(embedder.embed(["citation retrieval"])[0], limit=1)[0]

    assert result.chunk.paper.title == "A Paper"
    assert result.chunk.paper.doi == "10/example"
    assert result.chunk.page == 4


class _EmbeddingResponse:
    def __init__(self, data):
        self.data = data


class _EmbeddingClient:
    class embeddings:
        @staticmethod
        def create(model, input):
            assert model == "local-embedder"
            return _EmbeddingResponse(
                [
                    type("Item", (), {"index": 1, "embedding": [0.0, 1.0]})(),
                    type("Item", (), {"index": 0, "embedding": [1.0, 0.0]})(),
                ]
            )


def test_openai_compatible_embedder_sorts_results_and_tracks_dimension():
    embedder = OpenAICompatibleEmbedder("local-embedder", "unused", client=_EmbeddingClient())

    vectors = embedder.embed(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert embedder.dimension == 2
