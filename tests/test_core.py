from pathlib import Path
from tempfile import TemporaryDirectory

from bibliograph.chunking import chunk_pages, split_text
from bibliograph.embeddings import (
    HashEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
)
from bibliograph.models import Paper
from bibliograph.store import SQLiteIndex


def test_split_text_overlaps_and_preserves_all_short_text():
    chunks = split_text("one two three four five", max_words=3, overlap_words=1)
    assert chunks == ["one two three", "three four five"]


def test_chunk_pages_preserves_section_context_in_embedding_text():
    paper = Paper("SECTION", "Sectioned Paper")

    chunks = chunk_pages(
        paper,
        [(2, "The result is important. It generalizes well.", "Results")],
        max_words=20,
        overlap_words=2,
    )

    assert chunks[0].section == "Results"
    assert chunks[0].text.startswith("Section: Results\n\n")


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


def test_sqlite_vec_index_persists_across_reopen():
    paper = Paper("PERSIST", "Persistent Paper")
    chunk = chunk_pages(paper, [(2, "persistent vector evidence")], max_words=20, overlap_words=2)
    embedder = HashEmbedder()
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        index = SQLiteIndex(path)
        index.upsert(chunk, embedder.embed([item.text for item in chunk]))
        index.close()

        reopened = SQLiteIndex(path)
        result = reopened.search(embedder.embed(["vector evidence"])[0], limit=1)
        reopened.close()

    assert result[0].chunk.paper.zotero_key == "PERSIST"


def test_changing_embedding_identity_clears_persistent_index_for_reindexing():
    paper = Paper("MODEL", "Model-specific Paper")
    chunk = chunk_pages(paper, [(1, "model-specific evidence")], max_words=20, overlap_words=2)
    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "index.db"
        first = SQLiteIndex(path, embedding_id="model-a")
        first.upsert(chunk, [[1.0, 0.0]])
        first.mark_file_indexed("MODEL", 1, __file__)
        first.close()

        changed = SQLiteIndex(path, embedding_id="model-b")

        assert changed.reindexed is True
        assert changed.search([1.0, 0.0]) == []
        assert changed.needs_file_index("MODEL", 1, __file__) is True
        changed.upsert(chunk, [[1.0, 0.0, 0.0]])
        assert changed.search([1.0, 0.0, 0.0])[0].chunk.paper.zotero_key == "MODEL"
        changed.close()


def test_sentence_transformer_uses_persistent_cache_and_offline_mode(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", type(
        "SentenceTransformers", (), {"SentenceTransformer": FakeModel}
    ))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_CACHE", ".cache/models")

    embedder = SentenceTransformerEmbedder.from_environment(local_files_only=True)

    assert embedder.cache_dir == ".cache/models"
    assert calls == [(
        "all-MiniLM-L6-v2",
        {"cache_folder": ".cache/models", "local_files_only": True},
    )]
