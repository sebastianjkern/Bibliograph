from bibliograph.embeddings import HashEmbedder
from bibliograph.models import Paper, TextChunk
from bibliograph.retrieval import find_citations, find_claim_sources, split_draft
from bibliograph.store import SQLiteIndex


def test_split_draft_ignores_short_fragments():
    assert split_draft("Note. This sentence has enough words to keep.") == [
        "This sentence has enough words to keep."
    ]


def test_find_citations_returns_one_match_per_draft_passage():
    paper = Paper("P1", "Semantic Study", ("Author",), "2020")
    chunk = TextChunk("P1:1:0", paper, "semantic search finds related evidence", page=1)
    embedder = HashEmbedder()
    index = SQLiteIndex(":memory:")
    index.upsert([chunk], embedder.embed([chunk.text]))

    matches = find_citations("Semantic search finds related evidence.", index, embedder, limit=1)

    assert len(matches) == 1
    assert matches[0].sources[0].chunk.paper.title == "Semantic Study"


def test_find_claim_sources_supports_short_direct_claims():
    paper = Paper("P1", "Semantic Study", ("Author",), "2020")
    chunk = TextChunk("P1:1:0", paper, "semantic search evidence", page=1)
    embedder = HashEmbedder()
    index = SQLiteIndex(":memory:")
    index.upsert([chunk], embedder.embed([chunk.text]))

    matches = find_claim_sources("semantic search", index, embedder, limit=1)

    assert matches[0].draft_text == "semantic search"
    assert matches[0].sources[0].chunk.paper.title == "Semantic Study"
