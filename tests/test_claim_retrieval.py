from bibliograph.drafts import parse_draft
from bibliograph.embeddings import HashEmbedder
from bibliograph.models import Paper, TextChunk
from bibliograph.retrieval import find_claim_citations
from bibliograph.store import SQLiteIndex


def test_claim_retrieval_preserves_source_location_and_existing_citations():
    index = SQLiteIndex(":memory:")
    paper = Paper("P1", "Methods", ("Smith",), "2022")
    chunk = TextChunk("P1:1:0", paper, "Bayesian methods improve uncertainty estimates", page=7)
    embedder = HashEmbedder()
    index.upsert([chunk], embedder.embed([chunk.text]))
    claims = parse_draft("This claim describes Bayesian methods.\\citep{existing2020}", "latex")

    matches = find_claim_citations(claims, index, embedder, limit=1)

    assert matches[0].line_start == 1
    assert matches[0].citation_keys == ("existing2020",)
    assert matches[0].sources[0].chunk.page == 7
