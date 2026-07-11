from bibliograph.embeddings import HashEmbedder
from bibliograph.ingest import index_chunks, paper_from_zotero_item
from bibliograph.store import SQLiteIndex


def test_paper_from_zotero_item_keeps_citation_fields():
    paper = paper_from_zotero_item(
        {
            "data": {
                "key": "K1",
                "title": "A useful study",
                "date": "2024-05-01",
                "DOI": "10/example",
                "creators": [{"firstName": "Ada", "lastName": "Lovelace"}],
            }
        },
        ["Methods"],
    )
    assert paper.citation_label == "Lovelace (2024)"
    assert paper.doi == "10/example"
    assert paper.collections == ("Methods",)


def test_index_chunks_is_idempotent():
    paper = paper_from_zotero_item({"data": {"key": "K1", "title": "Study"}})
    index = SQLiteIndex(":memory:")
    embedder = HashEmbedder()
    pages = [(1, "A stable chunk for indexing.")]

    assert index_chunks(index, embedder, paper, pages, max_words=20, overlap_words=2) == 1
    assert index_chunks(index, embedder, paper, pages, max_words=20, overlap_words=2) == 1
    assert len(index.search(embedder.embed(["stable indexing"])[0])) == 1
