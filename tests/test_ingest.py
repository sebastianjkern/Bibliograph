from pathlib import Path
from tempfile import TemporaryDirectory

from bibliograph.embeddings import HashEmbedder
from bibliograph.ingest import extract_pdf_sections, index_chunks, paper_from_zotero_item
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


def test_extract_pdf_sections_removes_repeated_margins_and_references(monkeypatch):
    from types import ModuleType

    class Page:
        rect = type("Rect", (), {"height": 100})()

        def __init__(self, blocks):
            self.blocks = blocks

        def get_text(self, kind):
            assert kind == "blocks"
            return self.blocks

    class Document:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self.pages)

        def __len__(self):
            return len(self.pages)

    fake_fitz = ModuleType("fitz")
    fake_fitz.open = lambda path: Document(
        [
            Page(
                [
                    (0, 10, 100, 20, "Journal Header\n1", 0, 0),
                    (0, 30, 100, 40, "1 Introduction", 1, 0),
                    (0, 45, 100, 60, "Climate systems vary over time.", 2, 0),
                ]
            ),
            Page(
                [
                    (0, 10, 100, 20, "Journal Header\n2", 0, 0),
                    (0, 30, 100, 40, "2 Results", 1, 0),
                    (0, 45, 100, 60, "The result is robust.", 2, 0),
                    (0, 70, 100, 80, "References", 3, 0),
                    (0, 82, 100, 95, "Ignored reference entry.", 4, 0),
                ]
            ),
        ]
    )
    monkeypatch.setitem(__import__("sys").modules, "fitz", fake_fitz)

    with TemporaryDirectory(dir=".") as directory:
        path = Path(directory) / "paper.pdf"
        path.write_bytes(b"%PDF-test")
        pages = extract_pdf_sections(path)

    assert pages == [
        (1, "Climate systems vary over time.", "1 Introduction"),
        (2, "The result is robust.", "2 Results"),
    ]
