from __future__ import annotations

import re

import pytest

from bibliograph.adapters.ragtime import (
    RagtimeBackend,
    RagtimeIndexError,
    inspect_ragtime_index,
)
from bibliograph.domain import Paper


def _embedding():
    vocabulary = ("road", "trade", "cell", "protein")

    def embed(texts):
        return [
            [float(len(re.findall(rf"\b{term}\w*", text.casefold()))) for term in vocabulary]
            + [1.0]
            for text in texts
        ]

    return {
        "id": "test/scientific-v1",
        "embed_documents": embed,
        "embed_queries": embed,
        "batch_size": 8,
    }


def test_ragtime_backend_indexes_scientific_papers_and_returns_bibliograph_chunks(tmp_path):
    database = tmp_path / "index.db"
    road_pdf = tmp_path / "road.pdf"
    biology_pdf = tmp_path / "biology.pdf"
    road_pdf.write_bytes(b"%PDF-road")
    biology_pdf.write_bytes(b"%PDF-biology")

    with RagtimeBackend(database, embedding=_embedding(), mode="write") as backend:
        backend.index_pdf(
            Paper("ROAD", "Roads and Markets", ("Ada Author",), "2024", "10.1/road"),
            "ROAD",
            "1",
            road_pdf,
            extract_pages=lambda _path: [
                (1, "Better road quality lowers trade costs and expands market access.", "Results")
            ],
        )
        backend.index_pdf(
            Paper("BIO", "Cell Biology"),
            "BIO",
            "1",
            biology_pdf,
            extract_pages=lambda _path: [(3, "Protein folding occurs inside a cell.", "Methods")],
        )
        hits, details = backend.retrieve("road trade market access", limit=5)

    assert hits
    assert hits[0][0].paper.zotero_key == "ROAD"
    assert hits[0][0].page == 1
    assert hits[0][0].section == "Results"
    assert hits[0][0].chunk_id in details
    assert inspect_ragtime_index(database)["documents"] == 2


def test_ragtime_backend_rejects_legacy_databases_with_rebuild_guidance(tmp_path):
    database = tmp_path / "legacy.db"
    database.write_bytes(b"not a Ragtime database")

    with pytest.raises(RagtimeIndexError, match="sync --rebuild"):
        RagtimeBackend(database, embedding=_embedding(), mode="read")
