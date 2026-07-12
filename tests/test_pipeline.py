from bibliograph.domain import Chunk, Paper
from bibliograph.pipeline.drafts import parse_draft
from bibliograph.pipeline.retrieval import enrich_hits, retrieve_claims
from bibliograph.render import render_check, render_search


def test_format_aware_draft_parser_returns_plain_claim_mappings():
    claims = parse_draft(
        "Road quality shapes trade costs and market access. \\cite{roads2024}\n\n"
        "A second claim needs a citation too.",
        "latex",
    )

    assert claims == [
        {
            "text": "Road quality shapes trade costs and market access.",
            "source_format": "latex",
            "line_start": 1,
            "citation_keys": ("roads2024",),
        },
        {
            "text": "A second claim needs a citation too.",
            "source_format": "latex",
            "line_start": 3,
            "citation_keys": (),
        },
    ]


def test_shared_retrieval_pipeline_drives_search_and_check_rendering():
    paper = Paper("P1", "Road study", ("Ada",), "2024", "10/example")
    chunk = Chunk("P1:1:0", paper, "Road quality improves market access.", page=1)
    claims = [{"text": "Road quality affects market access.", "citation_keys": ()}]

    def embed(texts):
        assert texts == ["Road quality affects market access."]
        return [[1.0, 0.0]]

    def search(vector, *, limit, query_text):
        assert vector == [1.0, 0.0]
        assert limit == 3
        assert query_text == claims[0]["text"]
        return [(chunk, 0.8)]

    results = retrieve_claims(claims, embed_queries=embed, search=search, limit=3)
    items = enrich_hits(
        results,
        context_for=lambda _chunk: "Before. Road quality improves market access. After.",
        select_evidence=lambda _claim, _hit, _context: (
            "Road quality improves market access.",
            "Direct support.",
        ),
        explain=lambda _claim, _evidence: "unused",
    )

    assert "# Local sources" in render_search(items)
    report = render_check(items)
    assert "# Citation suggestions" in report
    assert "Direct support." in report
