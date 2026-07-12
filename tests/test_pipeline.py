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
    assert "- **Title:** Road study" in report
    assert "- **Support score:** `0.80`" in report
    assert "- **DOI:** 10/example" in report
    assert "- **Page:** 1" in report
    assert "### Excerpt" in report
    assert "### Why this fits" in report


def test_rendered_references_are_sorted_by_support_score():
    paper = Paper("P1", "Road study", ("Ada",), "2024", "10/example")
    chunk_high = Chunk("P1:1:0", paper, "Road quality improves market access.", page=1)
    chunk_low = Chunk("P1:2:0", paper, "Infrastructure supports trade.", page=2)
    items = [
        {
            "claim": {"text": "Claim", "citation_keys": ()},
            "chunk": chunk_low,
            "score": 0.2,
            "evidence": "Infrastructure supports trade.",
            "rationale": "Low score",
        },
        {
            "claim": {"text": "Claim", "citation_keys": ()},
            "chunk": chunk_high,
            "score": 0.9,
            "evidence": "Road quality improves market access.",
            "rationale": "High score",
        },
    ]

    report = render_search(items)

    assert report.index("High score") < report.index("Low score")


def test_evidence_enrichment_prefers_sentence_excerpts_over_metadata_like_quotes():
    paper = Paper("P1", "Road study", ("Akpan",), "2024", "10/example")
    chunk = Chunk(
        "P1:1:0",
        paper,
        "Section: 6. Concluding Remarks South Africa has improved road quality. "
        "Transport infrastructure promotes cross-border trade and raises domestic output, "
        "thus fostering regional integration.",
        page=7,
    )
    claims = [{"text": "Transport infrastructure fosters regional integration.", "citation_keys": ()}]

    def embed(texts):
        return [[1.0, 0.0]]

    def search(_vector, *, limit, query_text):
        return [(chunk, 0.9)]

    items = enrich_hits(
        retrieve_claims(claims, embed_queries=embed, search=search, limit=1),
        context_for=lambda _chunk: chunk.text,
        select_evidence=lambda _claim, _hit, _context: (
            "Section: 6. Concluding Remarks",
            "Heading-like text should be rejected.",
        ),
        explain=lambda _claim, evidence: f"Rationale for {evidence}",
        top_only=True,
    )

    assert items[0]["evidence"].startswith("Transport infrastructure promotes cross-border trade")
    assert items[0]["rationale"] == f"Rationale for {items[0]['evidence']}"


def test_explain_strips_label_prefixes_from_llm_output():
    from bibliograph.pipeline.llm_tasks import explain

    content = explain(
        "A claim",
        "Some evidence",
        complete=lambda _messages, json_mode=False: "SUPPORT. The evidence directly addresses the claim.",
    )

    assert content == "The evidence directly addresses the claim."
