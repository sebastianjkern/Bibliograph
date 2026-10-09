from threading import Lock
from time import sleep

from bibliograph.domain import Chunk, Paper
from bibliograph.knowledge import QueryRequest, RetrievalResult, RetrievalTrace
from bibliograph.pipeline.drafts import parse_draft
from bibliograph.pipeline.retrieval import enrich_hits, retrieve_claims, search_claim
from bibliograph.render import render_check, render_search



def test_search_render_explains_empty_results_and_keeps_claim():
    rendered = render_search([], claim="Road quality improves market access.")

    assert "> Claim: Road quality improves market access." in rendered
    assert "No matching evidence was found in the indexed PDFs." in rendered


def test_render_prettifies_local_pdf_filename_titles():
    paper = Paper("LOCAL-1", "spatial_spillover_of_conflict")
    chunk = Chunk("LOCAL-1:1:0", paper, "Conflict along roads raises maize prices.")
    item = {
        "claim": {"text": "Conflict near roads raises maize prices."},
        "chunk": chunk,
        "score": 0.8,
        "evidence": chunk.text,
        "rationale": "Direct evidence.",
    }

    rendered = render_search([item])

    assert "Spatial Spillover Of Conflict" in rendered
    assert "spatial_spillover_of_conflict" not in rendered


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
    assert "| **Title** | Road study |" in report
    assert "| **Vector similarity** | 0.80 |" in report
    assert "| **Lexical closeness** | 0.00 |" in report
    assert "| **Rerank support** | 0.80 |" in report
    assert "| **DOI** | 10/example |" in report
    assert "| **Page** | 1 |" in report
    assert "### Excerpt" in report
    assert "### Explanation" in report


def test_retrieval_expands_queries_and_deduplicates_before_reranking():
    paper = Paper("P1", "Road study", ("Ada",), "2024")
    first = Chunk("P1:1:0", paper, "Road quality improves market access.")
    second = Chunk("P1:2:0", paper, "Transport infrastructure lowers trade costs.")
    searched = []

    def embed(texts):
        assert texts == [
            "Road quality affects market access.",
            "Transport infrastructure and trade",
        ]
        return [[1.0, 0.0], [0.0, 1.0]]

    def search(vector, *, limit, query_text):
        searched.append((vector, query_text))
        if query_text == "Road quality affects market access.":
            return [(first, 0.7), (second, 0.2)]
        return [(first, 0.9), (second, 0.8)]

    reranked = []

    def rerank(claim, hits):
        reranked.append((claim, hits))
        return list(reversed(hits))

    result = retrieve_claims(
        [{"text": "Road quality affects market access.", "citation_keys": ()}],
        embed_queries=embed,
        search=search,
        expand=lambda _claim: ["Transport infrastructure and trade"],
        rerank=rerank,
        limit=2,
    )[0]

    assert [query for _vector, query in searched] == [
        "Road quality affects market access.",
        "Transport infrastructure and trade",
    ]
    assert reranked[0][0] == "Road quality affects market access."
    assert [hit[1] for hit in reranked[0][1]] == [0.9, 0.8]
    assert result["queries"] == (
        "Road quality affects market access.",
        "Transport infrastructure and trade",
    )


def test_enrichment_queue_processes_suggestions_in_parallel():
    hits = [
        (
            Chunk(
                f"P{index}:0",
                Paper(f"P{index}", f"Road study {index}", ("Ada",), "2024"),
                f"Evidence sentence {index}.",
            ),
            0.8,
        )
        for index in range(4)
    ]
    active = 0
    maximum_active = 0
    lock = Lock()

    def rerank(_claim, candidates, *, progress=None):
        return list(candidates)

    def select(_claim, hit, _context):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        sleep(0.02)
        with lock:
            active -= 1
        return hit[0].text, "Direct support."

    result = search_claim(
        {"text": "A claim", "citation_keys": ()},
        embed_queries=lambda _texts: [[1.0]],
        search=lambda _vector, **_kwargs: hits,
        rerank=rerank,
        context_for=lambda chunk: chunk.text,
        select_evidence=select,
    )

    assert maximum_active > 1
    assert len(result["items"]) == 4


def test_only_top_ten_distinct_articles_are_enriched_after_reranking():
    hits = []
    for index in range(12):
        paper = Paper(f"P{index}", f"Paper {index}")
        hits.append((Chunk(f"P{index}:0", paper, f"Evidence {index}."), index / 10))
    duplicate = Chunk("P11:1", hits[-1][0].paper, "Second passage from paper 11.")
    hits.append((duplicate, 0.95))
    enriched = []

    result = search_claim(
        {"text": "A claim", "citation_keys": ()},
        embed_queries=lambda _texts: [[1.0]],
        search=lambda _vector, **_kwargs: hits,
        context_for=lambda chunk: chunk.text,
        select_evidence=lambda _claim, hit, _context: (
            enriched.append(hit[0].paper.zotero_key) or hit[0].text,
            "Direct support.",
        ),
    )

    assert len(result["hits"]) == 10
    assert len({hit[0].paper.zotero_key for hit in result["hits"]}) == 10
    assert len(enriched) == 10
    assert "P0" not in enriched
    assert "P1" not in enriched


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
    claims = [
        {
            "text": "Transport infrastructure fosters regional integration.",
            "citation_keys": (),
        }
    ]

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
        complete=lambda _messages, json_mode=False: (
            "SUPPORT. The evidence directly addresses the claim."
        ),
    )

    assert content == "The evidence directly addresses the claim."


def test_knowledge_backend_receives_neutral_query_and_returns_trace():
    paper = Paper("P1", "Road study")
    chunk = Chunk("P1:1:0", paper, "Road quality improves market access.")
    requests = []
    trace = RetrievalTrace(
        nodes=({"id": chunk.chunk_id, "kind": "chunk"},),
        edges=(),
        metadata={"seeds": ({"id": chunk.chunk_id, "kind": "chunk"},)},
    )

    class Backend:
        def retrieve_request(self, request: QueryRequest, *, limit, include_trace):
            requests.append((request, limit, include_trace))
            return RetrievalResult(
                hits=((chunk, 0.8),),
                score_details={chunk.chunk_id: {"retrieval": 0.8}},
                trace=trace,
            )

    progress_messages = []
    result = search_claim(
        {"text": "Road quality affects market access.", "citation_keys": ()},
        backend=Backend(),
        expand=lambda _claim: ["Transport infrastructure and trade"],
        enrich=False,
        include_trace=True,
        progress=progress_messages.append,
    )

    request, limit, include_trace = requests[0]
    assert request.text == "Road quality affects market access."
    assert [hypothesis.text for hypothesis in request.hypotheses] == [
        "Transport infrastructure and trade"
    ]
    assert limit == 10
    assert include_trace is True
    assert result["retrieval_trace"] is trace
    partial_results = next(
        message for message in progress_messages if message.startswith("Search subtask results:")
    )
    assert "\\n" not in partial_results
    assert "Query plan: 1 alternatives" in partial_results
    assert "Alternative:" not in partial_results
    assert "Road study" in partial_results
    assert "relevance 0.800" in partial_results
    assert "Road quality improves market access." in partial_results
    assert "started from 1 indexed passages" in partial_results
    assert "P1:1:0" not in partial_results


def test_backend_search_honors_requested_result_limit():
    paper = Paper("P1", "Road study")
    chunks = [
        Chunk(f"P1:{index}:0", paper, f"Road access evidence passage {index}.")
        for index in range(6)
    ]

    class Backend:
        def retrieve_request(self, _request, *, limit, include_trace):
            assert limit == 10
            return RetrievalResult(
                hits=tuple((chunk, 0.8 - index * 0.01) for index, chunk in enumerate(chunks)),
                score_details={},
            )

    result = search_claim(
        {"text": "Road access affects trade.", "citation_keys": ()},
        backend=Backend(),
        limit=2,
        enrich=False,
    )

    assert len(result["hits"]) == 2


def test_backend_retrieval_search_claim_accepts_backend_keyword():
    paper = Paper("P1", "Road study")
    chunk = Chunk("P1:1:0", paper, "Road quality improves market access.")

    class Backend:
        def retrieve(self, query, *, alternatives, limit):
            assert query == "Road quality affects market access."
            assert alternatives == []
            assert limit == 10
            return [(chunk, 0.8)], {chunk.chunk_id: {"retrieval": 0.8}}

    result = search_claim(
        {"text": "Road quality affects market access.", "citation_keys": ()},
        backend=Backend(),
        enrich=False,
    )

    assert result["hits"] == [(chunk, 0.8)]
    assert result["items"][0]["chunk"] == chunk
