from threading import Lock
from time import sleep

from bibliograph.domain import Chunk, Paper
from bibliograph.knowledge import QueryRequest, RetrievalResult, RetrievalTrace
from bibliograph.pipeline.drafts import parse_draft
from bibliograph.pipeline.retrieval import enrich_hits, retrieve_claims, search_claim
from bibliograph.render import render_check, render_search


def test_search_render_explains_empty_results_without_repeating_claim():
    rendered = render_search([], claim="Road quality improves market access.")

    assert "> Claim:" not in rendered
    assert "No assessed supporting or contradicting evidence was found." in rendered


def test_systemone_reranks_passages_in_one_named_question_batch():
    from bibliograph.pipeline.llm_tasks import rerank_with_systemone

    paper = Paper("P1", "Road study")
    first = Chunk("P1:1:0", paper, "Roads increased trade.")
    second = Chunk("P1:2:0", paper, "This is background on roads.")
    calls = []

    class BatchDecider:
        def noul_many(self, state, questions):
            calls.append((state, questions))
            return {name: (0.9 if name == "passage_0" else 0.1) for name in questions}

        def noul(self, *_args):
            raise AssertionError("batch-capable decider should use one batch request")

    reranked = rerank_with_systemone(
        "Roads improve trade.",
        [(first, 0.5), (second, 0.4)],
        decider=BatchDecider(),
    )

    assert len(calls) == 1
    state, questions = calls[0]
    assert state["claim"] == "Roads improve trade."
    assert len(state["passages"]) == len(questions) == 2
    assert state["passages"][0]["text"] == first.text
    assert state["passages"][1]["text"] == second.text
    assert reranked == [(first, 0.9), (second, 0.1)]


def test_render_prettifies_local_pdf_filename_titles():
    paper = Paper("LOCAL-1", "spatial_spillover_of_conflict")
    chunk = Chunk("LOCAL-1:1:0", paper, "Conflict along roads raises maize prices.")
    item = {
        "claim": {"text": "Conflict near roads raises maize prices."},
        "chunk": chunk,
        "score": 0.8,
        "evidence": chunk.text,
        "rationale": "Direct evidence.",
        "evidence_relation": "supports",
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
    assert "Assessment:** supports" in report
    assert "| **Title** | Road study |" in report
    assert "| **Vector similarity** | 0.80 |" in report
    assert "| **Lexical closeness** | 0.00 |" in report
    assert "| **Retrieval relevance** | 0.80 |" in report
    assert "| **DOI** | 10/example |" in report
    assert "| **Page** | 1 |" in report
    assert "### Source context" in report
    assert "**Assessment:** supports" in report


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
        limit=10,
        one_per_paper=True,
    )

    assert len(result["hits"]) == 10
    assert len({hit[0].paper.zotero_key for hit in result["hits"]}) == 10
    assert len(enriched) == 10
    assert "P0" not in enriched
    assert "P1" not in enriched


def test_cross_paper_synthesis_preserves_relations_and_source_provenance():
    from bibliograph.pipeline.evidence_workflow import _synthesize_cross_paper

    paper_a = Paper("P1", "Study A")
    paper_b = Paper("P2", "Study B")
    items = [
        {
            "chunk": Chunk("P1:1", paper_a, "Treatment increased the outcome.", page=3),
            "evidence_relation": "supports",
            "matched_excerpt": "Treatment increased the outcome.",
            "evidence_scope": {"population": "adults", "unit": "individual"},
        },
        {
            "chunk": Chunk("P2:1", paper_b, "Treatment did not change the outcome.", page=8),
            "evidence_relation": "contradicts",
            "matched_excerpt": "Treatment did not change the outcome.",
            "evidence_scope": {"population": "children", "unit": "school"},
        },
    ]

    synthesis = _synthesize_cross_paper(items)

    assert synthesis["source_ids_by_relation"] == {
        "supports": ["P1"],
        "contradicts": ["P2"],
        "partial": [],
        "mixed": [],
        "insufficient": [],
    }
    assert synthesis["scope_variation"]["population"] == ["adults", "children"]
    assert "papers disagree" in synthesis["summary"]
    assert "overall, the assessed evidence is mixed" in synthesis["conclusion"].lower()
    assert synthesis["sources"][0]["matched_quote"] == items[0]["matched_excerpt"]


def test_synthesis_of_multiple_passages_from_one_paper_does_not_claim_cross_paper_variation():
    from bibliograph.pipeline.evidence_workflow import _synthesize_cross_paper

    paper = Paper("P1", "One paper")
    items = [
        {
            "chunk": Chunk("P1:1", paper, "First result", page=4),
            "evidence_relation": "supports",
            "evidence_scope": {"population": "ten markets"},
        },
        {
            "chunk": Chunk("P1:2", paper, "Second result", page=24),
            "evidence_relation": "supports",
            "evidence_scope": {"population": "most affected markets"},
        },
    ]

    synthesis = _synthesize_cross_paper(items)

    assert synthesis["paper_count"] == 1
    assert synthesis["passage_count"] == 2
    assert synthesis["scope_variation"] == {}
    assert "Evidence from 1 paper across 2 assessed passages" in synthesis["summary"]
    assert "scope differs across papers" not in synthesis["summary"]
    assert "supports the claim within the sources' reported scopes" in synthesis["conclusion"]


def test_search_report_groups_passages_by_paper():
    paper = Paper("P1", "One paper")
    claim = {"text": "The treatment changed the outcome."}
    first = {
        "claim": claim,
        "chunk": Chunk("P1:1", paper, "The treatment changed the outcome.", page=4),
        "score": 0.9,
        "evidence": "The treatment changed the outcome.",
        "matched_excerpt": "The treatment changed the outcome.",
        "evidence_relation": "supports",
        "evidence_status": "supports",
        "rationale": "Direct result.",
    }
    second = {
        **first,
        "chunk": Chunk("P1:2", paper, "The treatment changed the outcome by 4%.", page=5),
        "score": 0.8,
        "evidence": "The treatment changed the outcome by 4%.",
        "matched_excerpt": "The treatment changed the outcome by 4%.",
    }

    rendered = render_search([first, second])

    assert rendered.count("### Source 1: One paper") == 1
    assert "#### Passage 1 · p. 4" in rendered
    assert "#### Passage 2 · p. 5" in rendered
    assert "**Passage-level assessment:** supports" in rendered


def test_search_report_hides_unassessed_passages_unless_verbose():
    paper = Paper("P1", "A study")
    supported = {
        "claim": {"text": "The treatment increased outcomes."},
        "chunk": Chunk("P1:1:0", paper, "The treatment increased outcomes.", page=4),
        "score": 0.4,
        "evidence": "The treatment increased outcomes.",
        "matched_excerpt": "The treatment increased outcomes.",
        "evidence_relation": "supports",
        "evidence_status": "supports",
        "rationale": "The source passage supports the claim within its stated scope.",
    }
    unassessed = {
        **supported,
        "chunk": Chunk("P1:2:0", paper, "The study aims to evaluate treatment.", page=5),
        "evidence": "The study aims to evaluate treatment.",
        "matched_excerpt": None,
        "evidence_relation": "insufficient",
        "evidence_status": "insufficient",
    }

    default_report = render_search([unassessed, supported])
    verbose_report = render_search([unassessed, supported], verbose=True)

    assert "The treatment increased outcomes." in default_report
    assert "The study aims to evaluate treatment." not in default_report
    assert "The study aims to evaluate treatment." in verbose_report
    assert "Retrieval relevance" in default_report
    assert "Rerank support" not in default_report

    synthesis_report = render_search(
        [supported],
        synthesis={
            "summary": (
                "Evidence from 2 papers across 2 assessed passages: 1 paper(s) with supporting, "
                "0 with partial, 1 with contradicting, and 0 with mixed evidence."
            ),
            "paper_count": 2,
            "sources": [
                {
                    "source_id": "P1",
                    "title": "A study",
                    "page": 4,
                    "relation": "supports",
                },
                {
                    "source_id": "P2",
                    "title": "Another study",
                    "page": 9,
                    "relation": "contradicts",
                },
            ],
        },
    )
    assert "Claim-level synthesis" not in synthesis_report
    assert "**Evidence base:**" in synthesis_report
    assert "**Passage accounting:**" not in synthesis_report
    assert "2 papers · 2 passages" in synthesis_report
    assert "⟦support⟧1 supporting⟦/support⟧" in synthesis_report
    assert "⟦contradict⟧1 contradicting⟦/contradict⟧" in synthesis_report
    assert "descriptive" not in synthesis_report
    assert "scope differs" not in synthesis_report
    assert "Source-level evidence" not in synthesis_report
    assert "#### Passage 1 · p. 4" not in synthesis_report

    verbose_report = render_search([supported], verbose=True, synthesis={"sources": [{
        "source_id": "P1", "title": "A study", "page": 4, "relation": "supports"
    }]})
    assert "Source-level evidence" in verbose_report
    assert "#### Passage 1 · p. 4" in verbose_report


def test_claim_coverage_prints_excerpt_before_its_source_attribution():
    report = render_search(
        [],
        claim="A claim",
        synthesis={
            "sources": [{"source_id": "P1", "title": "A long source title", "page": 9}],
            "claim_coverage": [
                {
                    "name": "Component",
                    "status": "partial",
                    "evidence": [
                        {"title": "A long source title", "page": 9, "quote": "A short excerpt."}
                    ],
                }
            ],
        },
    )

    assert "  - “A short excerpt.”\n    — A long source title, p. 9" in report
    assert "Source evidence (" not in report


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
            "evidence_relation": "supports",
        },
        {
            "claim": {"text": "Claim", "citation_keys": ()},
            "chunk": chunk_high,
            "score": 0.9,
            "evidence": "Road quality improves market access.",
            "rationale": "High score",
            "evidence_relation": "supports",
        },
    ]

    report = render_search(items)

    assert report.index("Road quality improves market access") < report.index(
        "Infrastructure supports trade"
    )


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


def test_refine_queries_targets_gaps_in_assessed_evidence_ledger():
    from bibliograph.pipeline.llm_tasks import refine_queries

    prompts = []

    def complete(messages, *, json_mode=False, response_schema=None):
        prompts.append((messages, json_mode, response_schema))
        return '{"queries":["treatment effect on reported outcome"]}'

    queries = refine_queries(
        "Treatment changed an outcome.",
        [
            "Source A, page 2: insufficient. Assessment gap/reason: population is unclear. "
            "Retrieved context: treatment effect for the measured outcome."
        ],
        complete=complete,
    )

    assert queries == ["treatment effect on reported outcome"]
    assert prompts[0][1] is True
    prompt = "\\n".join(message["content"] for message in prompts[0][0])
    assert "Treatment changed an outcome." in prompt
    assert "Assessment gap/reason: population is unclear" in prompt
    assert "ASSESSED EVIDENCE LEDGER" in prompt
    assert "Do not generate queries from general associations" in prompt
    assert "missing population" in prompt


def test_langgraph_search_refines_queries_from_retrieved_context():
    from bibliograph.knowledge import RetrievalResult

    paper = Paper("P1", "Contextual study")
    initial = Chunk("P1:1:0", paper, "The relationship was estimated in the sample.")
    followup = Chunk("P1:2:0", paper, "The treatment increased the measured outcome.")
    requests = []
    trace = RetrievalTrace(nodes=({"id": "P1:1:0", "kind": "chunk"},))

    class Backend:
        def retrieve_request(self, request, *, limit, include_trace):
            requests.append((request, limit, include_trace))
            chunk = initial if request.text == "Claim about treatment and outcome" else followup
            return RetrievalResult(
                hits=((chunk, 0.8),),
                score_details={chunk.chunk_id: {"retrieval": 0.8}},
                trace=trace,
            )

        def context_for(self, chunk):
            return f"Adjacent context clarifies the terminology. {chunk.text}"

    refinements = []
    explanations = []

    def explain(claim, evidence):
        explanations.append((claim, evidence))
        return "Explanation based on displayed context."

    def refine(claim, excerpts):
        refinements.append((claim, excerpts))
        return ["treatment effect on measured outcome"]

    def select(_claim, hit, context):
        if hit[0] == followup:
            return (
                "The treatment increased the measured outcome.",
                "The excerpt reports a finding.",
            )
        return {
            "relation": "insufficient",
            "matched_quote": "",
            "reason": "The population and unit of analysis are unclear.",
            "scope": {"population": "", "unit": "", "outcome": "", "geography": "", "time": ""},
        }

    result = search_claim(
        {"text": "Claim about treatment and outcome", "citation_keys": ()},
        backend=Backend(),
        expand=lambda _claim: [],
        refine=refine,
        rerank=lambda _claim, hits: list(hits),
        select_evidence=select,
        explain=explain,
        enrich=True,
    )

    assert [request.text for request, _limit, _trace in requests] == [
        "Claim about treatment and outcome",
        "treatment effect on measured outcome",
    ]
    assert all(limit == 20 and include_trace for _request, limit, include_trace in requests)
    assert refinements[0][0] == "Claim about treatment and outcome"
    assert "Adjacent context clarifies the terminology" in refinements[0][1][0]
    assert (
        "Assessment gap/reason: The population and unit of analysis are unclear"
        in refinements[0][1][0]
    )
    assert result["refinement_rounds"] == 1
    assert result["refinement_trace"][0]["queries"] == ["treatment effect on measured outcome"]
    assert result["candidate_queries"][followup.chunk_id] == [
        "treatment effect on measured outcome"
    ]
    supported_item = next(item for item in result["items"] if item["chunk"] == followup)
    expected_context = (
        "Adjacent context clarifies the terminology. The treatment increased the measured outcome."
    )
    assert supported_item["evidence"] == expected_context
    assert supported_item["matched_excerpt"] == "The treatment increased the measured outcome."
    assert supported_item["rationale"] == "The excerpt reports a finding."
    assert explanations == []
    assert supported_item["evidence_status"] == "supports"
    rendered_supported = render_search([supported_item])
    assert (
        "⟦highlight⟧The treatment increased the measured outcome.⟦/highlight⟧" in rendered_supported
    )
    assert "### Matched excerpt" not in rendered_supported
    unverified_item = next(item for item in result["items"] if item["chunk"] == initial)
    assert unverified_item["evidence_status"] == "insufficient"
    assert unverified_item["rationale"] == "The population and unit of analysis are unclear."


def test_claim_only_expansion_is_fallback_when_initial_retrieval_has_no_candidates():
    from bibliograph.knowledge import RetrievalResult

    queries = []

    class Backend:
        def retrieve_request(self, request, *, limit, include_trace):
            queries.append(request.text)
            return RetrievalResult(hits=(), score_details={})

    expanded = []

    def expand(claim):
        expanded.append(claim)
        return ["treatment outcome measured in clinics"]

    result = search_claim(
        {"text": "Treatment improved clinic outcomes.", "citation_keys": ()},
        backend=Backend(),
        expand=expand,
        refine=lambda *_args: [],
    )

    assert expanded == ["Treatment improved clinic outcomes."]
    assert queries == [
        "Treatment improved clinic outcomes.",
        "treatment outcome measured in clinics",
    ]
    assert result["refinement_trace"][0]["planning_basis"] == (
        "no initial candidates; claim-only fallback expansion"
    )


def test_langgraph_extends_local_context_before_query_refinement():
    from bibliograph.knowledge import RetrievalResult

    paper = Paper("P1", "Context extension study")
    chunk = Chunk("P1:1:0", paper, "The treatment was assessed.", section="Results")
    context_windows = []

    class Backend:
        def retrieve_request(self, _request, *, limit, include_trace):
            return RetrievalResult(hits=((chunk, 0.8),), score_details={})

        def context_for(self, _chunk, *, window=1):
            context_windows.append(window)
            if window == 1:
                return "The treatment was assessed."
            return "In the sampled clinics, the treatment increased the measured outcome."

    def select(_claim, _hit, context):
        quote = "In the sampled clinics, the treatment increased the measured outcome."
        if quote in context:
            return {
                "relation": "supports",
                "matched_quote": quote,
                "reason": "The passage reports the outcome for the sampled clinics.",
                "scope": {
                    "population": "sampled clinics",
                    "unit": "clinic",
                    "outcome": "measured outcome",
                    "geography": "",
                    "time": "",
                },
                "quote_role": "finding",
                "quote_directness": "direct",
            }
        return {
            "relation": "insufficient",
            "matched_quote": "",
            "reason": "The result and population are not stated in this chunk.",
            "scope": {},
        }

    result = search_claim(
        {"text": "The treatment increased the measured outcome in clinics.", "citation_keys": ()},
        backend=Backend(),
        expand=lambda _claim: [],
        refine=lambda *_args: [],
        rerank=lambda _claim, hits: list(hits),
        select_evidence=select,
    )

    item = result["items"][0]
    assert context_windows == [1, 2]
    assert item["evidence_relation"] == "supports"
    assert item["matched_excerpt"] == (
        "In the sampled clinics, the treatment increased the measured outcome."
    )
    assert item["evidence_scope"]["population"] == "sampled clinics"
    assert item["rationale"] == "The passage reports the outcome for the sampled clinics."
    assert result["context_windows"][chunk.chunk_id] == 2
    assert result["refinement_rounds"] == 0


def test_workflow_rejects_assessment_quotes_not_found_in_context():
    from bibliograph.pipeline.evidence_workflow import _normalise_assessment

    result = _normalise_assessment(
        {
            "relation": "supports",
            "matched_quote": "Invented supporting sentence.",
            "reason": "The model claimed support.",
            "quote_directness": "direct",
        },
        "The source only describes its methodology.",
    )

    assert result["relation"] == "insufficient"
    assert result["matched_quote"] == ""


def test_langgraph_search_stops_refining_after_supported_evidence():
    from bibliograph.knowledge import RetrievalResult

    paper = Paper("P1", "Evidence paper")
    chunk = Chunk("P1:1:0", paper, "The intervention increased the outcome.")

    class Backend:
        def retrieve_request(self, _request, *, limit, include_trace):
            return RetrievalResult(hits=((chunk, 0.9),), score_details={})

    refined = []
    result = search_claim(
        {"text": "Intervention increased the outcome", "citation_keys": ()},
        backend=Backend(),
        expand=lambda _claim: [],
        refine=lambda *_args: refined.append(True) or ["more terms"],
        select_evidence=lambda _claim, _hit, _context: (chunk.text, "Direct result."),
    )

    assert refined == []
    assert result["refinement_rounds"] == 0
    assert result["queries"] == ("Intervention increased the outcome",)


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
    assert all(not request.hypotheses for request, _limit, _trace in requests)
    assert [request.text for request, _limit, _trace in requests] == [
        "Road quality affects market access."
    ]
    assert limit == 20
    assert include_trace is True
    assert len(result["query_traces"]) == 1
    assert result["retrieval_trace"] is trace
    retrieval_update = next(
        message for message in progress_messages if message.startswith("Retrieved candidates ·")
    )
    assert retrieval_update == "Retrieved candidates · 1 unique · 1 query variant(s)"
    assert all(not message.startswith("Search subtask results:") for message in progress_messages)
    assert all("graph expansion followed" not in message for message in progress_messages)


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
            assert alternatives == ()
            assert limit == 20
            return [(chunk, 0.8)], {chunk.chunk_id: {"retrieval": 0.8}}

    result = search_claim(
        {"text": "Road quality affects market access.", "citation_keys": ()},
        backend=Backend(),
        enrich=False,
    )

    assert result["hits"] == [(chunk, 0.8)]
    assert result["items"][0]["chunk"] == chunk
