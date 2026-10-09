from bibliograph.domain import Chunk, Paper
from bibliograph.pipeline.classification import PAPER_EVIDENCE_PROFILE, paper_evidence_strategy
from bibliograph.pipeline.llm_tasks import rerank


def test_paper_evidence_strategy_attaches_category_scores_and_version():
    calls = []

    def complete(messages, *, json_mode=False, response_schema=None):
        calls.append((messages, json_mode, response_schema))
        return (
            '{"scores":{"background":0.02,"data":0.03,"methods":0.08,'
            '"results":0.82,"robustness":0.02,"discussion":0.02,"limitations":0.01}}'
        )

    strategy = paper_evidence_strategy(complete)
    prepared = strategy("Section: Results\n\nWe find that road access increases trade.")

    assert prepared["category"] == "results"
    assert prepared["metadata"]["evidence_role"] == "results"
    assert prepared["metadata"]["evidence_role_scores"]["results"] == 0.82
    assert prepared["metadata"]["classification_profile"] == PAPER_EVIDENCE_PROFILE["name"]
    assert calls[0][1] is True
    assert calls[0][2]["properties"]["scores"]["required"] == [
        category["name"] for category in PAPER_EVIDENCE_PROFILE["categories"]
    ]



def test_reranker_receives_ingested_evidence_role_as_context():
    chunk = Chunk(
        "P1:1:0",
        Paper("P1", "Road study"),
        "We find that road access increases trade.",
        section="Results",
        evidence_role="results",
    )
    prompts = []

    def complete(messages, *, json_mode=False, response_schema=None):
        prompts.append("\n".join(message["content"] for message in messages))
        return (
            '{"items":[{"candidate":0,"relation":"supports",'
            '"evidence_role":"result","directly_entails_claim":true,'
            '"contains_claim_specific_result":true,"contains_explicit_finding":true,'
            '"requires_unsupported_inference":false,"source_type":"primary"}]}'
        )

    ranked = rerank("Road access increases trade.", [(chunk, 0.5)], complete=complete)

    assert ranked[0][0] == chunk
    assert "indexed evidence role results" in prompts[0]
    assert "Preserve the claim's population" in prompts[0]
