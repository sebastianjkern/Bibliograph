"""Paper-specific chunk classification using Ikarus's classification contracts."""

from collections.abc import Callable
from typing import Any

from core.knowledge.classification import (
    ClassificationProfile,
    IngestionStrategy,
    classification_strategy,
    make_classifier,
)

from .llm_tasks import _complete_json, _json_object

PROFILE_VERSION = "bibliograph-paper-evidence-v1"

PAPER_EVIDENCE_PROFILE: ClassificationProfile = {
    "name": PROFILE_VERSION,
    "question": (
        "Classify the role of this passage in an empirical scientific paper. "
        "Use its section heading as context, but classify the passage itself."
    ),
    "categories": (
        {"name": "background", "description": "Prior work, literature review, or context."},
        {"name": "data", "description": "Data sources, samples, variables, or measurement."},
        {
            "name": "methods",
            "description": "Research design, identification, or estimation methods.",
        },
        {"name": "results", "description": "Findings or estimates produced by this paper."},
        {
            "name": "robustness",
            "description": (
                "Robustness checks, sensitivity analyses, or alternative specifications."
            ),
        },
        {"name": "discussion", "description": "Interpretation, implications, or conclusions."},
        {"name": "limitations", "description": "Caveats, limitations, or scope conditions."},
    ),
}


def paper_evidence_strategy(complete: Callable[..., str]) -> IngestionStrategy:
    """Build an Ikarus ingestion strategy that attaches evidence-role scores."""
    category_names = tuple(
        category["name"] for category in PAPER_EVIDENCE_PROFILE["categories"]
    )
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores"],
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(category_names),
                "properties": {
                    name: {"type": "number", "minimum": 0, "maximum": 1}
                    for name in category_names
                },
            }
        },
    }

    def decision_backend(request) -> dict[str, float]:
        categories = "\n".join(
            f"- {item['name']}: {item['description']}"
            for item in request["categories"]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify a scientific-paper passage by its role. Return a score from 0 to 1 "
                    "for every category, where scores indicate fit and need not sum to 1. "
                    "Use the passage rather than relying only on its heading."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION: {request['question']}\nCATEGORIES:\n{categories}\n\n"
                    f"PASSAGE:\n{request['state']}"
                ),
            },
        ]
        payload = _json_object(_complete_json(complete, messages, schema))
        scores = payload.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("classification response must contain category scores")
        return {name: float(scores.get(name, 0.0)) for name in category_names}

    classifier = make_classifier(decision_backend, PAPER_EVIDENCE_PROFILE)
    strategy = classification_strategy(classifier)

    def prepare(text: str) -> dict[str, Any]:
        classified = strategy(text)
        return {
            **classified,
            "metadata": {
                "evidence_role": classified["category"],
                "evidence_role_scores": dict(classified["scores"]),
                "classification_profile": PROFILE_VERSION,
            },
        }

    return prepare
