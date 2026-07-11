from bibliograph.export import to_markdown
from bibliograph.models import CitationSource, DraftMatch, Paper, TextChunk
from bibliograph.suggestions import (
    EvidenceSelection,
    GroundedTemplateGenerator,
    OpenAIEvidenceExtractor,
    suggest_citations,
)


def _match() -> DraftMatch:
    paper = Paper("P1", "A Study", ("Doe",), "2021", "10/example")
    chunk = TextChunk("P1:3:0", paper, "Evidence from page three", page=3)
    return DraftMatch("The draft claim.", (CitationSource(chunk, 0.9),))


def test_suggestion_is_grounded_and_contains_page_and_doi():
    suggestion = suggest_citations([_match()], GroundedTemplateGenerator())[0]
    assert suggestion.citation == "Doe (2021), A Study, p. 3. DOI: 10/example"
    assert suggestion.evidence == "Evidence from page three"


def test_markdown_export_contains_evidence():
    suggestion = suggest_citations([_match()])[0]
    markdown = to_markdown([suggestion])
    assert "Evidence from page three" in markdown
    assert "The draft claim." in markdown


def test_markdown_export_groups_suggestion_details():
    suggestion = suggest_citations([_match()])[0]

    markdown = to_markdown([suggestion])

    assert "## Suggestion 1" in markdown
    assert "### Draft claim" in markdown
    assert "### Evidence" in markdown
    assert "### Rationale" in markdown
    assert "### Source" in markdown
    assert "**Support score:** `0.90`" in markdown


def test_suggestion_uses_only_verbatim_selected_evidence():
    class Extractor:
        def __init__(self, quote):
            self.quote = quote

        def extract(self, draft_text, source, context):
            return EvidenceSelection(True, self.quote, "The exact sentence supports the claim.")

    selected = suggest_citations(
        [_match()],
        evidence_extractor=Extractor("Exact supporting sentence."),
        context_provider=lambda chunk: "Before. Exact supporting sentence. After.",
    )[0]
    rejected = suggest_citations(
        [_match()],
        evidence_extractor=Extractor("Invented sentence."),
        context_provider=lambda chunk: "Before. Exact supporting sentence. After.",
    )[0]

    assert selected.evidence == "Exact supporting sentence."
    assert rejected.evidence == "Evidence from page three"


def test_evidence_extractor_sends_claim_source_and_context():
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = (
                    '{"supports_claim": true, "quote": "Evidence from page three", '
                    '"rationale": "It supports the claim."}'
                )

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    selection = OpenAIEvidenceExtractor("test-model", client=Client()).extract(
        "The draft claim.", _match().sources[0], "Nearby context."
    )

    assert selection.supports_claim is True
    assert calls[0]["model"] == "test-model"
    user_message = calls[0]["messages"][1]["content"]
    assert "The draft claim." in user_message
    assert "Nearby context." in user_message
