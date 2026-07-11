from bibliograph.export import to_markdown
from bibliograph.models import CitationSource, DraftMatch, Paper, TextChunk
from bibliograph.suggestions import GroundedTemplateGenerator, suggest_citations


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
