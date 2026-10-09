from bibliograph.render import _highlight_matched_excerpt
from bibliograph.text_matching import contains_text, find_text_span


def test_matching_tolerates_pdf_linebreak_hyphenation_and_preserves_original_span():
    source = "Conflict raises maize prices along trans-\nportation routes."
    quote = "Conflict raises maize prices along transportation routes."

    span = find_text_span(source, quote)

    assert span is not None
    assert source[span[0] : span[1]] == "Conflict raises maize prices along trans-\nportation routes."
    assert contains_text(source, quote)
    assert _highlight_matched_excerpt(source, quote) == (
        "⟦highlight⟧Conflict raises maize prices along trans-\n"
        "portation routes.⟦/highlight⟧"
    )


def test_matching_is_case_and_whitespace_insensitive_but_requires_full_quote():
    source = "The estimate is statistically significant."

    assert contains_text(source, "  THE estimate is\n statistically significant. ")
    assert not contains_text(source, "estimate is significant")
