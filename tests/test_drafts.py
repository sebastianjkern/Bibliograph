from bibliograph.drafts import detect_format, parse_draft


def test_latex_parser_preserves_citations_and_removes_markup():
    claims = parse_draft(
        "\\section{Background}\nThis is a meaningful claim about methods "
        "\\citep{smith2022,doe2023}.",
        "latex",
    )
    assert claims[0].text == "Background This is a meaningful claim about methods ."
    assert claims[0].citation_keys == ("smith2022", "doe2023")


def test_typst_parser_preserves_at_citations():
    claims = parse_draft("This is a meaningful claim about methods @smith2022.", "typst")
    assert claims[0].citation_keys == ("smith2022",)
    assert "@smith2022" not in claims[0].text


def test_detect_format_uses_extension():
    assert detect_format("draft.tex") == "latex"
    assert detect_format("draft.typ") == "typst"
