from bibliograph.adapters.context import join_chunk_context


def test_context_joins_overlapping_chunks_without_repeated_section_labels():
    context = join_chunk_context(
        [
            (0, "Section: Results\n\nThe intervention increased the outcome in the sample."),
            (1, "Section: Results\n\ncreased the outcome in the sample. Effects varied by region."),
        ],
        center_ordinal=1,
        max_words=100,
    )

    assert context.count("increased the outcome") == 1
    assert "Section:" not in context
    assert context.endswith("Effects varied by region.")


def test_context_word_limit_keeps_window_centered_on_matched_chunk():
    before = " ".join(f"before{i}" for i in range(30))
    target = "The exact finding is reported here."
    after = " ".join(f"after{i}" for i in range(30))

    context = join_chunk_context(
        [(0, before), (1, target), (2, after)],
        center_ordinal=1,
        max_words=20,
    )

    assert len(context.split()) == 20
    assert "exact finding" in context
