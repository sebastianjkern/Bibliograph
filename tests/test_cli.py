from bibliograph.cli import build_parser


def test_cli_requires_a_command():
    parser = build_parser()
    args = parser.parse_args(["index-pdf", "paper.pdf", "--key", "K1", "--title", "Study"])
    assert args.command == "index-pdf"
    assert args.title == "Study"


def test_cli_check_accepts_draft_and_collection():
    args = build_parser().parse_args(["check", "draft.tex", "Methods", "--llm-model", "reranker"])
    assert args.command == "check"
    assert args.collection == "Methods"
    assert args.llm_model == "reranker"


def test_cli_has_local_first_defaults():
    args = build_parser().parse_args(["check", "draft.tex", "Methods"])
    assert args.embedding_provider == "sentence-transformers"
    assert args.download_missing is True
    assert args.download_source == "remote"
    assert args.llm_model == "qwen/qwen3-1.7b"
    assert args.output is None
