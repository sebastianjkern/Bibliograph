from bibliograph.cli import build_parser


def test_cli_requires_a_command():
    parser = build_parser()
    args = parser.parse_args(["index-pdf", "paper.pdf", "--key", "K1", "--title", "Study"])
    assert args.command == "index-pdf"
    assert args.title == "Study"
