from bibliograph.cli import build_parser, main
from bibliograph.remote import ManualDownloadHint, RemoteDownloadError


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


def test_cli_reports_manual_hint_for_failed_remote_download(monkeypatch):
    def fail(*args, **kwargs):
        raise RemoteDownloadError(
            "10/example",
            [ManualDownloadHint("https://publisher.example/article", "pydoi", "manual")],
            "download failed",
        )

    monkeypatch.setattr("bibliograph.remote.download_remote_pdf", fail)

    assert main(["download-pdf", "--source", "remote", "--doi", "10/example"]) == 1
