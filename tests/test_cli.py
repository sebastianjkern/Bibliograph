from bibliograph.cli import _llm_model, build_parser, main
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


def test_cli_accepts_direct_local_source_lookup():
    args = build_parser().parse_args(["find-sources", "causal effects of roads", "--limit", "3"])

    assert args.command == "find-sources"
    assert args.claim == "causal effects of roads"
    assert args.limit == 3
    assert args.llm_model == "essentialai/rnj-1"
    assert args.no_rerank is False


def test_cli_has_local_first_defaults():
    args = build_parser().parse_args(["check", "draft.tex", "Methods"])
    assert args.embedding_provider == "sentence-transformers"
    assert args.download_missing is True
    assert args.download_source == "remote"
    assert args.llm_model == "essentialai/rnj-1"
    assert args.output is None


def test_cli_accepts_native_ollama_backends():
    args = build_parser().parse_args(
        [
            "--embedding-provider",
            "ollama",
            "--model",
            "nomic-embed-text",
            "--llm-provider",
            "ollama",
            "find-sources",
            "local models",
        ]
    )

    assert args.embedding_provider == "ollama"
    assert args.llm_provider == "ollama"
    assert _llm_model(args) == "llama3.2"


def test_cli_reports_manual_hint_for_failed_remote_download(monkeypatch):
    def fail(*args, **kwargs):
        raise RemoteDownloadError(
            "10/example",
            [ManualDownloadHint("https://publisher.example/article", "pydoi", "manual")],
            "download failed",
        )

    monkeypatch.setattr("bibliograph.remote.download_remote_pdf", fail)

    assert main(["download-pdf", "--source", "remote", "--doi", "10/example"]) == 1
