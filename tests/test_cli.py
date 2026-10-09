from pathlib import Path

from bibliograph.cli import _disabled_llm_stages, _settings_overrides, build_parser, main
from bibliograph.commands.search import _stage_progress


def test_cli_exposes_the_four_primary_commands():
    parser = build_parser()

    assert parser.parse_args(["sync", "Methods"]).command == "sync"
    assert parser.parse_args(["ingest", "papers"]).command == "ingest"
    assert parser.parse_args(["search", "Road quality affects trade"]).command == "search"
    assert parser.parse_args(["check", "draft.tex"]).command == "check"
    assert parser.parse_args(["status"]).command == "status"
    assert parser.parse_args(["sync", "--reset-pdf-index"]).reset_pdf_index is True
    assert parser.parse_args(["sync", "--reset"]).reset_pdf_index is True
    assert parser.parse_args(["ingest-pdfs", "papers", "--reset-pdf-index"]).reset_pdf_index
    assert parser.parse_args(["ingest", "papers", "--reset"]).reset_pdf_index


def test_help_reports_the_resolved_provider_configuration(monkeypatch, capsys):
    monkeypatch.setattr(
        "bibliograph.cli.load_settings",
        lambda *_args: {
            "db": "index.db",
            "pdf_dir": "papers",
            "collection": "Research",
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text-v2-moe:latest",
                "base_url": "http://localhost:11434",
            },
            "llm": {
                "provider": "ollama",
                "model": "edtorre/gemma4:12qat-hermes",
                "base_url": "http://localhost:11434",
                "mode": "optional",
                "stages": ["rerank", "evidence"],
            },
        },
    )

    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "nomic-embed-text-v2-moe:latest" in output
    assert "edtorre/gemma4:12qat-hermes" in output
    assert "ollama" in output
    assert "rerank, evidence" in output


def test_search_progress_tracks_async_batch_completion_fractionally():
    assert _stage_progress("Expanding queries · 2/4 complete") == 0.5
    assert _stage_progress("Embedding expanded queries") == 2.0
    assert _stage_progress("Retrieving evidence · 3/5 queries") == 2.4
    assert _stage_progress("Reranking candidates · 2/4 batches complete") == 3.5
    assert _stage_progress("Enriching finalists · 3/4 complete") == 4.75


def test_cli_keeps_read_workflow_shims_for_one_release():
    parser = build_parser()

    assert parser.parse_args(["find", "claim"]).command == "legacy-search"
    assert parser.parse_args(["find-sources", "claim"]).command == "legacy-search"
    assert parser.parse_args(["suggest", "draft.tex"]).command == "legacy-suggest"
    legacy_search = parser.parse_args(
        ["find", "claim", "--no-rerank", "--no-evidence-extraction"]
    )
    assert legacy_search.no_rerank is True
    assert legacy_search.no_evidence_extraction is True
    no_enrichment = parser.parse_args(["search", "claim", "--no-enrichment"])
    assert no_enrichment.no_enrichment is True
    assert _disabled_llm_stages(no_enrichment) == ("evidence", "rationale")
    assert (
        parser.parse_args(["suggest", "draft.tex", "--no-evidence-extraction"])
        .no_evidence_extraction
    )


def test_cli_converts_runtime_flags_to_settings_overrides():
    args = build_parser().parse_args(
        [
            "--db",
            "project.db",
            "--embedding-provider",
            "hash",
            "--embedding-batch-size",
            "16",
            "--llm-mode",
            "off",
            "search",
            "claim",
        ]
    )

    assert _settings_overrides(args) == {
        "db": "project.db",
        "embedding_provider": "hash",
        "embedding_batch_size": 16,
        "llm_mode": "off",
    }


def test_main_dispatches_search_without_constructing_sync_dependencies(monkeypatch):
    captured = {}

    monkeypatch.setattr("bibliograph.cli.load_settings", lambda *args: {"loaded": True})

    def fake_search(settings, claim, **kwargs):
        captured.update(settings=settings, claim=claim, kwargs=kwargs)
        return {"markdown": "# Search"}

    monkeypatch.setattr("bibliograph.cli.run_search", fake_search)

    assert (
        main(
            [
                "--quiet",
                "search",
                "a test claim",
                "--no-llm",
                "--no-rerank",
                "--no-evidence-extraction",
            ]
        )
        == 0
    )
    assert captured == {
        "settings": {"loaded": True},
        "claim": "a test claim",
        "kwargs": {
            "limit": 5,
            "min_score": 0.0,
            "no_llm": True,
            "disabled_stages": ("rerank", "evidence"),
            "show_progress": False,
            "enrich": True,
            "one_per_paper": False,
        },
    }


def test_legacy_check_explicitly_syncs_before_read_only_check(monkeypatch):
    calls = []
    monkeypatch.setattr("bibliograph.cli.load_settings", lambda *args: {"loaded": True})
    monkeypatch.setattr(
        "bibliograph.cli.run_sync",
        lambda settings, **kwargs: calls.append(("sync", kwargs)) or {"indexed": []},
    )
    monkeypatch.setattr(
        "bibliograph.cli.run_check",
        lambda settings, draft, **kwargs: calls.append(("check", {"draft": draft, **kwargs}))
        or {"markdown": "# Check"},
    )

    assert main(["--quiet", "check", "draft.tex", "Methods", "--rebuild-db"]) == 0
    assert calls == [
        (
            "sync",
            {
                "collection": "Methods",
                "rebuild": True,
                "show_progress": False,
            },
        ),
        (
            "check",
            {
                "draft": Path("draft.tex"),
                "limit": 5,
                "min_score": 0.0,
                "no_llm": False,
                "disabled_stages": (),
                "show_progress": False,
                "enrich": True,
            },
        ),
    ]


def test_sync_accepts_legacy_rebuild_flag_after_the_subcommand(monkeypatch):
    calls = []
    monkeypatch.setattr("bibliograph.cli.load_settings", lambda *args: {"loaded": True})
    monkeypatch.setattr(
        "bibliograph.cli.run_sync",
        lambda settings, **kwargs: calls.append((settings, kwargs))
        or {
            "collection": "Methods",
            "discovered": 0,
            "indexed": [],
            "unchanged": [],
            "empty": [],
            "missing": [],
            "downloaded": [],
        },
    )

    assert main(["--quiet", "sync", "Methods", "--reset-pdf-index"]) == 0
    assert calls == [
        (
            {"loaded": True},
            {
                "collection": "Methods",
                "rebuild": False,
                "show_progress": False,
                "force_reindex": True,
            },
        )
    ]


def test_read_commands_reject_legacy_database_rebuild_flag(monkeypatch):
    monkeypatch.setattr("bibliograph.cli.load_settings", lambda *args: {})

    assert main(["--quiet", "--rebuild-db", "search", "claim"]) == 1
