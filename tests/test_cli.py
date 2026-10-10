from pathlib import Path

from bibliograph.cli import (
    _disabled_llm_stages,
    _emit,
    _llm_stage_overview,
    _settings_overrides,
    build_parser,
    main,
)
from bibliograph.commands.search import _assessment_progress_text, _progress_description


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


def test_overview_resolves_each_stage_provider_and_active_state():
    llm = {
        "provider": "ollama",
        "model": "shared-model",
        "mode": "optional",
        "stages": ["expand", "evidence"],
        "models": {
            "rationale": {"provider": "codex", "model": "gpt-6-luna"},
            "evidence": "evidence-model",
        },
    }

    overview = _llm_stage_overview(llm)

    assert overview == [
        ("Query expansion", "ollama", "shared-model", True),
        ("Reranking", "ollama", "shared-model", False),
        ("Evidence selection", "ollama", "evidence-model", True),
        ("Rationale", "codex", "gpt-6-luna", False),
    ]


def test_overview_marks_all_stages_inactive_when_llm_is_off():
    overview = _llm_stage_overview(
        {
            "provider": "ollama",
            "model": "shared-model",
            "mode": "off",
            "stages": ["rationale"],
            "models": {"rationale": {"provider": "codex", "model": "gpt-6-luna"}},
        }
    )

    assert all(not enabled for _, _, _, enabled in overview)
    assert overview[-1][:3] == ("Rationale", "codex", "gpt-6-luna")


def test_assessment_progress_colors_relations_and_keeps_other_text_neutral():
    progress = _assessment_progress_text(
        "10 passages · 6 support, 1 contradict, 2 mixed, 1 unresolved",
        " · round 1 · 4 passages updated",
    )

    assert "10 passages" in progress.plain
    assert "round 1" in progress.plain
    styled_segments = [
        (progress.plain[span.start : span.end], span.style)
        for span in progress.spans
    ]
    assert any(segment == "6" and "green" in str(style) for segment, style in styled_segments)
    assert any(segment == "1" and "red" in str(style) for segment, style in styled_segments)
    assert any(segment == "2" and "yellow" in str(style) for segment, style in styled_segments)
    assert any(segment == "1" and "cyan" in str(style) for segment, style in styled_segments)
    assert any(
        "10 passages" in segment and "dim" in str(style)
        for segment, style in styled_segments
    )


def test_report_output_file_strips_terminal_color_markers(tmp_path):
    output = tmp_path / "report.md"
    _emit(
        "Coverage: ⟦support⟧2 papers with supporting⟦/support⟧ and "
        "⟦contradict⟧1 with contradicting⟦/contradict⟧.",
        output,
    )

    rendered = output.read_text(encoding="utf-8")
    assert rendered == "Coverage: 2 papers with supporting and 1 with contradicting."
    assert "⟦" not in rendered


def test_search_progress_uses_concise_workflow_state_labels():
    assert _progress_description("Plan query · initial claim search") == "Searching the claim"
    assert _progress_description("Retrieved candidates · 36 unique · 4 query variant(s)") == (
        "Retrieved · 36 unique · 4 query variant(s)"
    )
    assert _progress_description(
        "Context extension · round 1 · 5 passage(s) updated"
    ) == "Extending local context · round 1 · 5 passage(s) updated"
    assert _progress_description(
        "Query extension · round 2 · 4 new queries · assessed evidence ledger"
    ) == "Planning follow-up queries · round 2 · 4 new queries · assessed evidence ledger"



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
    assert parser.parse_args(["search", "claim", "--verbose"]).verbose is True
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
            "verbose": False,
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
