from pathlib import Path

import pytest

from bibliograph.settings import load_settings

_ENVIRONMENT = (
    "BIBLIOGRAPH_DB",
    "BIBLIOGRAPH_PDF_DIR",
    "BIBLIOGRAPH_COLLECTION",
    "BIBLIOGRAPH_ZOTERO_STORAGE_DIR",
    "BIBLIOGRAPH_ZOTERO_LIBRARY_ID",
    "BIBLIOGRAPH_ZOTERO_LIBRARY_TYPE",
    "BIBLIOGRAPH_ZOTERO_API_KEY",
    "BIBLIOGRAPH_EMBEDDING_PROVIDER",
    "BIBLIOGRAPH_EMBEDDING_MODEL",
    "BIBLIOGRAPH_EMBEDDING_BASE_URL",
    "BIBLIOGRAPH_EMBEDDING_API_KEY",
    "BIBLIOGRAPH_EMBEDDING_BATCH_SIZE",
    "BIBLIOGRAPH_EMBEDDING_CACHE_DIR",
    "BIBLIOGRAPH_EMBEDDING_OFFLINE",
    "BIBLIOGRAPH_EMBEDDING_DOCUMENT_PREFIX",
    "BIBLIOGRAPH_EMBEDDING_QUERY_PREFIX",
    "BIBLIOGRAPH_LLM_PROVIDER",
    "BIBLIOGRAPH_LLM_MODEL",
    "BIBLIOGRAPH_LLM_BASE_URL",
    "BIBLIOGRAPH_LLM_API_KEY",
    "BIBLIOGRAPH_LLM_MODE",
    "BIBLIOGRAPH_LLM_STAGES",
    "BIBLIOGRAPH_ACQUISITION_ORDER",
    "BIBLIOGRAPH_REMOTE_UNPAYWALL_EMAIL",
    "BIBLIOGRAPH_REMOTE_OPENALEX_API_KEY",
    "BIBLIOGRAPH_REMOTE_PLAYWRIGHT_PROFILE",
    "BIBLIOGRAPH_REMOTE_PLAYWRIGHT_HEADLESS",
    "BIBLIOGRAPH_PLAYWRIGHT_PROFILE",
    "BIBLIOGRAPH_PLAYWRIGHT_HEADLESS",
    "ZOTERO_STORAGE_DIR",
    "ZOTERO_LIBRARY_ID",
    "ZOTERO_LIBRARY_TYPE",
    "ZOTERO_API_KEY",
    "UNPAYWALL_EMAIL",
    "OPENALEX_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "LLM_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_LLM_MODEL",
    "EMBEDDING_MODEL",
    "SENTENCE_TRANSFORMERS_CACHE",
    "HF_HOME",
    "SENTENCE_TRANSFORMERS_OFFLINE",
)


def _clear_environment(monkeypatch):
    for variable in _ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)


def test_settings_precedence_is_override_environment_toml_then_defaults(tmp_path, monkeypatch):
    _clear_environment(monkeypatch)
    config = tmp_path / "bibliograph.toml"
    config.write_text(
        """
[profiles.research]
db = "from-toml.db"
collection = "Toml Collection"

[profiles.research.embedding]
provider = "hash"
model = "from-toml-model"
dimension = 48

[profiles.research.llm]
mode = "off"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("BIBLIOGRAPH_DB", "from-env.db")
    monkeypatch.setenv("BIBLIOGRAPH_EMBEDDING_MODEL", "from-env-model")
    monkeypatch.setenv("BIBLIOGRAPH_EMBEDDING_BATCH_SIZE", "16")

    settings = load_settings(
        config,
        "research",
        {"db": "from-cli.db", "embedding": {"model": "from-cli-model"}},
    )

    assert settings["db"] == "from-cli.db"
    assert settings["collection"] == "Toml Collection"
    assert settings["embedding"] == {
        "provider": "hash",
        "model": "from-cli-model",
        "base_url": "http://localhost:11434",
        "api_key": None,
        "batch_size": 16,
        "cache_dir": None,
        "local_files_only": False,
        "document_prefix": "",
        "query_prefix": "",
        "dimension": 48,
    }
    assert settings["llm"]["mode"] == "off"


def test_settings_selects_legacy_provider_credentials_and_zotero_remote_values(
    tmp_path, monkeypatch
):
    _clear_environment(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "legacy-embedding")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")
    monkeypatch.setenv("ZOTERO_API_KEY", "zotero-key")
    monkeypatch.setenv("ZOTERO_STORAGE_DIR", "D:/Zotero/storage")
    monkeypatch.setenv("UNPAYWALL_EMAIL", "reader@example.test")
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-key")
    monkeypatch.setenv("BIBLIOGRAPH_PLAYWRIGHT_PROFILE", "D:/Browser Profile")
    monkeypatch.setenv("BIBLIOGRAPH_PLAYWRIGHT_HEADLESS", "false")
    monkeypatch.setenv("BIBLIOGRAPH_ZOTERO_LIBRARY_ID", "generic-library")

    settings = load_settings(overrides={"embedding_provider": "openai"})

    assert settings["embedding"]["base_url"] == "http://openai.test/v1"
    assert settings["embedding"]["api_key"] == "legacy-key"
    assert settings["embedding"]["model"] == "legacy-embedding"
    assert settings["zotero"] == {
        "library_id": "generic-library",
        "library_type": "group",
        "api_key": "zotero-key",
        "storage_dir": "D:/Zotero/storage",
    }
    assert settings["zotero_storage_dir"] == "D:/Zotero/storage"
    assert settings["remote"] == {
        "email": "reader@example.test",
        "openalex_api_key": "openalex-key",
        "playwright_profile": "D:/Browser Profile",
        "playwright_headless": False,
    }


def test_settings_loads_default_profile_and_rejects_unknown_profile(tmp_path, monkeypatch):
    _clear_environment(monkeypatch)
    config = tmp_path / "bibliograph.toml"
    config.write_text("[profiles.default]\ndb = 'configured.db'\n", encoding="utf-8")

    assert load_settings(config)["db"] == "configured.db"
    with pytest.raises(ValueError, match="missing"):
        load_settings(config, profile="missing")


def test_explicit_missing_configuration_file_is_an_error(tmp_path, monkeypatch):
    _clear_environment(monkeypatch)

    with pytest.raises(FileNotFoundError, match="not found"):
        load_settings(Path(tmp_path / "does-not-exist.toml"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embedding": {"provider": "missing"}}, "embedding.provider"),
        ({"llm": {"provider": "hash"}}, "llm.provider"),
        ({"llm": {"stages": ["summarize"]}}, "llm.stages"),
        ({"acquisition": {"order": ["filesystem"]}}, "acquisition.order"),
    ],
)
def test_settings_rejects_unknown_pipeline_names(overrides, message, monkeypatch):
    _clear_environment(monkeypatch)

    with pytest.raises(ValueError, match=message):
        load_settings(overrides=overrides)


def test_evidence_classification_is_opt_in_and_requires_llm(monkeypatch):
    _clear_environment(monkeypatch)
    defaults = load_settings()
    assert defaults["classification"] == {"enabled": False, "model": None}

    with pytest.raises(ValueError, match="requires llm.mode"):
        load_settings(overrides={
            "classification": {"enabled": True},
            "llm": {"mode": "off"},
        })


def test_settings_accepts_stage_provider_and_model_overrides(monkeypatch):
    _clear_environment(monkeypatch)

    settings = load_settings(
        overrides={
            "llm": {
                "models": {
                    "expand": "local-model",
                    "rationale": {
                        "provider": "openai",
                        "model": "reasoning-model",
                        "base_url": "https://openai.example/v1",
                    },
                }
            }
        }
    )

    assert settings["llm"]["models"]["expand"] == "local-model"
    assert settings["llm"]["models"]["rationale"] == {
        "provider": "openai",
        "model": "reasoning-model",
        "base_url": "https://openai.example/v1",
    }


@pytest.mark.parametrize(
    "models",
    [
        {"rationale": {"provider": "missing", "model": "m"}},
        {"rationale": {"model": "  "}},
        {"rationale": {"unexpected": "value"}},
        {"rationale": 42},
    ],
)
def test_settings_rejects_invalid_stage_provider_overrides(models, monkeypatch):
    _clear_environment(monkeypatch)

    with pytest.raises(ValueError, match="llm.models"):
        load_settings(overrides={"llm": {"models": models}})


def test_settings_accepts_provider_aliases_and_resolves_browser_environment(monkeypatch):
    _clear_environment(monkeypatch)
    monkeypatch.setenv("BIBLIOGRAPH_REMOTE_PLAYWRIGHT_PROFILE", "C:/profile")
    monkeypatch.setenv("BIBLIOGRAPH_REMOTE_PLAYWRIGHT_HEADLESS", "0")

    settings = load_settings(overrides={"llm": {"provider": "openai-compatible"}})

    assert settings["llm"]["provider"] == "openai-compatible"
    assert settings["remote"]["playwright_profile"] == "C:/profile"
    assert settings["remote"]["playwright_headless"] is False
