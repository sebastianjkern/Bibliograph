"""User-facing defaults for the Bibliograph CLI and model adapters."""

import os

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_BASE_URL = "http://localhost:1234/v1"
DEFAULT_OPENAI_API_KEY = "lm-studio"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_LLM_MODEL = "qwen/qwen3-1.7b"
EMBEDDING_CONTENT_VERSION = "structured-pdf-v1"


def openai_settings() -> tuple[str, str]:
    """Return non-empty OpenAI-compatible credentials and endpoint settings."""
    return (
        os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY,
        os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
    )
