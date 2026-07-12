"""User-facing defaults for the Bibliograph CLI and model adapters."""

import os

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_BASE_URL = "http://localhost:1234/v1"
DEFAULT_OPENAI_API_KEY = "lm-studio"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
DEFAULT_LLM_MODEL = "essentialai/rnj-1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.2"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_CONTENT_VERSION = "structured-pdf-v1"


def openai_settings() -> tuple[str, str]:
    """Return non-empty OpenAI-compatible credentials and endpoint settings."""
    return (
        os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY,
        os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
    )


def ollama_base_url() -> str:
    """Return the native Ollama API endpoint."""
    return (os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
