![Bibliograph banner](./github-banner.svg)

# Bibliograph

Bibliograph finds grounded citation evidence in a local Zotero-backed paper library. It synchronizes PDFs into a SQLite/sqlite-vec index, searches claims semantically, and can use an LLM for reranking, quote selection, and short rationales.

The project is organized around one configured pipeline selected at startup: provider adapters are independent from retrieval, indexing, persistence, and CLI parsing. A provider never reads command-line arguments or environment variables itself.

### Example:

![Bibliograph example](./image.png)

## Install or reinstall

Bibliograph requires Python 3.12 or newer and uses [uv](https://docs.astral.sh/uv/).

```bash
uv python install 3.12
uv sync --extra all --extra dev
```

To repair or reinstall the local development environment after pulling changes, run:

```bash
uv sync --extra all --extra dev --reinstall
```

`uv run bibliograph ...` always uses the project environment. The `all` extra installs PDF extraction, Zotero, local embedding, OpenAI-compatible, and remote-acquisition integrations; omit it and choose only the extras you need for a smaller installation. The browser fallback for remote PDF acquisition additionally needs:

```bash
uv run playwright install chromium
```

## Configure a profile

Copy the example profile, then adjust its collection, paths, and model choices:

```bash
cp bibliograph.toml.example bibliograph.toml
```

On Windows PowerShell:

```powershell
Copy-Item bibliograph.toml.example bibliograph.toml
```

`bibliograph.toml` contains non-secret, semi-static choices. Profiles make it straightforward to keep separate local and hosted configurations:

```toml
[profiles.default]
db = "bibliograph.db"
pdf_dir = "pdfs"
collection = "My Collection"

[profiles.default.embedding]
provider = "ollama"
model = "nomic-embed-text"
batch_size = 32

[profiles.default.llm]
provider = "ollama"
model = "rnj-1"
mode = "optional" # off, optional, or required
stages = ["rerank", "evidence", "rationale"]

[profiles.default.acquisition]
order = ["cache", "zotero-storage", "zotero-api", "remote"]
```

Resolution is explicit: command-line override, then environment variable, then selected TOML profile, then built-in default. Put CLI overrides before the command, for example:

```bash
uv run bibliograph --profile local --embedding-model nomic-embed-text sync
```

## Secrets and Zotero

Copy `.env.example` to `.env` and fill in the secrets that apply to your chosen adapters. The generic `BIBLIOGRAPH_*` variables are preferred; established `ZOTERO_*`, `OPENAI_*`, `OLLAMA_*`, and remote-acquisition variables remain supported for one transition release.

```env
BIBLIOGRAPH_ZOTERO_LIBRARY_ID=your_library_id
BIBLIOGRAPH_ZOTERO_LIBRARY_TYPE=user
BIBLIOGRAPH_ZOTERO_API_KEY=your_api_key
BIBLIOGRAPH_REMOTE_UNPAYWALL_EMAIL=you@example.org
```

Use `BIBLIOGRAPH_ZOTERO_LIBRARY_TYPE=group` for a group library. Bibliograph checks configured and common Zotero storage locations before trying the Zotero API or legal remote sources.

For the default Ollama profile, start Ollama and pull the selected models:

```bash
ollama pull nomic-embed-text
ollama pull rnj-1
```

OpenAI-compatible servers use `provider = "openai"` with `base_url` and the relevant API-key environment variables. Local SentenceTransformers embeddings use `provider = "sentence-transformers"`; the deterministic `hash` embedding provider is useful for offline smoke tests.

## Commands

Only `sync` changes the database. The read commands never create, migrate, clear, or update an index.

```bash
# Incrementally synchronize the configured or named Zotero collection.
uv run bibliograph sync
uv run bibliograph sync "My Collection"

# Build a sibling staging database, validate it, and atomically replace the live one.
uv run bibliograph sync --rebuild

# Search the existing index for a claim.
uv run bibliograph search "Road infrastructure improves regional market access"

# Check a LaTeX or Typst draft using the existing index.
uv run bibliograph check example.tex

# Inspect database contents without loading providers or Zotero.
uv run bibliograph status

# Also test configured embedding/chat providers (and report diagnostics).
uv run bibliograph status --probe
```

`status --probe` also checks Zotero access. `search` and `check` accept `--limit`, `--min-score`, `--no-llm`, `--no-rerank`, `--no-evidence-extraction`, and `--output`. Use `--no-llm` to keep all retrieval decisions deterministic. `sync` reports embedding batches as they are stored, and validates the exact embedding model before a rebuild starts. If an embedding provider cannot be reached, the live database remains untouched. Optional LLM failures are logged and fall back deterministically; `llm.mode = "required"` makes them command errors instead.

Each database has one active text embedding fingerprint. A changed provider, model, or preprocessing setup produces rebuild guidance rather than silently clearing vectors. Schema-v1 indexes are derived data and should be replaced with `sync --rebuild`.

For one compatibility release, `find-sources`/`find` delegate to `search`, `suggest` delegates to `check`, and `check DRAFT COLLECTION` runs a sync before checking. These shims issue deprecation warnings. The old global `--rebuild-db` spelling is also retained only for legacy use; prefer `sync --rebuild`.

## Architecture

```text
bibliograph/
  cli.py          argument parsing, dispatch, exit codes
  bootstrap.py    composition root and dependency lifetimes
  settings.py     TOML, environment, and CLI resolution
  domain.py       Paper and Chunk durable records
  providers/      explicit embedding/chat factory registries
  adapters/       SQLite, Zotero, PDF, and acquisition adapters
  pipeline/       indexing, retrieval, LLM tasks, and draft parsing
  commands/       sync, search, check, and status use cases
  render.py       pure Markdown rendering
```

Adding a provider means adding one closure-based adapter, registering its factory, and testing its contract. Command handlers and pipeline tasks do not branch on provider names.

## Development checks

```bash
uv run ruff check .
uv run pytest
```

Always verify retrieved evidence before inserting a citation into a manuscript.
