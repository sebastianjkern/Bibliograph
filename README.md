![Bibliograph-Banner](./github-banner.svg)

# Bibliograph

Bibliograph helps find citation evidence in a local Zotero-backed paper library. It supports LaTeX and Typst drafts, semantic search, local LLM reranking, legal PDF retrieval, and a persistent SQLite/sqlite-vec index.

## Setup

Bibliograph requires Python 3.12 or newer.

```bash
uv python install 3.12
uv sync
uv run playwright install chromium
```

Create the environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## Configure Zotero

Set these values in `.env`:

```env
ZOTERO_LIBRARY_ID=your_library_id
ZOTERO_LIBRARY_TYPE=user
ZOTERO_API_KEY=your_api_key
```

Use `ZOTERO_LIBRARY_TYPE=group` for a group library.

Bibliograph automatically checks common Zotero profile locations for local PDFs. If needed, configure the storage directory explicitly:

```env
ZOTERO_STORAGE_DIR=C:\path\to\Zotero\storage
```

Local PDFs are imported before any API or remote download is attempted.

## Configure LM Studio

Start the LM Studio server and load:

- embedding model: `text-embedding-nomic-embed-text-v1.5`
- reranking model: `essentialai/rnj-1`

The defaults are:

```env
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://localhost:1234/v1
```

For another OpenAI-compatible server, change `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

## Common workflows

Create a draft such as `example.tex` or `example.typ`.

### Local check without updating the index

Use the existing local vector index and do not synchronize Zotero or download PDFs:

```bash
uv run bibliograph suggest example.tex --no-llm
```

The command searches the current `bibliograph.db` index. Omit `--no-llm` to generate LLM-assisted rationales.

### Local check with index update

Synchronize one Zotero collection, import local Zotero PDFs, index new or changed files, and check the draft:

```bash
uv run bibliograph check example.tex "My Collection"
```

By default, missing paper PDFs may also be retrieved through legal remote sources. To update only from local files:

```bash
uv run bibliograph check example.tex "My Collection" --no-download-missing
```

Reports are printed to the console by default. Save one with:

```bash
uv run bibliograph check example.tex "My Collection" --output citation-check.md
```

### Search one claim without updating the index

```bash
uv run bibliograph find-sources \
  "Road infrastructure improves regional market access" \
  --limit 5
```

This searches only the existing local index and reranks the results with the configured LLM. Use `--no-rerank` for deterministic retrieval:

```bash
uv run bibliograph find-sources "Road infrastructure improves regional market access" --no-rerank
```

`find` is also accepted as a shorter command name.

## Useful commands

Index a local PDF manually:

```bash
uv run bibliograph index-pdf paper.pdf \
  --key PAPER1 \
  --title "Paper title" \
  --author "Author surname" \
  --year 2024 \
  --doi 10.1234/example
```

Download one Zotero attachment by DOI or title:

```bash
uv run bibliograph download-pdf --source zotero --doi 10.1234/example
```

Download one paper from legal remote sources:

```bash
uv run bibliograph download-pdf --source remote --doi 10.1234/example
```

Use a different database for another project:

```bash
uv run bibliograph --db project-bibliography.db check example.tex "My Collection"
```

## Embedding options

The default embedding backend is the cached `all-MiniLM-L6-v2` SentenceTransformer model:

```bash
uv run bibliograph check example.tex "My Collection"
```

Use an OpenAI-compatible embedding server instead:

```bash
uv run bibliograph \
  --embedding-provider openai \
  --model text-embedding-nomic-embed-text-v1.5 \
  find-sources "Climate predictability depends on slowly varying components"
```

Changing the embedding model or structured PDF-processing version automatically clears incompatible vectors so the local PDFs can be reindexed.

## Current features

- Incremental Zotero collection synchronization.
- Automatic import from Zotero’s local PDF storage.
- PDF validation and legal remote download fallbacks.
- Playwright support for publisher pages requiring browser access.
- PyMuPDF-based cleanup of repeated headers, footers, page numbers, and references.
- Section-aware, sentence-based embedding chunks.
- SentenceTransformer, deterministic hash, and OpenAI-compatible embeddings.
- Persistent SQLite/sqlite-vec vector search.
- LLM-assisted reranking with a deterministic fallback.
- LaTeX, Typst, and plain-text draft search.

## Code structure

- `bibliograph/cli.py` — command-line workflows.
- `bibliograph/ingest.py` — PDF extraction, cleanup, and indexing.
- `bibliograph/chunking.py` — section-aware text chunking.
- `bibliograph/embeddings.py` — embedding backends.
- `bibliograph/store.py` — persistent SQLite/sqlite-vec index.
- `bibliograph/retrieval.py` — semantic and hybrid retrieval.
- `bibliograph/reranker.py` — LLM and heuristic reranking.
- `bibliograph/remote.py` — DOI, legal remote, and Playwright PDF retrieval.
- `bibliograph/zotero_sync.py` — Zotero synchronization and local-storage imports.
- `connections/` — Zotero client and attachment helpers.
- `tests/` — unit and integration tests.

## Validation

Run the checks with:

```bash
uv run ruff check .
uv run pytest
```

Always verify retrieved evidence before inserting a citation into a paper.
