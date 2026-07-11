# Bibliograph

Bibliograph is a local citation discovery assistant. It indexes papers from a Zotero-backed library, scans draft passages for semantic matches, and returns evidence-backed citation suggestions.

## Current State

This repository is an early-stage prototype rather than a finished product. It currently provides:

* `bibliograph/` — the reusable ingestion, chunking, embedding, retrieval, suggestion, export, and CLI modules.
* `app.py` — the original single-PDF proof-of-concept, retained for compatibility.
* `connections/zotero_collections.py` — helper functions for fetching Zotero collection metadata.
* `connections/zotero_items.py` — helper functions for fetching Zotero items and exporting them to CSV.
* `connections/zotero_pdfs.py` — helper functions for downloading PDF attachments from Zotero item metadata.
* `connections/reload_embeddings.py` — Zotero metadata lookup and single-item PDF downloading.

## What the project solves today

* Collects Zotero items and collection metadata from a configured Zotero library.
* Downloads one missing PDF attachment at a time from Zotero by DOI, title, or item key.
* Extracts text from local PDF files and splits it into searchable sentences.
* Stores page-aware chunk embeddings and complete citation metadata in a persistent SQLite index using sqlite-vec nearest-neighbor search.
* Supports deterministic local vectors, SentenceTransformer models, and OpenAI-compatible embedding APIs.
* Scans a draft paragraph-by-paragraph and returns matching evidence with scores.
* Formats grounded citation suggestions with author, year, DOI, and page information.

## Tutorial

The main workflow is:

1. Give Bibliograph a Typst or LaTeX draft.
2. Select a Zotero collection.
3. Download individual Zotero PDFs by DOI or title when needed.
4. Let Bibliograph incrementally index new or changed local PDFs.
5. Retrieve and optionally LLM-rerank evidence for each draft claim.

Bibliograph never downloads an entire Zotero collection automatically.

### Setup

Install the package and the tools used by the complete workflow:

```bash
uv sync --extra dev --extra pdf --extra zotero --extra embeddings --extra llm
```

Copy the environment template:

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead.

### Configure Zotero

Edit `.env`:

```env
ZOTERO_LIBRARY_ID=your_library_id
ZOTERO_LIBRARY_TYPE=user
ZOTERO_API_KEY=your_api_key
```

Use `ZOTERO_LIBRARY_TYPE=group` for a group library. The API key must have read access to the library.

### Create a draft

Bibliograph supports `.tex`, `.latex`, and `.typ` files. For example, create `example.tex`:

```latex
\\section{Background}

Bayesian methods improve uncertainty estimates in small-data settings.
This claim should be supported by a paper in the Zotero collection.
```

Or create `example.typ`:

```typst
= Background

Bayesian methods improve uncertainty estimates in small-data settings.
This claim should be supported by a paper in the Zotero collection.
```

Existing LaTeX citations such as `\\citep{smith2022}` and Typst citations such as `@smith2022` are preserved in the parsed claim metadata.

### Choose an embedding backend

The default hash backend is deterministic and useful for testing, but it is not a meaningful semantic model:

```bash
uv run bibliograph check example.tex "My Collection"
```

For local semantic embeddings, use SentenceTransformers:

```bash
uv run bibliograph --embedding-provider sentence-transformers --model all-MiniLM-L6-v2 check example.tex "My Collection"
```

SentenceTransformers models are cached persistently. Set a custom cache location with `SENTENCE_TRANSFORMERS_CACHE` or `--embedding-cache-dir`:

```bash
uv run bibliograph --embedding-provider sentence-transformers --embedding-cache-dir .cache/models check example.tex "My Collection"
```

After the model has been downloaded once, prevent all Hugging Face network checks with offline mode:

```bash
uv run bibliograph --embedding-provider sentence-transformers --embedding-cache-dir .cache/models --embedding-offline check example.tex "My Collection"
```

The same setting can be enabled in `.env` with `SENTENCE_TRANSFORMERS_OFFLINE=true`. Offline mode requires the requested model to already exist in the cache.

For OpenAI or another OpenAI-compatible embeddings API, set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` in `.env`:

```bash
uv run bibliograph --embedding-provider openai --model text-embedding-3-small check example.tex "My Collection"
```

A custom endpoint can be selected with `--embedding-base-url`.

### Download one Zotero PDF

Download by DOI, without finding a Zotero parent key manually:

```bash
uv run bibliograph download-pdf --doi 10.1234/example --output-dir pdfs
```

Or use the exact Zotero title:

```bash
uv run bibliograph download-pdf --title "Paper title" --output-dir pdfs
```

The command resolves the Zotero parent item internally, finds its PDF attachment, and skips the download if that attachment is already local. If multiple papers match a title, use the DOI or the optional parent item key. If one item has multiple PDFs, select one with `--attachment-key`.

### Run the complete citation check

Run the end-to-end workflow:

```bash
uv run bibliograph check example.tex "My Collection" --pdf-dir pdfs --output citation-check.md
```

The command:

* checks the selected Zotero collection;
* detects new or changed attachments using Zotero versions and local file hashes;
* indexes only local PDFs that need indexing;
* reports Zotero PDFs that are not downloaded locally;
* parses the draft into claims;
* combines semantic and lexical retrieval;
* returns the closest evidence passages; and
* optionally reranks candidates with an LLM.

It does not download PDFs. Use the single-item command above for each missing paper, then rerun `check`.

To let `check` download missing attachments through the same single-item Zotero backend, opt in explicitly:

```bash
uv run bibliograph check example.tex "My Collection" --pdf-dir pdfs --download-missing
```

Missing-paper names, DOIs, and available attachment keys are printed in the report. Papers without an available attachment key are still listed by title and DOI; the backend resolves them through the parent Zotero item when possible.

### Enable LLM reranking

Set `OPENAI_API_KEY` and, if needed, `OPENAI_BASE_URL` in `.env`, then provide a chat model:

```bash
uv run bibliograph check example.tex "My Collection" --pdf-dir pdfs --llm-model your-reranker-model --output citation-check.md
```

The reranker receives the draft claim and retrieved evidence passages. It returns structured support scores and reorders the candidates; it does not replace the source evidence.

### Review the output

Open `citation-check.md`. Each suggestion includes:

* the draft claim;
* the suggested paper and citation metadata;
* the page number when available;
* the extracted evidence passage;
* a support score; and
* a short rationale.

Always verify the evidence before inserting a citation.

### Rerun after adding papers

After adding papers to Zotero:

1. Download each needed PDF with `download-pdf`.
2. Run the same `check` command again.

The SQLite index records attachment versions and file hashes, so unchanged papers are skipped. Use a separate database with `--db project-bibliography.db` when working on multiple projects.

### Directly index a local PDF

If a PDF is already available locally, it can be indexed directly:

```bash
uv run bibliograph index-pdf paper.pdf --key ABC123 --title "Paper title" --author "Author surname" --year 2024 --doi 10.1234/example
```

For plain text drafts, the simpler `suggest` command is also available:

```bash
uv run bibliograph suggest draft.txt --output suggestions.md
```

### Troubleshooting

* **Collection not found:** the collection name must match exactly; use its Zotero collection key if necessary.
* **No matches:** confirm that PDFs exist in `pdfs`, use a semantic embedding backend, and lower `--min-score` if one was provided.
* **Authentication errors:** verify the Zotero library ID, library type, and API key permissions.
* **Scanned PDFs:** PDFs must contain extractable text; OCR is not currently included.
* **Start over:** use a new database with `--db fresh-index.db` rather than deleting an existing index.

### Original prototype

```bash
uv run python app.py
```

The script currently opens `test.pdf`, extracts sentences, stores them in a local Chroma collection, and lets you query them interactively.

## Current limitations

The repository is not yet a complete discovery system. Remaining issues include:

* Zotero synchronization currently indexes local files and reports missing PDFs; downloading remains an explicit one-paper-at-a-time action.
* OCR for scanned PDFs and structural section extraction are not yet included.
* SQLite/sqlite-vec is optimized for local, single-user workloads; a dedicated vector service may be preferable for multi-user deployments.
* No browser UI is included yet; Markdown export is the current review workflow.

## Future direction

The next steps for the project are:

* add OCR and better academic section/chunk extraction
* benchmark sqlite-vec settings and add optional dedicated vector backends for larger deployments
* add a lightweight UI or API server for review and citation insertion

## Why this matters

Traditional citation-based discovery is useful, but it misses many semantic relationships between papers. This repo aims to build a pipeline that can support:

* semantic search over paper content
* retrieval of relevant documents even when they are not directly cited
* a Zotero-backed ingestion workflow for personal research libraries

## License

All rights remain with the author.
