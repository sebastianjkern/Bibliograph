# Bibliograph

Bibliograph is a local citation discovery assistant. It indexes papers from a Zotero-backed library, scans draft passages for semantic matches, and returns evidence-backed citation suggestions.

## Current State

This repository is an early-stage prototype rather than a finished product. It currently provides:

* `bibliograph/` — the reusable ingestion, chunking, embedding, retrieval, suggestion, export, and CLI modules.
* `app.py` — the original single-PDF proof-of-concept, retained for compatibility.
* `connections/zotero_collections.py` — helper functions for fetching Zotero collection metadata.
* `connections/zotero_items.py` — helper functions for fetching Zotero items and exporting them to CSV.
* `connections/zotero_pdfs.py` — helper functions for downloading PDF attachments from Zotero item metadata.
* `connections/reload_embeddings.py` — a new workflow to download all available PDF attachments for items in a specified Zotero collection.

## What the project solves today

* Collects Zotero items and collection metadata from a configured Zotero library.
* Downloads PDF attachments from Zotero items in a given collection.
* Extracts text from local PDF files and splits it into searchable sentences.
* Stores page-aware chunk embeddings and complete citation metadata in a persistent SQLite index.
* Supports deterministic local vectors, SentenceTransformer models, and OpenAI-compatible embedding APIs.
* Scans a draft paragraph-by-paragraph and returns matching evidence with scores.
* Formats grounded citation suggestions with author, year, DOI, and page information.

## How to use the current tools

### Setup

Install the core package and development tools:

```bash
uv sync --extra dev
```

For PDF indexing, install the PDF extra. For a local embedding model, install the embeddings extra:

```bash
uv sync --extra pdf --extra embeddings
```

Copy `.env.example` to `.env` when using Zotero or an OpenAI-compatible model.

### Index a PDF

The CLI uses deterministic local vectors by default, which is useful for testing. Pass `--model all-MiniLM-L6-v2` with `--embedding-provider sentence-transformers` after installing the embeddings extra for semantic embeddings.

```bash
bibliograph index-pdf paper.pdf --key ABC123 --title "Paper title" \
  --author "Author surname" --year 2024 --doi 10.1234/example
```

To use OpenAI or another OpenAI-compatible embeddings endpoint:

```bash
bibliograph --embedding-provider openai \
  --model text-embedding-3-small \
  --embedding-base-url https://api.openai.com/v1 \
  index-pdf paper.pdf --key ABC123 --title "Paper title"
```

Set `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, and optionally `OPENAI_BASE_URL` in `.env`. Local compatible servers can use any non-empty API key accepted by the server.

### Scan a draft

```bash
bibliograph suggest draft.txt --output suggestions.md
```

### Check a draft against a Zotero collection

This is the end-to-end workflow. It syncs one selected collection’s metadata, indexes only local PDFs that are new or changed, parses a Typst or LaTeX draft into claims, retrieves hybrid matches, and optionally reranks them with an LLM:

```bash
bibliograph check paper.tex "My Collection" \
  --pdf-dir pdfs --llm-model local-reranker --output citation-check.md
```

The command does not download PDFs. Missing attachment keys are listed in the report; download one explicitly with `bibliograph download-pdf KEY`, then rerun the check.

For grounded LLM rationales, set `OPENAI_API_KEY` and pass an OpenAI-compatible chat model:

```bash
bibliograph suggest draft.txt --llm-model local-model --output suggestions.md
```

### Download one missing PDF from Zotero

Set environment variables in a `.env` file:

```env
ZOTERO_LIBRARY_ID=<your_library_id>
ZOTERO_LIBRARY_TYPE=<library_type>
ZOTERO_API_KEY=<your_api_key>
```

Download a single parent item’s first PDF attachment. If that attachment is already present locally, nothing is downloaded:

```bash
bibliograph download-pdf PARENT_ITEM_KEY --output-dir pdfs
```

If the parent has several PDFs, select one explicitly with `--attachment-key`. This command never enumerates or downloads an entire collection.

### Run the original sentence-level prototype

```bash
python app.py
```

The script currently opens `test.pdf`, extracts sentences, stores them in a local Chroma collection, and lets you query them interactively.

## Current limitations

The repository is not yet a complete discovery system. Remaining issues include:

* Zotero downloading and indexing are not yet one automated sync command; the new CLI indexes local PDFs and the legacy connector downloads collection files.
* OCR for scanned PDFs and structural section extraction are not yet included.
* The SQLite index is intentionally small and uses brute-force cosine search.
* No browser UI is included yet; Markdown export is the current review workflow.

## Future direction

The next steps for the project are:

* add automated Zotero sync with attachment hashes and collection metadata
* add OCR, better academic chunking, hybrid retrieval, and reranking
* add a lightweight UI or API server for review and citation insertion

## Why this matters

Traditional citation-based discovery is useful, but it misses many semantic relationships between papers. This repo aims to build a pipeline that can support:

* semantic search over paper content
* retrieval of relevant documents even when they are not directly cited
* a Zotero-backed ingestion workflow for personal research libraries

## License

All rights remain with the author.
