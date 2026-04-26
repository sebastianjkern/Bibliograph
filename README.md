# Paracite

**Paracite** is a research discovery workspace for building a secondary paper retrieval pipeline centered on semantic search and Zotero PDF ingestion.

## Current State

This repository is an early-stage prototype rather than a finished product. It currently provides:

* `app.py` — a local proof-of-concept for extracting sentences from a PDF, embedding them with `sentence-transformers`, storing them in `chromadb`, and performing semantic search.
* `connections/zotero_collections.py` — helper functions for fetching Zotero collection metadata.
* `connections/zotero_items.py` — helper functions for fetching Zotero items and exporting them to CSV.
* `connections/zotero_pdfs.py` — helper functions for downloading PDF attachments from Zotero item metadata.
* `connections/reload_embeddings.py` — a new workflow to download all available PDF attachments for items in a specified Zotero collection.

## What the project solves today

* Collects Zotero items and collection metadata from a configured Zotero library.
* Downloads PDF attachments from Zotero items in a given collection.
* Extracts text from local PDF files and splits it into searchable sentences.
* Stores sentence embeddings in a local Chroma collection for semantic retrieval.

## How to use the current tools

### Download PDFs from a Zotero collection

Set environment variables in a `.env` file:

```env
ZOTERO_LIBRARY_ID=<your_library_id>
ZOTERO_LIBRARY_TYPE=<library_type>
ZOTERO_API_KEY=<your_api_key>
```

Run:

```bash
python connections/reload_embeddings.py "My Collection Name" --output-dir pdfs
```

### Run sentence-level semantic search on a PDF

```bash
python app.py
```

The script currently opens `test.pdf`, extracts sentences, stores them in a local Chroma collection, and lets you query them interactively.

## Current limitations

The repository is not yet a complete discovery system. Remaining issues include:

* No integrated Zotero-to-search pipeline yet: PDF download, text extraction, embedding, and indexing are separate scripts.
* No central metadata storage or paper graph construction.
* No web/API layer or user interface.
* Limited error handling for PDF parsing and Zotero retrieval failures.
* No test coverage or packaging for reusable modules.
* Minimal handling of duplicate attachments, collection nesting, and library sync state.

## Future direction

The next steps for the project are:

* unify Zotero ingestion, PDF extraction, embedding generation, and vector store updates into a single pipeline
* add metadata and relationship modeling for papers and collections
* support a persistent retrieval layer with search and related-work APIs
* integrate Zotero collection structure and item metadata more fully
* add tests, configuration management, and a lightweight UI or API server

## Why this matters

Traditional citation-based discovery is useful, but it misses many semantic relationships between papers. This repo aims to build a pipeline that can support:

* semantic search over paper content
* retrieval of relevant documents even when they are not directly cited
* a Zotero-backed ingestion workflow for personal research libraries

## License

All rights remain with the author.