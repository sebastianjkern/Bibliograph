![Bibliograph-Banner](./github-banner.svg)

# Bibliograph

Bibliograph is a local citation discovery assistant. It indexes papers from a Zotero-backed library, scans draft passages for semantic matches, and returns evidence-backed citation suggestions.

## Current State

This repository is an early-stage prototype rather than a finished product. It currently provides:

* `bibliograph/` — the reusable ingestion, chunking, embedding, retrieval, suggestion, export, and CLI modules.
* `connections/zotero_collections.py` — helper functions for fetching Zotero collection metadata.
* `connections/zotero_items.py` — helper functions for fetching Zotero items and exporting them to CSV.
* `connections/zotero_pdfs.py` — helper functions for downloading PDF attachments from Zotero item metadata.
* `connections/reload_embeddings.py` — Zotero metadata lookup and single-item PDF downloading.

## What the project solves today

* Collects Zotero items and collection metadata from a configured Zotero library.
* Downloads one missing PDF at a time from Zotero or legal remote open-access sources.
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

Install the package and all tools used by the complete workflow:

```bash
uv sync
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
ZOTERO_STORAGE_DIR=path/to/zotero/storage
```

Use `ZOTERO_LIBRARY_TYPE=group` for a group library. The API key must have read access to the library.

`ZOTERO_STORAGE_DIR` can point to Zotero's local `storage` directory. Bibliograph checks this directory first and imports valid PDFs from `storage/<attachment-key>/` into `pdfs` before trying any API or remote download. If it is unset, standard Zotero profile locations are discovered automatically. On Windows this is commonly under `%APPDATA%\Zotero\Zotero\Profiles\<profile>\storage`; on Linux, under `~/.zotero/zotero/<profile>/storage`.

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

### Quick-start examples

After configuring Zotero and creating `example.tex`, run the complete workflow with the defaults:

```bash
uv run bibliograph check example.tex "My Collection"
```

This uses the cached SentenceTransformer embedding model, the `pdfs` directory, the persistent `bibliograph.db` index, local Zotero PDFs before remote downloads for missing DOI papers, LLM reranking with `qwen/qwen3-1.7b` when available, and prints the report to the console. Add `--output citation-check.md` when you want to save the report.

The index records the selected embedding backend and model. If you change models later, Bibliograph clears the incompatible vectors and reindexes the available PDFs automatically.

To index a PDF that is already on disk and then search a draft:

```bash
uv run bibliograph index-pdf paper.pdf --key PAPER1 --title "Paper title"
uv run bibliograph suggest example.tex
```

To search the local library for one claim without providing a draft or syncing Zotero:

```bash
uv run bibliograph find-sources "Road infrastructure improves regional market access" --limit 5
```

The command reranks the retrieved sources with the configured LLM by default. Add `--no-rerank` for deterministic retrieval order, or `--output sources.md` to save the ranked evidence.

For a fully local setup with LM Studio, start its local server, load `qwen/qwen3-1.7b` and `text-embedding-nomic-embed-text-v1.5`, then run:

```bash
uv run bibliograph --embedding-provider openai \
  --model text-embedding-nomic-embed-text-v1.5 \
  check example.tex "My Collection" --llm-model qwen/qwen3-1.7b
```

The default LM Studio endpoint is `http://localhost:1234/v1`; set `OPENAI_BASE_URL` if the server uses another URL. The default API key value is `lm-studio`, so no real OpenAI key is needed for this local setup.

### Choose an embedding backend

The default backend is the cached `all-MiniLM-L6-v2` SentenceTransformer model. Its first use downloads the model; later runs reuse the local cache. For a dependency-free deterministic smoke test, explicitly select the hash backend:

```bash
uv run bibliograph --embedding-provider hash check example.tex "My Collection"
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

To download from a remote source instead of a Zotero attachment, use a DOI:

```bash
uv run bibliograph download-pdf --source remote --doi 10.1234/example --output-dir pdfs
```

Remote downloads use `doidownloader`. It checks the DOI's publisher route, publisher metadata, known publisher PDF routes, and legally available open-access locations. This can use publisher access provided by your university network, but it does not bypass authentication or paywalls. Bibliograph requires Python 3.12 or newer because `doidownloader` is part of the main installation.

When a publisher page needs browser-based access, Bibliograph falls back to Playwright. Install its Chromium browser once:

```bash
uv run playwright install chromium
```

For institutional login cookies, set `BIBLIOGRAPH_PLAYWRIGHT_PROFILE` to a dedicated browser profile directory. Set `BIBLIOGRAPH_PLAYWRIGHT_HEADLESS=false` when an interactive login is needed. The fallback only follows HTTPS links discovered on DOI-resolved publisher pages and accepts responses with a PDF signature.

If your default `uv` environment uses Python 3.11 or older, create the downloader environment explicitly:

```bash
uv python install 3.12
uv sync --python 3.12
uv run --python 3.12 bibliograph download-pdf --source remote --doi 10.1234/example
```

If automatic retrieval fails, Bibliograph uses `pyDOI` to resolve the DOI and logs the publisher landing page and DOI URL as manual download hints. Open one of those URLs in a browser with your institutional access, save the PDF into `pdfs`, and rerun the check.

### Run the complete citation check

Run the end-to-end workflow:

```bash
uv run bibliograph check example.tex "My Collection" --pdf-dir pdfs --output citation-check.md
```

The command:

* checks the selected Zotero collection;
* imports already-downloaded PDFs from Zotero's local storage when configured;
* detects new or changed attachments using Zotero versions and local file hashes;
* indexes only local PDFs that need indexing;
* reports Zotero PDFs that are not downloaded locally;
* parses the draft into claims;
* combines semantic and lexical retrieval;
* returns the closest evidence passages; and
* optionally reranks candidates with an LLM.

It downloads missing PDFs one at a time when `--download-missing` is enabled. It never bulk-downloads the collection.

To let `check` download missing papers through the Zotero attachment backend, opt in explicitly:

```bash
uv run bibliograph check example.tex "My Collection" --pdf-dir pdfs --download-missing
```

To use remote open-access downloads instead:

```bash
uv run bibliograph check example.tex "My Collection" --pdf-dir pdfs --download-missing --download-source remote
```

Remote downloads require a DOI. Zotero still supplies the collection metadata and paper title; the PDF bytes come from the publisher or legal open-access route selected by `doidownloader`.

PDF indexing and remote downloads are limited to paper-like Zotero item types (`journalArticle`, `book`, `conferencePaper`, `report`, `thesis`, `preprint`, and related types). Datasets, webpages, software, media, and other non-paper records are ignored before attachment lookup or download. Zotero PDF attachments are selected using the attachment's `contentType` metadata.

Remote resolvers are extensible through the `RemoteResolver` protocol in `bibliograph.remote`. A publisher-specific integration can implement `resolve(doi)` and return `RemoteCandidate` objects containing an HTTPS URL and an explicit legal basis. Pass custom resolvers to `download_remote_pdf(..., resolvers=[...])`; do not add arbitrary scraping or paywall-bypass handlers.

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

### Full Python example

The following example calls the library directly instead of using the CLI. It synchronizes one Zotero collection, optionally downloads missing PDFs from the legal remote resolver chain, indexes them with sqlite-vec, checks a Typst or LaTeX draft, optionally reranks results with an LLM, and writes a Markdown report.

Save it as `check_draft.py` in the repository root:

```python
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from bibliograph.drafts import parse_draft_file
from bibliograph.embeddings import SentenceTransformerEmbedder
from bibliograph.export import to_markdown
from bibliograph.remote import download_remote_pdf
from bibliograph.reranker import HeuristicReranker, OpenAIReranker, rerank_matches
from bibliograph.retrieval import find_claim_citations
from bibliograph.store import SQLiteIndex
from bibliograph.suggestions import suggest_citations
from bibliograph.zotero_sync import is_remote_downloadable_item, sync_collection
from connections.reload_embeddings import find_collection_key, load_zotero_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path, help="A .tex, .latex, or .typ file")
    parser.add_argument("collection", help="Zotero collection name or key")
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdfs"))
    parser.add_argument("--db", default="bibliograph.db")
    parser.add_argument("--output", type=Path, default=Path("citation-check.md"))
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing DOI PDFs from legal remote sources",
    )
    args = parser.parse_args()

    load_dotenv()

    # Uses EMBEDDING_MODEL, SENTENCE_TRANSFORMERS_CACHE, and
    # SENTENCE_TRANSFORMERS_OFFLINE from the environment when configured.
    embedder = SentenceTransformerEmbedder.from_environment()
    index = SQLiteIndex(args.db, dimension=embedder.dimension, embedding_id=embedder.identity)
    zotero = load_zotero_client()
    collection_key = find_collection_key(zotero, args.collection)

    try:
        report = sync_collection(
            zotero,
            collection_key,
            args.pdf_dir,
            index,
            embedder,
            zotero_storage_dir=os.getenv("ZOTERO_STORAGE_DIR"),
        )

        if args.download_missing:
            for paper in report.missing_papers:
                if not is_remote_downloadable_item(paper.item_type):
                    print(f"Skipping non-paper Zotero item: {paper.title}")
                    continue
                if not paper.doi:
                    print(f"No DOI available for: {paper.title}")
                    continue
                try:
                    result = download_remote_pdf(
                        paper.doi,
                        args.pdf_dir,
                        title=paper.title,
                        item_key=paper.item_key,
                    )
                    print(f"{result.resolver}: {result.path}")
                except (FileNotFoundError, ValueError) as error:
                    print(f"Could not download {paper.title}: {error}")

            # Index PDFs downloaded during this run.
            report = sync_collection(
                zotero,
                collection_key,
                args.pdf_dir,
                index,
                embedder,
                zotero_storage_dir=os.getenv("ZOTERO_STORAGE_DIR"),
            )

        claims = parse_draft_file(args.draft)
        matches = find_claim_citations(claims, index, embedder, limit=5)

        reranker_model = os.getenv("RERANKER_MODEL")
        if reranker_model and os.getenv("OPENAI_API_KEY"):
            reranker = OpenAIReranker(
                reranker_model,
                os.environ["OPENAI_API_KEY"],
                os.getenv("OPENAI_BASE_URL"),
            )
        else:
            reranker = HeuristicReranker()
        matches = rerank_matches(matches, reranker)

        report_text = (
            f"Indexed: {len(report.indexed)} | "
            f"Unchanged: {len(report.unchanged)} | "
            f"Missing papers: {len(report.missing_papers)}\n\n"
            + to_markdown(suggest_citations(matches))
        )
        args.output.write_text(report_text, encoding="utf-8")
        print(f"Wrote {args.output}")
    finally:
        index.close()


if __name__ == "__main__":
    main()
```

Run it with:

```bash
uv run python check_draft.py example.tex "My Collection" --download-missing
```

Remove `--download-missing` to report missing papers without downloading them. Set `RERANKER_MODEL` and `OPENAI_API_KEY` to enable LLM reranking; otherwise the example uses the deterministic retrieval order.

### Troubleshooting

* **Collection not found:** the collection name must match exactly; use its Zotero collection key if necessary.
* **No matches:** confirm that PDFs exist in `pdfs`, use a semantic embedding backend, and lower `--min-score` if one was provided.
* **Authentication errors:** verify the Zotero library ID, library type, and API key permissions.
* **Scanned PDFs:** PDFs must contain extractable text; OCR is not currently included.
* **Start over:** use a new database with `--db fresh-index.db` rather than deleting an existing index.

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
