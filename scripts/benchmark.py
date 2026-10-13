#!/usr/bin/env python3
"""Build and evaluate Bibliograph retrieval on a BEIR-format SciFact dataset."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import multiprocessing
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from queue import Empty
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.providers.model_runtimes.registry import build_embedding  # noqa: E402

from bibliograph.adapters.ikarus import IkarusBackend  # noqa: E402
from bibliograph.domain import Paper  # noqa: E402
from bibliograph.knowledge import KnowledgeDocument, QueryRequest  # noqa: E402
from bibliograph.pipeline.retrieval import (  # noqa: E402
    DEFAULT_SEARCH_LIMIT,
    initial_retrieval_limit,
)
from bibliograph.pipeline.timing import TimingRecorder  # noqa: E402
from bibliograph.settings import load_settings  # noqa: E402

DEFAULT_DATA = ROOT / "benchmark" / "data" / "scifact"
DEFAULT_INDEX = ROOT / "benchmark" / ".cache" / "scifact.db"
_WORKER_BACKEND = None
_WORKER_PROGRESS_QUEUE = None


def _close_retrieval_worker() -> None:
    if _WORKER_BACKEND is not None:
        _WORKER_BACKEND.close()


def _initialize_retrieval_worker(settings, progress_queue) -> None:
    global _WORKER_BACKEND, _WORKER_PROGRESS_QUEUE

    _WORKER_PROGRESS_QUEUE = progress_queue
    worker_embedding = build_embedding(settings)
    _WORKER_BACKEND = IkarusBackend(
        settings["db"],
        mode="read",
        embedding=worker_embedding,
        classification_enabled=settings.get("classification", {}).get("enabled", False),
    )
    atexit.register(_close_retrieval_worker)


def _retrieve_in_worker(query_id, query, limit, query_vector):
    if _WORKER_BACKEND is None:
        raise RuntimeError("Benchmark retrieval worker was not initialized")

    def report(step: str) -> None:
        if _WORKER_PROGRESS_QUEUE is not None:
            _WORKER_PROGRESS_QUEUE.put((query_id, step))

    return query_id, *_ranked_document_ids(
        _WORKER_BACKEND,
        query,
        limit,
        report,
        query_vector,
    )


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as source:
        header = next(source, "").strip().split("\t")
        if header != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"Unexpected qrels header in {path}: {header}")
        for line_number, line in enumerate(source, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Invalid qrels row at {path}:{line_number}")
            query_id, document_id, score = fields
            qrels[query_id][document_id] = int(score)
    return dict(qrels)


def _ensure_index(
    backend: IkarusBackend,
    corpus: list[dict[str, Any]],
) -> int:
    """Index abstracts as one-page source documents using Bibliograph's chunker."""
    indexed = 0
    with tempfile.TemporaryDirectory(prefix="bibliograph-scifact-") as temporary:
        temp_dir = Path(temporary)
        for number, record in enumerate(corpus, start=1):
            document_id = str(record["_id"])
            title = str(record.get("title") or document_id)
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            source_key = f"scifact-{document_id}"
            path = temp_dir / f"{document_id}.txt"
            path.write_text(text, encoding="utf-8")
            paper = Paper(zotero_key=source_key, title=title)
            backend.index_document(
                KnowledgeDocument(
                    source_key=source_key,
                    paper=paper,
                    path=path,
                    version="scifact-v1",
                    metadata={"scifact_document_id": document_id},
                ),
                extract_pages=lambda source, title=title: [
                    (1, Path(source).read_text(encoding="utf-8"), title)
                ],
            )
            indexed += 1
            if number % 250 == 0 or number == len(corpus):
                print(f"Indexed {number}/{len(corpus)} corpus records", file=sys.stderr)
    return indexed


def _ranked_document_ids(
    backend: IkarusBackend,
    query: str,
    limit: int,
    progress,
    query_vector: list[float],
) -> tuple[list[str], dict[str, dict[str, float | int]]]:
    timings = TimingRecorder()
    with timings.measure("retrieval_engine"):
        result = backend.retrieve_request(
            QueryRequest(text=query),
            limit=limit,
            progress=progress,
            query_vector=query_vector,
            timings=timings,
        )
    ranked: list[str] = []
    seen: set[str] = set()
    for chunk, _score in result.hits:
        source_key = chunk.paper.zotero_key
        document_id = source_key.removeprefix("scifact-")
        if document_id not in seen:
            ranked.append(document_id)
            seen.add(document_id)
    return ranked, timings.snapshot()


def _run_parallel_retrieval(
    evaluated,
    query_vectors,
    *,
    settings,
    workers,
    limit,
    verbose=False,
) -> dict[str, list[str]]:
    context = multiprocessing.get_context("spawn")
    progress_queue = context.Queue()
    rankings: dict[str, list[str]] = {}
    retrieval_timings: list[dict[str, dict[str, float | int]]] = []
    worker_count = min(workers, len(evaluated))
    print(
        f"Starting {worker_count} retrieval worker process(es)",
        file=sys.stderr,
        flush=True,
    )
    started = time.perf_counter()
    completed = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_retrieval_worker,
        initargs=(dict(settings), progress_queue),
    ) as executor:
        pending = {
            executor.submit(
                _retrieve_in_worker,
                str(query["_id"]),
                str(query["text"]),
                limit,
                query_vectors[str(query["_id"])],
            ): str(query["_id"])
            for query in evaluated
        }
        while pending:
            try:
                query_id, step = progress_queue.get(timeout=0.1)
                print(f"Claim {query_id} · Engine · {step}", file=sys.stderr, flush=True)
            except Empty:
                pass
            done, _ = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                query_id, document_ids, timings_one = future.result()
                rankings[query_id] = document_ids
                retrieval_timings.append(timings_one)
                completed += 1
                elapsed = time.perf_counter() - started
                remaining = elapsed / completed * (len(evaluated) - completed)
                print(
                    f"Evaluated {completed}/{len(evaluated)} claims · "
                    f"{elapsed:.1f}s elapsed · ~{remaining:.1f}s remaining",
                    file=sys.stderr,
                    flush=True,
                )
    while True:
        try:
            query_id, step = progress_queue.get_nowait()
        except Empty:
            break
        print(f"Claim {query_id} · Engine · {step}", file=sys.stderr, flush=True)
    progress_queue.close()
    progress_queue.join_thread()
    if verbose and retrieval_timings:
        totals: dict[str, float] = defaultdict(float)
        for sample in retrieval_timings:
            for stage, timing in sample.items():
                totals[stage] += float(timing["seconds"])
        print(
            "Cumulative engine stage timings across workers: "
            + " · ".join(f"{stage} {seconds:.2f}s" for stage, seconds in sorted(totals.items())),
            file=sys.stderr,
            flush=True,
        )
    return rankings


def _metrics(
    queries: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    rankings: dict[str, list[str]],
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    evaluated = [query for query in queries if str(query["_id"]) in qrels]
    if not evaluated:
        raise ValueError("No query IDs from queries.jsonl have judgments in the selected qrels")
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        recalls: list[float] = []
        hit_rates: list[float] = []
        ndcgs: list[float] = []
        reciprocal_ranks: list[float] = []
        for query in evaluated:
            query_id = str(query["_id"])
            relevant = {
                doc_id for doc_id, score in qrels[query_id].items() if score > 0
            }
            ranked = rankings.get(query_id, [])[:cutoff]
            relevant_found = sum(document_id in relevant for document_id in ranked)
            recalls.append(relevant_found / len(relevant) if relevant else 0.0)
            hit_rates.append(float(relevant_found > 0))
            reciprocal_ranks.append(
                next(
                    (
                        1.0 / rank
                        for rank, doc_id in enumerate(ranked, start=1)
                        if doc_id in relevant
                    ),
                    0.0,
                )
            )
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, doc_id in enumerate(ranked, start=1)
                if doc_id in relevant
            )
            ideal_hits = min(len(relevant), cutoff)
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        metrics[f"Recall@{cutoff}"] = sum(recalls) / len(recalls)
        metrics[f"HitRate@{cutoff}"] = sum(hit_rates) / len(hit_rates)
        metrics[f"MRR@{cutoff}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
        metrics[f"nDCG@{cutoff}"] = sum(ndcgs) / len(ndcgs)
    metrics["queries"] = float(len(evaluated))
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="SciFact BEIR data directory"
    )
    parser.add_argument("--split", default="test", help="qrels split (default: test)")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="SQLite index path")
    parser.add_argument(
        "--config", type=Path, help="Bibliograph TOML config (defaults to normal settings)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Normal search result limit (default: 5)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Claims embedded per provider request (default: 32)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel retrieval worker processes (default: 2; use 1 for sequential)",
    )
    parser.add_argument(
        "-n",
        "--max-claims",
        type=int,
        help="Evaluate only the first N judged claims (default: all)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Delete and rebuild the benchmark index"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show embedding and engine timing diagnostics"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.limit <= 0 or args.embedding_batch_size <= 0 or args.workers <= 0:
        raise ValueError("--limit, --embedding-batch-size, and --workers must be positive")
    if args.max_claims is not None and args.max_claims <= 0:
        raise ValueError("--max-claims must be positive")
    corpus_path = args.data / "corpus.jsonl"
    queries_path = args.data / "queries.jsonl"
    qrels_path = args.data / "qrels" / f"{args.split}.tsv"
    for path in (corpus_path, queries_path, qrels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required SciFact file not found: {path}")

    corpus = list(_jsonl(corpus_path))
    queries = list(_jsonl(queries_path))
    qrels = _read_qrels(qrels_path)
    settings = load_settings(args.config)
    embedding = build_embedding(settings["embedding"])
    args.index.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild:
        for path in (args.index, Path(f"{args.index}-wal"), Path(f"{args.index}-shm")):
            path.unlink(missing_ok=True)

    if not args.index.exists():
        with IkarusBackend(args.index, embedding=embedding, mode="write") as backend:
            count = _ensure_index(backend, corpus)
        print(f"Built SciFact index with {count} abstracts at {args.index}", file=sys.stderr)
    else:
        print(
            f"Using existing index at {args.index} (pass --rebuild to replace it)",
            file=sys.stderr,
        )

    # Match the regular evidence workflow's backend request size.
    # The adapter then over-fetches up to four times that count before fusion.
    backend_limit = initial_retrieval_limit(args.limit)
    max_cutoff = backend_limit * 4
    cutoffs = tuple(cutoff for cutoff in (1, 5, 10, 20, 50, 100) if cutoff <= max_cutoff)
    rankings: dict[str, list[str]] = {}
    judged = [query for query in queries if str(query["_id"]) in qrels]
    evaluated = judged[: args.max_claims] if args.max_claims is not None else judged
    if len(evaluated) < len(judged):
        print(
            f"Evaluating first {len(evaluated)} of {len(judged)} judged claims",
            file=sys.stderr,
            flush=True,
        )
    query_vectors: dict[str, list[float]] = {}
    batch_count = math.ceil(len(evaluated) / args.embedding_batch_size)
    embedding_started = time.perf_counter()
    for start in range(0, len(evaluated), args.embedding_batch_size):
        batch = evaluated[start : start + args.embedding_batch_size]
        batch_number = start // args.embedding_batch_size + 1
        print(
            f"Embedding claim batch {batch_number}/{batch_count} "
            f"({len(batch)} claims)",
            file=sys.stderr,
            flush=True,
        )
        vectors = embedding["embed_queries"]([str(query["text"]) for query in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned a different number of query vectors")
        query_vectors.update(
            (str(query["_id"]), vector) for query, vector in zip(batch, vectors, strict=True)
        )
    if args.verbose:
        print(
            f"Embedded {len(query_vectors)} claims in "
            f"{time.perf_counter() - embedding_started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    if args.workers > 1 and evaluated:
        worker_settings = {**settings, "db": str(args.index)}
        rankings = _run_parallel_retrieval(
            evaluated,
            query_vectors,
            settings=worker_settings,
            workers=args.workers,
            limit=backend_limit,
            verbose=args.verbose,
        )
    else:
        with IkarusBackend(args.index, embedding=embedding, mode="read") as backend:
            started = time.perf_counter()
            retrieval_timings: list[dict[str, dict[str, float | int]]] = []
            for number, query in enumerate(evaluated, start=1):
                query_id = str(query["_id"])
                print(
                    f"Claim {number}/{len(evaluated)} · starting retrieval engine",
                    file=sys.stderr,
                    flush=True,
                )

                def report_engine_step(step: str) -> None:
                    print(f"  Engine · {step}", file=sys.stderr, flush=True)

                rankings[query_id], timings_one = _ranked_document_ids(
                    backend,
                    str(query["text"]),
                    backend_limit,
                    report_engine_step,
                    query_vectors[query_id],
                )
                retrieval_timings.append(timings_one)
                elapsed = time.perf_counter() - started
                average = elapsed / number
                remaining = average * (len(evaluated) - number)
                print(
                    f"Evaluated {number}/{len(evaluated)} claims · "
                    f"{elapsed:.1f}s elapsed · ~{remaining:.1f}s remaining",
                    file=sys.stderr,
                    flush=True,
                )
            if args.verbose and retrieval_timings:
                totals: dict[str, float] = defaultdict(float)
                for sample in retrieval_timings:
                    for stage, timing in sample.items():
                        totals[stage] += float(timing["seconds"])
                print(
                    "Engine stage timings: "
                    + " · ".join(
                        f"{stage} {seconds:.2f}s" for stage, seconds in sorted(totals.items())
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    print(f"SciFact {args.split} retrieval results (document-level qrels)")
    print(f"Index: {args.index}")
    for name, value in _metrics(evaluated, qrels, rankings, cutoffs).items():
        if name.startswith("HitRate@"):
            print(f"{name}: {value:.1%}")
        elif name == "queries":
            print(f"{name}: {int(value)}")
        else:
            print(f"{name}: {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
