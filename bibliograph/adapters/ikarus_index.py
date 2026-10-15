"""Inspection and atomic publication of Ikarus index generations."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapters.providers.vector_stores import vectorlite

from .ikarus_errors import IkarusIndexError


def is_ikarus_index(path: str | Path) -> bool:
    database = Path(path)
    if not database.is_file():
        return False
    connection = None
    try:
        connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
        row = connection.execute(
            "SELECT backend FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        return row is not None and row[0] == "ikarus"
    except sqlite3.DatabaseError:
        return False
    finally:
        if connection is not None:
            connection.close()


def inspect_ikarus_index(path: str | Path, *, vector_metadata_table: str) -> dict[str, Any]:
    if vector_metadata_table not in {"vector_metadata", "vectorlite_metadata"}:
        raise ValueError("Unsupported vector metadata table")
    database = Path(path)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        meta = connection.execute(
            "SELECT * FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        manifest = json.loads(
            connection.execute(
                "SELECT manifest FROM ikarus_index_manifest WHERE index_name='default'"
            ).fetchone()[0]
        )

        def count(sql: str) -> int:
            return int(connection.execute(sql).fetchone()[0])

        return {
            "path": str(database),
            "schema_version": int(meta["schema_version"]),
            "papers": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='paper'"),
            "documents": count(
                "SELECT COUNT(*) FROM bibliograph_sources WHERE state IN ('indexed','empty')"
            ),
            "chunks": count("SELECT COUNT(*) FROM graph_nodes WHERE kind='chunk'"),
            "vectors": count(f"SELECT COUNT(*) FROM {vector_metadata_table}"),
            "missing_vectors": count(
                f"SELECT COUNT(*) FROM graph_nodes AS nodes "
                f"LEFT JOIN {vector_metadata_table} AS vectors "
                "ON vectors.identifier=nodes.identifier "
                "WHERE nodes.kind='chunk' AND vectors.identifier IS NULL"
            ),
            "embedding_id": manifest["embedding_id"],
            "dimension": int(manifest["embedding_dimension"]),
            "extraction_chunking_fingerprint": manifest["ingestion_fingerprint"],
            "last_successful_sync": meta["last_successful_sync"],
            "pending_failures": count(
                "SELECT COUNT(*) FROM bibliograph_sources WHERE state NOT IN ('indexed','empty')"
            ),
            "rebuild_required": False,
        }
    finally:
        connection.close()


def _can_seed_ikarus_index(path: str | Path, embedding_id: str | None = None) -> bool:
    """Return whether graph content can seed an atomic sync generation."""
    database = Path(path)
    if not is_ikarus_index(database):
        return False
    connection = None
    try:
        connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
        meta = connection.execute(
            "SELECT embedding_id FROM bibliograph_index_meta WHERE singleton=1"
        ).fetchone()
        if meta is None or (embedding_id is not None and meta[0] != embedding_id):
            return False
        graph_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='graph_nodes'"
        ).fetchone()
        return graph_table is not None
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


def _seed_staging_database(source: Path, staging: Path, index_namespace: str | None) -> None:
    """Clone a complete index and give its Vectorlite files a private generation path."""
    source_connection = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    staging_connection = sqlite3.connect(staging)
    try:
        source_connection.backup(staging_connection)
        if index_namespace is None:
            return
        tables = {
            row[0]
            for row in staging_connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "vectorlite_metadata" not in tables:
            return
        config = (
            staging_connection.execute(
                "SELECT value FROM vectorlite_config WHERE key='index_namespace'"
            ).fetchone()
            if "vectorlite_config" in tables
            else None
        )
        old_namespace = config[0] if config else None
        dimensions = [
            int(row[0])
            for row in staging_connection.execute(
                "SELECT DISTINCT dimension FROM vectorlite_metadata"
            )
        ]
        for dimension in dimensions:
            table = f"vectorlite_vec_{dimension}"
            declaration_row = staging_connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if declaration_row is None:
                raise IkarusIndexError(
                    f"Vectorlite table for dimension {dimension} is missing; rebuild the index."
                )
            old_path = (
                Path(f"{old_namespace}.vectorlite-{dimension}.hnsw")
                if old_namespace
                else source.with_name(f"{source.name}.vectorlite-{dimension}.hnsw")
            )
            new_path = Path(f"{index_namespace}.vectorlite-{dimension}.hnsw")
            old_sql_path = str(old_path).replace("'", "''")
            new_sql_path = str(new_path).replace("'", "''")
            declaration = declaration_row[0] or ""
            if not old_path.is_file() or old_sql_path not in declaration:
                # The staged adapter will discard this unusable HNSW table and
                # re-embed its graph chunks from the copied SQLite generation.
                continue
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_path, new_path)
            staging_connection.execute("PRAGMA writable_schema=ON")
            staging_connection.execute(
                "UPDATE sqlite_master SET sql=? WHERE type='table' AND name=?",
                (declaration.replace(old_sql_path, new_sql_path), table),
            )
            staging_connection.execute("PRAGMA writable_schema=OFF")
        if "vectorlite_config" in tables:
            staging_connection.execute(
                "INSERT INTO vectorlite_config(key, value) VALUES ('index_namespace', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (index_namespace,),
            )
        staging_connection.execute(
            "PRAGMA schema_version="
            + str(staging_connection.execute("PRAGMA schema_version").fetchone()[0] + 1)
        )
        staging_connection.commit()
    finally:
        staging_connection.close()
        source_connection.close()


@contextmanager
def open_ikarus_staging(
    path: str | Path,
    *,
    embedding: Mapping[str, Any],
    ingestion_strategy: Callable[[str], dict[str, Any]] | None = None,
    classification_enabled: bool = False,
    seed_existing: bool = False,
    backend_factory: Callable[..., Any],
    vector_store,
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.ikarus-staging-", dir=target.parent)
    os.close(descriptor)
    staging = Path(name)
    index_namespace = (
        str(target.parent.resolve() / f".{target.name}.generation-{uuid4().hex}")
        if vector_store is vectorlite
        else None
    )
    backend = None
    published = False
    try:
        if seed_existing:
            _seed_staging_database(target, staging, index_namespace)
        backend = backend_factory(
            staging,
            embedding=embedding,
            mode="write",
            ingestion_strategy=ingestion_strategy,
            classification_enabled=classification_enabled,
            vector_index_namespace=index_namespace,
        )
        yield backend
        summary = backend.stats()
        backend.close()
        backend = None
        if not all(
            summary[field] > 0 for field in ("documents", "chunks", "vectors")
        ) or summary.get("missing_vectors", 0):
            raise IkarusIndexError(
                "Refusing to publish an incomplete staged index: every chunk must have a vector."
            )
        if vector_store is vectorlite:
            staged_connection = sqlite3.connect(f"file:{staging.resolve()}?mode=ro", uri=True)
            try:
                dimensions = [
                    int(row[0])
                    for row in staged_connection.execute(
                        "SELECT DISTINCT dimension FROM vectorlite_metadata"
                    )
                ]
            finally:
                staged_connection.close()
            missing_sidecars = [
                str(Path(f"{index_namespace}.vectorlite-{dimension}.hnsw"))
                for dimension in dimensions
                if not Path(f"{index_namespace}.vectorlite-{dimension}.hnsw").is_file()
            ]
            if missing_sidecars:
                raise IkarusIndexError(
                    "Refusing to publish an index with missing Vectorlite sidecars: "
                    + ", ".join(missing_sidecars)
                )
        os.replace(staging, target)
        published = True
    finally:
        if backend is not None:
            backend.close()
        for candidate in (staging, Path(f"{staging}-wal"), Path(f"{staging}-shm")):
            candidate.unlink(missing_ok=True)
        # A KeyboardInterrupt can land after os.replace() but before the flag
        # assignment. Keep the generation if the published DB already points
        # at it; otherwise cleanup would leave a valid DB with a missing HNSW.
        target_uses_generation = False
        if index_namespace is not None and not published and target.is_file():
            try:
                published_connection = sqlite3.connect(f"file:{target.resolve()}?mode=ro", uri=True)
                try:
                    namespace_row = published_connection.execute(
                        "SELECT value FROM vectorlite_config WHERE key='index_namespace'"
                    ).fetchone()
                    target_uses_generation = (
                        namespace_row is not None and namespace_row[0] == index_namespace
                    )
                finally:
                    published_connection.close()
            except sqlite3.Error:
                pass
        if index_namespace is not None and not published and not target_uses_generation:
            for sidecar in target.parent.glob(Path(index_namespace).name + ".vectorlite-*.hnsw"):
                sidecar.unlink(missing_ok=True)
