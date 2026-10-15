"""Conversions between Bibliograph records and Ikarus graph records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.knowledge.graph import DocumentRecord, GraphEdge, GraphNode
from core.knowledge.ingest import DocumentGraph

from ..domain import Chunk, Paper


def _paper_metadata(paper: Paper, source_key: str, path: str | Path) -> dict[str, Any]:
    return {
        "zotero_key": paper.zotero_key,
        "title": paper.title,
        "authors": list(paper.authors),
        "year": paper.year,
        "doi": paper.doi,
        "citation_key": paper.citation_key,
        "collections": list(paper.collections),
        "source_key": source_key,
        "path": str(Path(path)),
        "modality": "scientific-paper",
    }


def _chunk_metadata(chunk: Chunk, record: DocumentRecord) -> dict[str, Any]:
    return {
        **dict(record.metadata or {}),
        "source": record.uri or record.identifier,
        "document_id": record.identifier,
        "revision": record.revision,
        "kind": "chunk",
        "text": chunk.text,
        "page": chunk.page,
        "section": chunk.section,
        "ordinal": chunk.ordinal,
        "content_kind": chunk.content_kind,
    }


def _chunk_node(chunk: Chunk, record: DocumentRecord) -> GraphNode:
    return GraphNode(
        chunk.chunk_id,
        "chunk",
        chunk.text,
        record.identifier,
        _chunk_metadata(chunk, record),
    )


def _paper_graph(record: DocumentRecord, chunks: Sequence[GraphNode]):
    metadata = dict(record.metadata or {})
    paper = GraphNode(
        f"paper:{record.identifier}",
        "paper",
        str(metadata.get("title") or record.identifier),
        record.identifier,
        {**metadata, "kind": "paper", "document_id": record.identifier},
    )
    nodes = [paper]
    edges = []
    sections: dict[str, GraphNode] = {}
    previous = None
    for chunk in chunks:
        section_name = chunk.metadata.get("section")
        if section_name:
            key = str(section_name).casefold()
            section = sections.get(key)
            if section is None:
                section = GraphNode(
                    f"section:{record.identifier}:{len(sections)}",
                    "section",
                    str(section_name),
                    record.identifier,
                    {
                        "kind": "section",
                        "document_id": record.identifier,
                        "title": str(section_name),
                    },
                )
                sections[key] = section
                nodes.append(section)
                edges.append(GraphEdge(paper.identifier, section.identifier, "contains"))
            edges.append(GraphEdge(chunk.identifier, section.identifier, "in_section"))
            edges.append(GraphEdge(section.identifier, chunk.identifier, "contains"))
        else:
            edges.append(GraphEdge(paper.identifier, chunk.identifier, "contains"))
        if previous is not None:
            edges.append(GraphEdge(previous, chunk.identifier, "next_chunk"))
        previous = chunk.identifier
    return DocumentGraph(tuple(nodes), tuple(edges))


def _paper_from_metadata(metadata, fallback: str) -> Paper:
    return Paper(
        str(metadata.get("zotero_key") or fallback),
        str(metadata.get("title") or fallback),
        tuple(str(author) for author in metadata.get("authors", ()) or ()),
        _optional_str(metadata.get("year")),
        _optional_str(metadata.get("doi")),
        tuple(str(value) for value in metadata.get("collections", ()) or ()),
        _optional_str(metadata.get("citation_key")),
    )


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _optional_str(value) -> str | None:
    return None if value is None else str(value)


def _chunk_paper(paper: Paper, pages: Sequence[tuple[int, str, str | None]]) -> list[Chunk]:
    from ..adapters.pdf import chunk_document

    return chunk_document(paper, pages)
