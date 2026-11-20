import json
import math
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Iterable, List, Optional

from retrieval.embedding_engine import FastEmbedEngine
from retrieval.models import DocumentChunk, RetrievedChunk
from retrieval.reranker import reciprocal_rank_fusion


class HybridKnowledgeStore:
    """Isolated SQLite FTS5/BM25 and local-vector index for one execution."""

    def __init__(
        self,
        database_path: Optional[Path],
        embedding_engine: FastEmbedEngine,
        hybrid_search: bool = True,
    ):
        self.database_path = Path(database_path) if database_path is not None else None
        if self.database_path is not None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        else:
            self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.embedding_engine = embedding_engine
        self.hybrid_search = hybrid_search
        self._lock = threading.RLock()
        self._closed = False
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self.connection.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB
                )"""
            )
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')"
            )
            self.connection.commit()

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        values = list(chunks)
        if not values:
            return 0
        vectors = self.embedding_engine.embed([chunk.text for chunk in values]) if self.hybrid_search else [None] * len(values)
        if len(vectors) != len(values):
            raise ValueError("Embedding result count does not match document chunk count.")
        with self._lock:
            self._ensure_open()
            for chunk, vector in zip(values, vectors):
                blob = struct.pack(f"<{len(vector)}f", *vector) if vector else None
                metadata = json.dumps(chunk.to_metadata(), ensure_ascii=True, separators=(",", ":"))
                self.connection.execute(
                    "INSERT OR REPLACE INTO chunks(chunk_id,source_id,text,metadata_json,embedding) VALUES(?,?,?,?,?)",
                    (chunk.chunk_id, chunk.source_id, chunk.text, metadata, blob),
                )
                self.connection.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk.chunk_id,))
                self.connection.execute("INSERT INTO chunks_fts(chunk_id,text) VALUES(?,?)", (chunk.chunk_id, chunk.text))
            self.connection.commit()
        return len(values)

    def retrieve(self, query: str, limit: int = 10) -> List[RetrievedChunk]:
        if not query or not query.strip() or limit < 1:
            return []
        self._ensure_open()
        candidate_limit = max(limit * 4, limit)
        lexical = self._keyword_search(query, candidate_limit)
        semantic = self._semantic_search(query, candidate_limit) if self.hybrid_search else []
        if self.hybrid_search:
            fused = reciprocal_rank_fusion(lexical, semantic, limit)
        else:
            fused = [(chunk, 1.0 / rank, None, rank) for rank, chunk in enumerate(lexical[:limit], 1)]
        return [RetrievedChunk(chunk, score, semantic_rank, lexical_rank) for chunk, score, semantic_rank, lexical_rank in fused]

    def _keyword_search(self, query: str, limit: int) -> List[DocumentChunk]:
        terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms[:32])
        with self._lock:
            rows = self.connection.execute(
                """SELECT c.metadata_json,c.text FROM chunks_fts f
                   JOIN chunks c ON c.chunk_id=f.chunk_id
                   WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?""",
                (match_query, limit),
            ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def _semantic_search(self, query: str, limit: int) -> List[DocumentChunk]:
        query_vector = self.embedding_engine.embed([query])[0]
        with self._lock:
            rows = self.connection.execute(
                "SELECT metadata_json,text,embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        scored = []
        for row in rows:
            vector_blob = row["embedding"]
            dimensions = len(vector_blob) // 4
            vector = struct.unpack(f"<{dimensions}f", vector_blob)
            score = _cosine(query_vector, vector)
            scored.append((score, _chunk_from_row(row)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Retrieval store is closed.")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.connection.close()
            if self.database_path is None:
                return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def _chunk_from_row(row: sqlite3.Row) -> DocumentChunk:
    metadata = json.loads(row["metadata_json"])
    return DocumentChunk(
        chunk_id=metadata["chunk_id"],
        source_id=metadata["source_id"],
        source_name=metadata["source_name"],
        source_reference=metadata["source_reference"],
        source_type=metadata["source_type"],
        retrieval_timestamp=metadata["retrieval_timestamp"],
        text=row["text"],
        page_number=metadata.get("page_number"),
        section_heading=metadata.get("section_heading"),
        chunk_index=metadata.get("chunk_index", 0),
        content_type=metadata.get("content_type"),
    )


def _cosine(left, right) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
