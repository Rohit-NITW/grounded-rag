import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .ingestion import Chunk, DocumentError


class Store:
    """SQLite is the source of truth. Each operation owns its connection."""

    def __init__(self, path: Path, max_chunks: int = 10_000):
        self.path = path
        self.max_chunks = max_chunks
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id)
                    ON DELETE CASCADE, filename TEXT NOT NULL, page INTEGER NOT NULL,
                    start_word INTEGER NOT NULL, text TEXT NOT NULL, vector TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
            """)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def documents(self) -> list[dict]:
        with self.connection() as conn:
            return [dict(row) for row in conn.execute("""
                SELECT d.*, COUNT(c.id) AS chunk_count FROM documents d
                LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id
                ORDER BY d.created_at DESC, d.id
            """)]

    def contains(self, document_id: str) -> bool:
        with self.connection() as conn:
            return conn.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone() is not None

    def add(self, document_id: str, chunks: list[Chunk], vectors: np.ndarray, size: int) -> bool:
        if not chunks or vectors.ndim != 2 or len(chunks) != len(vectors):
            raise ValueError("One embedding is required per chunk")
        if not np.isfinite(vectors).all() or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be finite, non-empty vectors")
        with self.connection() as conn:
            # Serialize capacity check + insert across processes. Failure rolls everything back.
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM documents WHERE id=?", (document_id,)).fetchone():
                return False
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count + len(chunks) > self.max_chunks:
                raise DocumentError("Index capacity reached; delete documents before adding more")
            previous = conn.execute("SELECT vector FROM chunks LIMIT 1").fetchone()
            if previous and len(json.loads(previous[0])) != vectors.shape[1]:
                raise DocumentError("Embedding dimensions changed; use a new data directory and re-ingest")
            conn.execute("INSERT INTO documents VALUES (?,?,?,?)", (
                document_id, chunks[0].filename, size, datetime.now(timezone.utc).isoformat()))
            conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", [
                (c.id, c.document_id, c.filename, c.page, c.start_word, c.text, json.dumps(v.tolist()))
                for c, v in zip(chunks, vectors, strict=True)
            ])
        return True

    def snapshot(self, document_ids: list[str] | None = None) -> tuple[list[dict], np.ndarray]:
        with self.connection() as conn:
            sql = "SELECT * FROM chunks"
            params: list[str] = document_ids or []
            if params:
                sql += " WHERE document_id IN (" + ",".join("?" for _ in params) + ")"
            rows = [dict(row) for row in conn.execute(sql + " ORDER BY id", params)]
        vectors = np.asarray([json.loads(row.pop("vector")) for row in rows], dtype=np.float32)
        return rows, vectors

    def delete(self, document_id: str) -> bool:
        with self.connection() as conn:
            return conn.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount > 0
