"""SQLite persistence: chunks, FTS5 for BM25, embeddings as float32 blobs.

One file on disk. FTS5 ships inside CPython's bundled SQLite; vectors are searched
brute-force in NumPy (see retrieve.py) because the corpus is thousands of chunks and
exact search in milliseconds beats an approximate index with failure modes.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np

from .ingest import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    doc     TEXT NOT NULL,
    page    INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    text    TEXT NOT NULL,
    UNIQUE (doc, ordinal)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);
"""


class Store:
    def __init__(self, path: Path | str):
        path = Path(path)
        if path.parent and str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)

    # -- write ---------------------------------------------------------------

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO chunks (doc, page, ordinal, text) VALUES (?,?,?,?)",
            [(c.doc, c.page, c.ordinal, c.text) for c in chunks],
        )
        # Standalone FTS (not external-content: there, `SELECT rowid FROM fts`
        # reads through to the content table and the index silently stays empty).
        # chunk ids are monotone, so index everything past the FTS high-water mark.
        self.conn.execute(
            "INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks "
            "WHERE id > (SELECT COALESCE(MAX(rowid), 0) FROM chunks_fts)"
        )
        self.conn.commit()
        return cur.rowcount

    def add_embeddings(self, ids: Sequence[int], vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (chunk_id, dim, vec) VALUES (?,?,?)",
            [(int(i), vecs.shape[1], v.tobytes()) for i, v in zip(ids, vecs)],
        )
        self.conn.commit()

    # -- read ----------------------------------------------------------------

    def bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        """FTS5 BM25. Lower rank value = better; return as (id, score) with
        score negated so that, everywhere downstream, bigger is better.

        Every token is double-quoted (user text can contain FTS5 operators —
        a bare `AND` at end-of-query is a syntax error) and tokens are joined
        with OR: natural-language questions carry words no chunk contains, and
        FTS5's implicit AND would zero out recall. BM25 still ranks by term
        weight, so OR costs precision-at-1 nothing measurable at this scale."""
        tokens = _TOKEN.findall(query)
        if not tokens:
            return []
        sanitized = " OR ".join(f'"{t}"' for t in tokens)
        rows = self.conn.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (sanitized, k),
        ).fetchall()
        return [(int(r), -float(s)) for r, s in rows]

    def all_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.conn.execute(
            "SELECT chunk_id, dim, vec FROM embeddings ORDER BY chunk_id"
        ).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        dim = rows[0][1]
        mat = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32)
        return ids, mat.reshape(len(rows), dim)

    def get_chunks(self, ids: Sequence[int]) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, doc, page, ordinal, text FROM chunks WHERE id IN ({marks})",
            [int(i) for i in ids],
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        return [
            {"id": r[0], "doc": r[1], "page": r[2], "ordinal": r[3], "text": r[4]}
            for i in ids
            if (r := by_id.get(int(i)))
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
