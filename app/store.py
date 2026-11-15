"""The 'database': SQLite with two tables.

- documents: structured metadata (author, department, status, date)
  -> used for exact lookups (counts, filters, dates)
- chunks: pieces of document text plus their embeddings
  -> used for semantic (meaning-based) search

In your Capgemini project both lived in Cosmos DB; SQLite keeps this demo
free and local, and the idea is exactly the same.
"""
import math
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).resolve().parent.parent / "data" / "agent.db"))


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS documents;
        CREATE TABLE documents (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            author      TEXT NOT NULL,
            department  TEXT NOT NULL,
            status      TEXT NOT NULL,
            uploaded_at TEXT NOT NULL      -- ISO date, e.g. 2026-10-14
        );
        CREATE TABLE chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL REFERENCES documents(id),
            text        TEXT NOT NULL,
            embedding   TEXT NOT NULL      -- JSON list of floats
        );
        """
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """How similar two embeddings are: 1.0 = same meaning, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
