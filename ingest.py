"""Load the documents in data/documents into the database.

Run once (and again whenever you change the documents):
    python ingest.py
"""
import json
from pathlib import Path

from app.llm import embed
from app.store import connect, init_db

DOCS_DIR = Path(__file__).resolve().parent / "data" / "documents"


def parse(path: Path) -> tuple[dict, str]:
    """Split a markdown file into its metadata header and its body text."""
    _, header, body = path.read_text(encoding="utf-8").split("---", 2)
    meta = dict(line.split(": ", 1) for line in header.strip().splitlines())
    return meta, body.strip()


def chunk(text: str) -> list[str]:
    """Split text into chunks. Here: one paragraph per chunk (simple and good enough)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def main() -> None:
    conn = connect()
    init_db(conn)
    for path in sorted(DOCS_DIR.glob("*.md")):
        meta, body = parse(path)
        conn.execute(
            "INSERT INTO documents (id, title, author, department, status, uploaded_at) "
            "VALUES (:id, :title, :author, :department, :status, :uploaded_at)",
            meta,
        )
        pieces = chunk(body)
        for text, vector in zip(pieces, embed(pieces)):
            conn.execute(
                "INSERT INTO chunks (document_id, text, embedding) VALUES (?, ?, ?)",
                (meta["id"], text, json.dumps(vector)),
            )
        print(f"Ingested {meta['id']} '{meta['title']}' ({len(pieces)} chunks)")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
