"""The tools the agent can use.

Each tool has two parts:
1. a normal Python function that does the work
2. a JSON schema that tells the model what the tool does and which arguments it takes

The model never runs code itself: it only *asks* to call a tool by name with
arguments, and our agent loop runs the Python function.

Security note: query_records does NOT let the model write SQL. The model can only
fill in a few allowed filters, and we build a parameterised query ourselves.
That prevents SQL injection and stops the agent from reading or changing
anything we didn't intend.
"""
import json

from app.llm import embed
from app.store import connect, cosine_similarity


# ---------------------------------------------------------------- tool 1 ----
def search_documents(query: str, document_ids: list[str] | None = None, top_k: int = 3) -> dict:
    """Semantic search: find the text chunks whose meaning is closest to the query."""
    query_vector = embed([query])[0]
    conn = connect()
    sql = "SELECT c.document_id, d.title, c.text, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id"
    params: list = []
    if document_ids:
        sql += f" WHERE c.document_id IN ({','.join('?' * len(document_ids))})"
        params = list(document_ids)
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    scored = sorted(
        (
            {
                "document_id": r["document_id"],
                "title": r["title"],
                "text": r["text"],
                "score": round(cosine_similarity(query_vector, json.loads(r["embedding"])), 3),
            }
            for r in rows
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {"results": scored[: max(1, min(top_k, 5))]}


# ---------------------------------------------------------------- tool 2 ----
def query_records(
    author: str | None = None,
    department: str | None = None,
    status: str | None = None,
    uploaded_after: str | None = None,
    uploaded_before: str | None = None,
) -> dict:
    """Structured lookup: exact filters on document metadata. Good for who/when/how many."""
    conditions, params = [], []
    if author:
        conditions.append("author LIKE ?")
        params.append(f"%{author}%")
    if department:
        conditions.append("department = ?")
        params.append(department)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if uploaded_after:
        conditions.append("uploaded_at >= ?")
        params.append(uploaded_after)
    if uploaded_before:
        conditions.append("uploaded_at <= ?")
        params.append(uploaded_before)

    sql = "SELECT id, title, author, department, status, uploaded_at FROM documents"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY uploaded_at DESC"

    conn = connect()
    records = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return {"count": len(records), "records": records}


# ------------------------------------------------- schemas for the model ----
TOOLS = {
    "search_documents": {
        "function": search_documents,
        "schema": {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": (
                    "Semantic search over the CONTENT of company documents (policies, processes). "
                    "Use for questions about what a document says. Not suitable for counting or dates."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for, in natural language."},
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: only search inside these documents (e.g. ['DOC-002']).",
                        },
                        "top_k": {"type": "integer", "description": "Number of results (1-5). Default 3."},
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "query_records": {
        "function": query_records,
        "schema": {
            "type": "function",
            "function": {
                "name": "query_records",
                "description": (
                    "Exact lookup of document METADATA: id, title, author, department, status, upload date. "
                    "Use for who / when / how many / which status questions. Returns a count and the records. "
                    "Call with no filters to list all documents."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "author": {"type": "string", "description": "Author name or part of it."},
                        "department": {"type": "string", "enum": ["HR", "Operations", "Finance", "IT"]},
                        "status": {"type": "string", "enum": ["draft", "published"]},
                        "uploaded_after": {"type": "string", "description": "ISO date YYYY-MM-DD (inclusive)."},
                        "uploaded_before": {"type": "string", "description": "ISO date YYYY-MM-DD (inclusive)."},
                    },
                },
            },
        },
    },
}
