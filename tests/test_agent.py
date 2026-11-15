"""Unit tests that run WITHOUT an API key.

The model is replaced by a fake client that returns scripted answers, and
embeddings by a simple word-counting function. This tests our own code (the
loop, the guardrails, the tools), not the model.

    pytest -q
"""
import hashlib
import json
from types import SimpleNamespace

import pytest

import app.store as store
import app.tools as tools
import ingest
from app.agent import run_agent


# ------------------------------------------------------------- fixtures ----
def fake_embed(texts):
    """Very simple 'embedding': count words into 1024 buckets. Similar words -> similar vectors."""
    vectors = []
    for text in texts:
        vector = [0.0] * 1024
        for word in text.lower().split():
            vector[int(hashlib.md5(word.strip(".,?").encode()).hexdigest(), 16) % 1024] += 1
        vectors.append(vector)
    return vectors


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(ingest, "embed", fake_embed)
    monkeypatch.setattr(tools, "embed", fake_embed)
    ingest.main()


def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class FakeClient:
    """Pretends to be the OpenAI client and returns the scripted messages in order."""

    def __init__(self, scripted_messages):
        self.scripted = list(scripted_messages)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.scripted.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def says(text):
    return SimpleNamespace(content=text, tool_calls=None)


def calls(*tool_calls):
    return SimpleNamespace(content=None, tool_calls=list(tool_calls))


# ---------------------------------------------------------------- tools ----
def test_query_records_counts_by_author():
    result = tools.query_records(author="Jan de Vries")
    assert result["count"] == 2
    assert {r["id"] for r in result["records"]} == {"DOC-002", "DOC-004"}


def test_query_records_filters_by_date_and_status():
    assert tools.query_records(status="draft")["count"] == 1
    assert tools.query_records(uploaded_after="2026-10-01")["count"] == 3


def test_query_records_is_safe_against_sql_injection():
    assert tools.query_records(author="x' OR '1'='1")["count"] == 0


def test_search_documents_finds_relevant_text():
    top = tools.search_documents("budget for a desk, chair or monitor")["results"][0]
    assert top["document_id"] == "DOC-001"


def test_search_documents_can_be_limited_to_documents():
    results = tools.search_documents("report", document_ids=["DOC-005"])["results"]
    assert {r["document_id"] for r in results} == {"DOC-005"}


# ----------------------------------------------------------- agent loop ----
def test_agent_answers_directly_without_tools():
    client = FakeClient([says("Hello!")])
    result = run_agent("Hi", client=client)
    assert result["answer"] == "Hello!" and result["trace"] == []


def test_agent_runs_multi_step_plan():
    client = FakeClient([
        calls(tool_call("1", "query_records", {"author": "Sofia"})),
        calls(tool_call("2", "search_documents", {"query": "deadline", "document_ids": ["DOC-001", "DOC-005"]})),
        says("Report incidents within one hour (DOC-005)."),
    ])
    result = run_agent("What do Sofia's documents say about deadlines?", client=client)

    assert [t["tool"] for t in result["trace"]] == ["query_records", "search_documents"]
    assert result["trace"][0]["result"]["count"] == 2
    assert result["stopped"] == "answered"
    # the tool result was sent back to the model before its next decision
    assert client.requests[1]["messages"][-1]["role"] == "tool"


def test_agent_stops_at_step_limit():
    looping = [calls(tool_call(str(i), "query_records", {})) for i in range(10)]
    result = run_agent("loop forever", client=FakeClient(looping), max_steps=3)
    assert result["stopped"] == "max_steps" and len(result["trace"]) == 3


def test_agent_blocks_tools_that_are_not_allowed():
    client = FakeClient([calls(tool_call("1", "search_documents", {"query": "x"})), says("I can't search.")])
    result = run_agent("search", client=client, allowed_tools=["query_records"])
    assert result["trace"][0]["ok"] is False
    assert "not allowed" in result["trace"][0]["result"]["error"]
    # the blocked tool was not even offered to the model
    assert [t["function"]["name"] for t in client.requests[0]["tools"]] == ["query_records"]


def test_agent_recovers_from_bad_tool_arguments():
    client = FakeClient([calls(tool_call("1", "query_records", {"colour": "blue"})), says("Sorry.")])
    result = run_agent("bad args", client=client)
    assert result["trace"][0]["ok"] is False and result["answer"] == "Sorry."
