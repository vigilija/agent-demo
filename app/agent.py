"""The agent loop: think -> act (call a tool) -> observe -> repeat.

This is the heart of the project. The model decides which tools to call and in
which order; this code runs those tools, feeds the results back, and enforces
the guardrails:

- max_steps:      the agent can't loop forever (cost and safety)
- allowed_tools:  the agent can only use tools on its allow-list
- error handling: a failing tool returns an error message instead of crashing,
                  so the model can recover (for example by retrying differently)
- trace/logging:  every step is recorded, so you can see and test what it did
"""
import json
import logging
import time
from datetime import date

from app.llm import CHAT_MODEL, get_client
from app.tools import TOOLS

log = logging.getLogger("agent")

MAX_TOOL_RESULT_CHARS = 6000  # keep tool output small so the context window doesn't explode

SYSTEM_PROMPT = """You are an internal knowledge assistant for a consulting company.
Today's date is {today}.

You have tools:
- search_documents: for questions about what documents SAY (policies, rules, processes).
- query_records: for facts ABOUT documents (who wrote them, when, how many, status).
You may call several tools, one after another, and use the result of one call to decide the next.
For example: first find documents with query_records, then search inside them with search_documents.

Rules:
- Answer ONLY from tool results. If the tools don't contain the answer, say you don't know.
- Mention the document id(s) you used, e.g. (DOC-001).
- Keep answers short and clear."""


def run_agent(
    question: str,
    *,
    client=None,
    allowed_tools: list[str] | None = None,
    max_steps: int = 6,
) -> dict:
    """Run the agent on one question. Returns the answer plus a full trace of the steps."""
    client = client or get_client()
    allowed = allowed_tools if allowed_tools is not None else list(TOOLS)
    tool_schemas = [TOOLS[name]["schema"] for name in allowed]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []

    for step in range(1, max_steps + 1):
        # 1. THINK: ask the model what to do next
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=tool_schemas or None,
            temperature=0,
        )
        message = response.choices[0].message

        # No tool calls means the model is done: this is the final answer
        if not message.tool_calls:
            log.info("step %d: final answer", step)
            return {"answer": message.content, "steps": step, "trace": trace, "stopped": "answered"}

        # Remember what the model asked for (required by the API before tool results)
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ],
            }
        )

        # 2. ACT: run each requested tool (the model may ask for several at once)
        for call in message.tool_calls:
            name = call.function.name
            started = time.perf_counter()
            try:
                if name not in allowed:
                    raise PermissionError(f"Tool '{name}' is not allowed for this agent.")
                args = json.loads(call.function.arguments or "{}")
                result = TOOLS[name]["function"](**args)
                ok = True
            except Exception as exc:  # report the error back to the model instead of crashing
                args = call.function.arguments
                result = {"error": f"{type(exc).__name__}: {exc}"}
                ok = False

            duration_ms = round((time.perf_counter() - started) * 1000)
            log.info("step %d: %s(%s) -> %s in %d ms", step, name, args, "ok" if ok else "ERROR", duration_ms)
            trace.append({"step": step, "tool": name, "arguments": args, "ok": ok, "duration_ms": duration_ms, "result": result})

            # 3. OBSERVE: give the result back to the model for the next round
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False)[:MAX_TOOL_RESULT_CHARS],
                }
            )

    log.warning("stopped: reached max_steps=%d", max_steps)
    return {
        "answer": "Sorry, I couldn't finish this question within the step limit.",
        "steps": max_steps,
        "trace": trace,
        "stopped": "max_steps",
    }
