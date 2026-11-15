# Document agent demo

A small AI agent built from scratch in Python. It answers questions about a set of
company documents by choosing between two tools:

| Tool | Kind of lookup | Good for |
|---|---|---|
| `search_documents` | Semantic (vector) search over document text | "What does the policy say about…?" |
| `query_records` | Exact, structured query over document metadata | "Who / when / how many / which status?" |

The model decides which tool to use, in which order, and when it has enough to answer.
That decision-making loop is what makes it an **agent** and not a fixed RAG pipeline.

```
question ─► model ─► wants a tool? ── no ──► final answer
              ▲            │ yes
              │            ▼
              └── result ◄─ run tool (if allowed)   (max 6 rounds)
```

## Project structure

```
app/llm.py      connection to OpenAI / Azure OpenAI, embeddings
app/store.py    SQLite database: documents (metadata) + chunks (text + embeddings)
app/tools.py    the two tools + their JSON schemas for the model
app/agent.py    THE AGENT LOOP + guardrails (step limit, allow-list, errors, trace)
app/api.py      FastAPI endpoint POST /ask
ingest.py       loads data/documents/*.md into the database
cli.py          ask questions from the terminal and see every step
evals/          evaluation set: does the agent pick the right tools and answer correctly?
tests/          unit tests with a fake model (no API key needed)
Dockerfile      container image (non-root, secrets at runtime)
```

## How to run it

You need Python 3.11+ and an OpenAI API key (a few cents of usage) or an Azure OpenAI resource.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your key in .env

pytest -q                          # 10 tests, no API key needed
python ingest.py                   # build the database (calls the embedding API)
python cli.py "How many documents did Jan de Vries upload?"
python -m evals.run_evals          # run the evaluation set
uvicorn app.api:app --reload       # then open http://localhost:8000/docs
```

With Docker:

```bash
docker build -t document-agent .
docker run --env-file .env -p 8000:8000 document-agent
```

### Questions to try (and what to look for in the steps)

- *"How many days per week can I work from home?"* → only `search_documents`
- *"Which documents are still in draft?"* → only `query_records`
- *"Summarise what Sofia Martins' documents say about deadlines."* → **two steps**: first
  `query_records` to find her documents, then `search_documents` inside them
- *"What's the policy on dogs in the office?"* → should say it doesn't know, not make something up

---

## Weekend plan: build it yourself

The code here is complete, but you'll learn most (and answer interview questions best)
if you **build it step by step yourself**, using these files as the reference. Run something
after every step.

### Saturday: the building blocks (about 4–5 hours)

1. **Setup (30 min).** Virtual env, install packages, `.env` with your key.
   Test the connection with a 3-line script that asks the model "hello".
2. **Data + database (1 h).** Write `store.py` and `ingest.py`. Look at the database
   afterwards (`sqlite3 data/agent.db "select * from documents"`). Understand: metadata
   in one table, text + embeddings in the other.
3. **The tools as normal functions (1 h).** Write `search_documents` and `query_records`
   and call them directly from Python, without any AI. Make sure they return the right data.
4. **Tool schemas (30 min).** Add the JSON descriptions. Notice that the *description*
   is what the model reads to decide which tool to use: it's effectively part of the prompt.
5. **The agent loop (1.5 h).** Write `agent.py`. Start without guardrails, just the loop.
   Print every message so you can see the model asking for tools and getting results.

### Sunday: make it production-minded (about 4 hours)

6. **Guardrails (1 h).** Add the step limit, the tool allow-list, error handling and the trace.
   Try to break it: ask for something impossible, give a tool bad input.
7. **Tests (1 h).** Write the unit tests with the fake client. This proves *your* code works
   without depending on the model.
8. **Evaluation (45 min).** Run `evals/run_evals.py`. Change the system prompt or a tool
   description and see what breaks: this is why evals matter.
9. **API + Docker (1 h).** Wrap it in FastAPI, build the image, run it in a container.

### Optional extras (if you have time)

- **Sub-agent:** a "summariser" agent with its own prompt and *no* tools, that the main agent
  can call as a tool. This is the "sub-agents and hand-offs" from the job description.
- **Human approval:** a tool that "sends an email" but only after the user confirms.
- **Simple React front end** calling `POST /ask` and showing the steps (React is in their stack).
- **Azure:** run it with Azure OpenAI and deploy the container to Azure Container Apps.

---

## How to talk about it in the interview

> "In my project work I built RAG pipelines: fixed flows of search, then answer. To get
> hands-on with agents, I built a small agent from scratch in Python, without a framework,
> so I'd understand every part. It has two tools, a semantic search over documents and a
> structured query over metadata, and the model decides which to use, or chains them: for
> example, it first finds the documents by one author, then searches inside them.
> I added the guardrails you'd need in a bank: a step limit, a tool allow-list, errors
> returned to the model instead of crashing, a full trace of every step, and the model can't
> write SQL. It only fills in safe, parameterised filters. I test the loop with a fake model
> in unit tests, and I have a small evaluation set that checks both tool choice and answers,
> which I rerun after every prompt change."

### Questions they may ask, and short answers

- **Why no framework (LangChain, Semantic Kernel, OpenAI Agents SDK)?** To understand the loop
  myself. Frameworks are useful, but in a bank you need to know exactly what happens at every step.
- **How do you stop the agent running forever or getting expensive?** Step limit, small tool
  outputs, a cheap model for simple steps, and rate limits at the API gateway.
- **How do you make tools safe?** Narrow tools with validated arguments instead of open-ended
  ones (no raw SQL), an allow-list per agent, and the user's own permissions for data access
  (in production: Entra ID on-behalf-of).
- **How do you test something that isn't deterministic?** Unit-test my own code with a fake
  model; evaluate the model's behaviour with a fixed set of cases and track the pass rate over time.
- **How would you observe it in production?** Log every step with its tool, arguments,
  duration and result (the trace), send it to Application Insights or OpenTelemetry,
  and review failed and slow cases.
