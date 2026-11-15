"""FastAPI wrapper, so a front end (or Postman/curl) can call the agent.

    uvicorn app.api:app --reload
    then open http://localhost:8000/docs
"""
import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent import run_agent

logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Document agent demo")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)  # input validation
    max_steps: int = Field(default=6, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str | None
    steps: int
    stopped: str
    trace: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return AskResponse(**run_agent(request.question, max_steps=request.max_steps))
