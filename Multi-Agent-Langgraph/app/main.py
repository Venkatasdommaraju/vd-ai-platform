from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings, llm_enabled
from .graph import run
from .observability import configure_observability

configure_observability()

app = FastAPI(
    title="Multi-Agent LangGraph Service",
    version="1.0.0",
    description="LangGraph planner → retriever → analyst → critic → synthesize, behind FastAPI + Pydantic.",
)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class Source(BaseModel):
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    plan: str
    critique: str
    revisions: int
    sources: list[Source]
    model: str
    grounded: bool


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": settings.llm_model or "offline-deterministic",
        "llm_enabled": llm_enabled(),
        "retrieval": "elasticsearch" if settings.es_url else "in-memory-tfidf",
        "max_revisions": settings.max_revisions,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = run(req.question)
    contexts = state.get("contexts", [])
    return ChatResponse(
        answer=state.get("answer", ""),
        plan=state.get("plan", ""),
        critique=state.get("critique", ""),
        revisions=state.get("revisions", 0),
        sources=[Source(**c) for c in contexts],
        model=settings.llm_model if llm_enabled() else "offline-deterministic",
        grounded=bool(contexts),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
