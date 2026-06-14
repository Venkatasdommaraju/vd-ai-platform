from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .llm import generate, llm_enabled, _cfg
from .rag import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("DATA_DIR", str(ROOT / "data"))

app = FastAPI(title="AI Portfolio Assistant", version="1.0.0")
retriever = Retriever(DATA_DIR)


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


class Source(BaseModel):
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    grounded: bool


@app.get("/api/health")
def health() -> dict:
    _, _, model = _cfg()
    return {"status": "ok", "chunks": len(retriever.chunks),
            "model": model or "offline-extractive",
            "llm_enabled": llm_enabled()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    contexts = retriever.search(req.question, k=req.top_k)
    answer = generate(req.question, contexts)
    return ChatResponse(
        answer=answer,
        sources=[Source(**c) for c in contexts],
        model=_cfg()[2] if llm_enabled() else "offline-extractive",
        grounded=bool(contexts),
    )


WEB_DIR = ROOT / "web"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
