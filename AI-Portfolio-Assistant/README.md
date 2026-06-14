---
title: AI Portfolio Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: RAG-powered chat service about Venkata's background and projects.
---

# AI Portfolio Assistant

A RAG-powered chat service that answers questions about Venkata's background, built with FastAPI and TF-IDF retrieval. Plug in any OpenAI-compatible LLM for generated answers — runs fully offline in extractive mode without one.

The retrieval layer is pure-Python TF-IDF (swappable for embeddings + a vector DB without touching the API layer). Documents live in `data/` as markdown files — adding new ones is as simple as dropping them in the folder.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: enable LLM-generated answers
cp .env.example .env   # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

uvicorn app.main:app --reload
```

Open http://localhost:8000 — the chat UI loads with suggested questions. Without an LLM key it still works (extractive mode).

Run tests:

```bash
pytest -q
```

---

## Deploy to Hugging Face Spaces

The YAML header at the top of this file configures a Docker-based Space. Push the repo, set the `LLM_*` variables as Secrets in the Space settings, and it's live.

```bash
git remote add space https://huggingface.co/spaces/<your-username>/ai-portfolio-assistant
git push space main
```

---

## API

`POST /api/chat`

```json
{ "question": "What LangGraph work has he done?", "top_k": 3 }
```

```json
{
  "answer": "He shipped a LangGraph multi-agent workflow to production at Pfizer... [experience]",
  "sources": [{ "source": "experience", "text": "...", "score": 0.43 }],
  "model": "offline-extractive",
  "grounded": true
}
```

`GET /api/health` → status, chunk count, LLM mode.

---

## Stack

- FastAPI + Pydantic v2
- Pure-Python TF-IDF retriever (no heavy ML deps, zero-config)
- Provider-neutral LLM layer — any OpenAI-compatible endpoint
- Single-file vanilla JS chat UI with typing indicator and source chips
- Docker-ready, ships Hugging Face Spaces config, Render blueprint, Fly.io config

---

## Author

**Venkata Subbaraju Dommaraju** — Senior AI & Data Engineer  
[venkatasubbarajudommaraju.com](https://venkatasubbarajudommaraju.com)
