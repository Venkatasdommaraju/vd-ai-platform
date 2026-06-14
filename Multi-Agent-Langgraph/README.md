---
title: Multi-Agent LangGraph Service
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: LangGraph planner-retriever-analyst-critic multi-agent service behind FastAPI.
---

# Multi-Agent LangGraph Service

A LangGraph service with a five-node agent graph: planner → retriever → analyst → critic → synthesize. The critic can route back to the analyst for revision (capped at MAX_REVISIONS=2), so the loop always terminates. Works offline with deterministic fallbacks — no API key needed to try it.

The web UI shows a live agent trace panel on the right side of the screen, with each step lighting up as it completes.

---

## Agent graph

```
START → planner → retriever → analyst → critic ─┐
                                ▲                │ (conditional)
                                └──── revise ────┤
                                                 ▼
                                            synthesize → END
```

- **planner** — breaks the question into a retrieval strategy
- **retriever** — TF-IDF cosine similarity over the knowledge base (swap for Elasticsearch, FAISS, or OpenSearch without touching the graph)
- **analyst** — drafts a grounded, cited answer from retrieved context
- **critic** — passes or requests revision with a one-line reason
- **synthesize** — packages the final answer with sources

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: point at any OpenAI-compatible LLM
cp .env.example .env   # set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

uvicorn app.main:app --reload
```

Open http://localhost:8000

---

## API

`GET /api/health`

```json
{
  "status": "ok",
  "model": "offline-deterministic",
  "llm_enabled": false,
  "retrieval": "in-memory-tfidf",
  "max_revisions": 2
}
```

`POST /api/chat`

```json
{ "question": "How does the critic revision loop work?" }
```

```json
{
  "answer": "The critic reviews each draft...",
  "plan": "Retrieve context about the critic node and revision logic.",
  "critique": "PASS",
  "revisions": 1,
  "sources": [{ "source": "agentic", "text": "...", "score": 0.51 }],
  "model": "offline-deterministic",
  "grounded": true
}
```

---

## Stack

- LangGraph (stateful multi-agent graph, conditional edges, revision loop)
- LangChain (LLM abstraction — provider-neutral)
- FastAPI + Pydantic v2 (REST layer, request/response validation)
- Pure-Python TF-IDF retriever (pluggable for Elasticsearch, FAISS, pgvector)
- LangSmith tracing (set LANGCHAIN_API_KEY to enable)
- Docker, Hugging Face Spaces config, Render blueprint, Fly.io config

---

## Deploy to Hugging Face Spaces

```bash
git remote add space https://huggingface.co/spaces/<username>/agentic-microservice
git push space main
```

Set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL as Space Secrets to enable generated answers.

---

## Author

**Venkata Subbaraju Dommaraju** — Senior AI & Data Engineer  
[venkatasubbarajudommaraju.com](https://venkatasubbarajudommaraju.com)
