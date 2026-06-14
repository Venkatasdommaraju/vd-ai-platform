# AI & Data Engineering Portfolio

This repo contains three projects I built to demonstrate real AI/ML engineering work — not tutorials or toy examples, but things I'd actually ship. Each one runs locally out of the box and deploys to Hugging Face Spaces with Docker.

Live site: [venkatasubbarajudommaraju.com](https://venkatasubbarajudommaraju.com)

---

## Projects

### 1. AI Portfolio Assistant (`/AI-Portfolio-Assistant`)
A RAG-based chat service that answers questions about my background and experience. Built with FastAPI, a pure-Python TF-IDF retriever, and any OpenAI-compatible LLM (or runs offline without one). The knowledge base is just markdown files in `data/` — easy to swap in embeddings and a vector DB if needed.

**Run it:** `uvicorn app.main:app --reload --port 8081`  
**Live demo:** _coming soon_

---

### 2. Multi-Agent LangGraph (`/Multi-Agent-Langgraph`)
A five-node LangGraph agent pipeline: planner → retriever → analyst → critic → synthesize. The critic can send the answer back for revision (capped at 2 loops so it always terminates). Same LLM-agnostic setup — point it at HuggingFace, OpenAI, Ollama, or anything OpenAI-compatible.

**Run it:** `uvicorn app.main:app --reload --port 8082`  
**Live demo:** _coming soon_

---

### 3. IoT Anomaly Detector (`/IOT-Anomaly-Detector`)
A sensor anomaly detection service trained on 6,000 synthetic IoT readings. Gradient Boosting model (ROC-AUC 0.75) served via FastAPI. The web UI has live sliders for all 11 sensor inputs and a risk gauge that updates in real time. No LLM needed — pure ML.

**Run it:** `uvicorn service.main:app --reload --port 8080`  
**Live demo:** _coming soon_

---

## Running locally

Each project has its own `.venv`. From each project folder:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # fill in LLM_API_KEY if you want LLM mode
uvicorn app.main:app --reload
```

Or use the launcher scripts from the root:

```
.\start-backend.ps1    # pick which backend to start
```

---

## Stack

Python, FastAPI, LangGraph, LangChain, scikit-learn, TF-IDF, Pydantic v2, Docker, Hugging Face Spaces

---

**Venkata Subbaraju Dommaraju** — Senior AI & Data Engineer  
Downingtown, PA | [LinkedIn](https://linkedin.com/in/venkata-subbaraju-dommaraju) | Venkata.dommaraju@gmail.com
