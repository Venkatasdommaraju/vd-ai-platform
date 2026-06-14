# Technology Stack

Stack: LangGraph for stateful multi-agent orchestration, LangChain for LLM abstraction, FastAPI and Pydantic for the REST API layer.

LLM support: provider-neutral via OpenAI-compatible /chat/completions interface. Works with OpenAI, Anthropic (via proxy), AWS Bedrock (via LiteLLM), Ollama, vLLM, or any self-hosted model. Offline mode uses deterministic fallback responses.

Retrieval: pure-Python TF-IDF with cosine similarity, designed to swap for FAISS, pgvector, Elasticsearch, or OpenSearch without touching the agent graph.

Observability: LangSmith tracing with structured timing spans per node. Each agent step (planner, retriever, analyst, critic, synthesize) is a named span with input/output logged.

Deployment: Docker-ready, ships Hugging Face Spaces config (SDK: Docker), Render blueprint (render.yaml), and Fly.io config (fly.toml). Zero-config local run with no API keys needed.

Testing: offline pytest suite runs the full graph without any LLM or network calls, verifying routing logic and state transitions.
