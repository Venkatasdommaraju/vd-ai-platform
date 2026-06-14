# Multi-Agent LangGraph Service

This service is a multi-agent system built on LangGraph. The agent graph has five nodes: planner, retriever, analyst, critic, and synthesize.

The planner breaks the user question into a retrieval strategy. The retriever pulls relevant context from the knowledge base using TF-IDF cosine similarity. The analyst generates a grounded answer using the retrieved context and prompt engineering. The critic reviews the answer for completeness and accuracy, either passing it or flagging it for revision. The synthesize node packages the final answer with source citations.

The revision loop is capped at MAX_REVISIONS (default 2) to prevent infinite loops.

The system supports any OpenAI-compatible LLM (OpenAI, AWS Bedrock via LiteLLM, Ollama, vLLM, Hugging Face Inference) and runs fully offline in deterministic mode when no LLM is configured. LangSmith tracing is optional and pluggable.

The retrieval backend defaults to in-memory TF-IDF but is designed to swap in Elasticsearch or any vector database with minimal changes.
