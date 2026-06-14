from pathlib import Path

from app.rag import Retriever, tokenize, load_documents

DATA = Path(__file__).resolve().parent.parent / "data"


def test_tokenize():
    assert tokenize("Hello, AWS-Bedrock!") == ["hello", "aws", "bedrock"]


def test_documents_load():
    chunks = load_documents(DATA)
    assert len(chunks) >= 5
    assert all(c.text for c in chunks)


def test_retrieval_relevance():
    r = Retriever(DATA)
    results = r.search("What RAG and LLM work has he done?", k=3)
    assert results, "expected at least one result"
    joined = " ".join(x["text"].lower() for x in results)
    assert "rag" in joined or "llm" in joined or "bedrock" in joined


def test_retrieval_project_query():
    r = Retriever(DATA)
    results = r.search("Tell me about the REMS project", k=2)
    assert any("rems" in x["text"].lower() for x in results)


def test_scores_descending():
    r = Retriever(DATA)
    results = r.search("AWS cloud experience", k=3)
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)
