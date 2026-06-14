from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from .config import settings
from .llm import call_llm
from .observability import span
from .rag import Retriever


class AgentState(TypedDict, total=False):
    question: str
    plan: str
    contexts: list[dict]
    sources: list[str]
    draft: str
    answer: str
    critique: str
    needs_revision: bool
    revisions: int


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever()


def _format_context(contexts: list[dict]) -> str:
    if not contexts:
        return "(no relevant context found)"
    return "\n\n".join(f"[{c['source']}] {c['text']}" for c in contexts)


def _extractive(contexts: list[dict]) -> str:
    if not contexts:
        return "I don't have information on that in the knowledge base yet."
    lines = ["Based on the retrieved context:"]
    for c in contexts:
        lines.append(f"- ({c['source']}) {c['text']}")
    return "\n".join(lines)


def plan_node(state: AgentState) -> dict:
    with span("planner"):
        q = state["question"]
        plan = call_llm(
            "You are a planning agent. In 1–2 sentences, outline how to answer the "
            "user's question using retrieved enterprise context. Be specific and brief.",
            q,
        )
        return {"plan": plan or f"Retrieve the most relevant context and answer directly: {q}"}


def retrieve_node(state: AgentState) -> dict:
    with span("retriever"):
        hits = _retriever().search(state["question"], k=settings.top_k)
        sources = list(dict.fromkeys(h["source"] for h in hits))
        return {"contexts": hits, "sources": sources}


def analyst_node(state: AgentState) -> dict:
    with span("analyst"):
        contexts = state.get("contexts", [])
        prior = state.get("critique")
        revise_hint = f"\n\nA reviewer asked you to improve the previous answer: {prior}" if prior else ""
        user = (
            f"Context:\n{_format_context(contexts)}\n\n"
            f"Question: {state['question']}{revise_hint}\n\n"
            "Write a concise, accurate answer grounded ONLY in the context above. "
            "Cite sources inline like [source]. If the context is insufficient, say so."
        )
        answer = call_llm(
            "You are an analyst agent that answers strictly from provided context and cites sources.",
            user,
        ) or _extractive(contexts)
        return {"draft": answer, "answer": answer, "revisions": state.get("revisions", 0) + 1}


def critic_node(state: AgentState) -> dict:
    with span("critic"):
        draft = state.get("draft", "")
        contexts = state.get("contexts", [])
        verdict = call_llm(
            "You are a critic agent. Reply with exactly 'PASS' if the answer is "
            "well-grounded, accurate and complete, otherwise reply 'REVISE: <one-line reason>'.",
            f"Question: {state['question']}\n\nAnswer: {draft}",
        )
        if verdict is None:
            # no LLM, flag short or unsupported answers for one revision pass
            needs = (len(draft) < 60 or not contexts) and state.get("revisions", 0) < 2
            verdict = "REVISE: answer is thin or ungrounded" if needs else "PASS"
        else:
            needs = verdict.strip().upper().startswith("REVISE")
        return {"critique": verdict, "needs_revision": needs}


def synthesize_node(state: AgentState) -> dict:
    with span("synthesize"):
        return {"answer": state.get("draft", "")}


def route_after_critic(state: AgentState) -> str:
    if state.get("needs_revision") and state.get("revisions", 0) <= settings.max_revisions:
        return "analyst"
    return "synthesize"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", plan_node)
    g.add_node("retriever", retrieve_node)
    g.add_node("analyst", analyst_node)
    g.add_node("critic", critic_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"analyst": "analyst", "synthesize": "synthesize"})
    g.add_edge("synthesize", END)
    return g.compile()


def run(question: str) -> dict[str, Any]:
    final = build_graph().invoke({"question": question, "revisions": 0})
    return dict(final)
