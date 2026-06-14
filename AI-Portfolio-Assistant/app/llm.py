from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about Venkata Subbaraju Dommaraju, "
    "a Senior AI/Data Engineer & Technical Lead, using ONLY the provided context. "
    "If the context does not contain the answer, say so honestly. Be concise, "
    "professional, and specific. Cite the source name in brackets, e.g. [experience]."
)


def _cfg():
    return (
        os.environ.get("LLM_BASE_URL", "").rstrip("/"),
        os.environ.get("LLM_API_KEY", ""),
        os.environ.get("LLM_MODEL", ""),
    )


def llm_enabled() -> bool:
    base, key, model = _cfg()
    return bool(base and key and model)


def _format_context(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']}] {c['text']}" for c in contexts)


def _extractive_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "I don't have information on that in my knowledge base yet."
    lines = ["Here's what I found relevant to your question:\n"]
    for c in contexts:
        lines.append(f"- ({c['source']}) {c['text']}")
    lines.append("\n[offline mode — configure an LLM endpoint to get a synthesized answer]")
    return "\n".join(lines)


def generate(question: str, contexts: list[dict]) -> str:
    if not llm_enabled():
        return _extractive_answer(question, contexts)

    base, key, model = _cfg()
    user_msg = (
        f"Context:\n{_format_context(contexts)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"[llm] {exc}")
        return _extractive_answer(question, contexts)
