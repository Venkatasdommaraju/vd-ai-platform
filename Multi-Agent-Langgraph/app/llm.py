from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .config import settings, llm_enabled


@lru_cache(maxsize=1)
def get_chat_llm():
    if not llm_enabled():
        return None
    from langchain_openai import ChatOpenAI

    kwargs = dict(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        temperature=0.2,
        max_tokens=700,
    )
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**kwargs)


def call_llm(system: str, user: str) -> Optional[str]:
    llm = get_chat_llm()
    if llm is None:
        return None
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return (resp.content or "").strip() or None
    except Exception as exc:
        print(f"[llm] {exc}")
        return None
