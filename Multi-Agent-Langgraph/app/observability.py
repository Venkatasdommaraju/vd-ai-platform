from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

logger = logging.getLogger("agentic")


def configure_observability() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if os.environ.get("LANGCHAIN_API_KEY") and not os.environ.get("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", "agentic-microservice")
        logger.info("LangSmith tracing enabled (project=%s)", os.environ["LANGCHAIN_PROJECT"])
    else:
        logger.info("LangSmith tracing not configured — set LANGCHAIN_API_KEY to enable.")


@contextmanager
def span(name: str):
    start = time.perf_counter()
    logger.info("→ %s", name)
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        logger.info("← %s (%.0f ms)", name, ms)
