from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    llm_base_url: str = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_model: str = os.environ.get("LLM_MODEL", "")

    data_dir: str = os.environ.get("DATA_DIR", "data")
    top_k: int = int(os.environ.get("TOP_K", "4"))

    es_url: str = os.environ.get("ELASTICSEARCH_URL", "")
    es_index: str = os.environ.get("ELASTICSEARCH_INDEX", "kb")

    max_revisions: int = int(os.environ.get("MAX_REVISIONS", "2"))


settings = Settings()


def llm_enabled() -> bool:
    return bool(settings.llm_api_key and settings.llm_model)
