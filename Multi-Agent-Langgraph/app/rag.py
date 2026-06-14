from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def load_documents(data_dir: str | Path) -> list[Chunk]:
    data_dir = Path(data_dir)
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        for i, para in enumerate(p.strip() for p in re.split(r"\n\s*\n", raw)):
            if len(para) < 40:
                continue
            chunks.append(Chunk(id=f"{path.stem}-{i}", source=path.stem, text=para))
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        df: Counter[str] = Counter()
        self.tfs: list[Counter[str]] = []
        for c in chunks:
            tf = Counter(tokenize(c.text))
            self.tfs.append(tf)
            for term in tf:
                df[term] += 1
        n = max(len(chunks), 1)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = [self._vec(tf) for tf in self.tfs]
        self.norms = [math.sqrt(sum(v * v for v in vec.values())) or 1e-9 for vec in self.vectors]

    def _vec(self, tf: Counter[str]) -> dict[str, float]:
        return {t: freq * self.idf.get(t, 0.0) for t, freq in tf.items()}

    def search(self, query: str, k: int = 4) -> list[dict]:
        q_vec = self._vec(Counter(tokenize(query)))
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1e-9
        scored = []
        for chunk, vec, norm in zip(self.chunks, self.vectors, self.norms):
            dot = sum(q_vec.get(t, 0.0) * w for t, w in vec.items())
            scored.append((chunk, dot / (q_norm * norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"source": c.source, "text": c.text, "score": round(s, 4)}
            for c, s in scored[:k] if s > 0
        ]


class ElasticsearchRetriever:
    def __init__(self, url: str, index: str):
        from elasticsearch import Elasticsearch
        self.es = Elasticsearch(url)
        self.index = index

    def search(self, query: str, k: int = 4) -> list[dict]:
        res = self.es.search(
            index=self.index,
            query={"match": {"text": query}},
            size=k,
        )
        return [
            {"source": h["_source"].get("source", "kb"), "text": h["_source"]["text"], "score": h["_score"]}
            for h in res["hits"]["hits"]
        ]


class Retriever:
    def __init__(self, data_dir: Optional[str] = None):
        if settings.es_url:
            self.backend = ElasticsearchRetriever(settings.es_url, settings.es_index)
        else:
            chunks = load_documents(data_dir or settings.data_dir)
            if not chunks:
                raise ValueError(f"No documents found in {data_dir or settings.data_dir!r}")
            self.backend = TfidfRetriever(chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        return self.backend.search(query, k=k)
