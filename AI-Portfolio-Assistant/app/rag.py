from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

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
        self.df: Counter[str] = Counter()
        self.tfs: list[Counter[str]] = []
        for c in chunks:
            tf = Counter(tokenize(c.text))
            self.tfs.append(tf)
            for term in tf:
                self.df[term] += 1
        n = max(len(chunks), 1)
        self.idf = {t: math.log((1 + n) / (1 + dft)) + 1 for t, dft in self.df.items()}
        self.vectors = [self._vectorize(tf) for tf in self.tfs]
        self.norms = [math.sqrt(sum(v * v for v in vec.values())) or 1e-9 for vec in self.vectors]

    def _vectorize(self, tf: Counter[str]) -> dict[str, float]:
        return {t: freq * self.idf.get(t, 0.0) for t, freq in tf.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q_tf = Counter(tokenize(query))
        q_vec = self._vectorize(q_tf)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1e-9
        scored: list[tuple[Chunk, float]] = []
        for chunk, vec, norm in zip(self.chunks, self.vectors, self.norms):
            dot = sum(q_vec.get(t, 0.0) * w for t, w in vec.items())
            scored.append((chunk, dot / (q_norm * norm)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(c, s) for c, s in scored[:k] if s > 0]


class Retriever:
    def __init__(self, data_dir: str | Path = "data"):
        self.chunks = load_documents(data_dir)
        if not self.chunks:
            raise ValueError(f"No documents found in {data_dir!r}")
        self.backend = TfidfRetriever(self.chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        results = self.backend.search(query, k=k)
        return [
            {"source": c.source, "text": c.text, "score": round(score, 4)}
            for c, score in results
        ]
