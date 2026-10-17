"""Lightweight stand-ins used in demo mode.

Demo mode needs no API key and downloads no model, so the whole app can be
tried (or tested) offline. Retrieval is a hashed bag of words and the "LLM"
simply returns the best matching sentence from the retrieved context.
"""

from __future__ import annotations

import math
import re
import zlib
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "which",
    "who", "why", "with", "that", "this", "can", "you", "your",
}


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class KeywordEmbeddings(Embeddings):
    """Hashed bag-of-words vectors: cheap, deterministic, and good enough to demo."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in tokens(text):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class DemoLLM:
    """Extractive answerer that mimics the grounded behaviour of the real prompt."""

    def invoke(self, messages):
        system = messages[0][1]
        question = messages[-1][1]
        context = system.split("Context:\n", 1)[-1]
        wanted = set(tokens(question))
        best, best_score, cite = "", 0, 1
        for line in context.split("\n"):
            match = re.match(r"\[(\d+)\]\s*(.*)", line)
            number, body = (int(match.group(1)), match.group(2)) if match else (1, line)
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                score = len(wanted & set(tokens(sentence)))
                if score > best_score:
                    best, best_score, cite = sentence.strip(), score, number
        if not best:
            return SimpleNamespace(
                content="I don't know based on the documents in the library."
            )
        return SimpleNamespace(content=f"{best} [{cite}] (demo mode: no language model used)")
