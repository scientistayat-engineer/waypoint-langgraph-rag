"""The document library: a vector store plus a small registry of what is in it."""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

MAX_DOCS = 60
MAX_CHARS_PER_DOC = 800
MAX_UPLOAD_CHARS = 20_000


@dataclass
class DocRecord:
    id: str
    text: str
    source: str


def chunk_text(text: str, limit: int = MAX_CHARS_PER_DOC) -> list[str]:
    """Split text on blank lines, then pack paragraphs into chunks up to `limit`."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        pieces = [para]
        if len(para) > limit:
            pieces = re.split(r"(?<=[.!?])\s+", para)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if current and len(current) + len(piece) + 1 > limit:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
        if current and len(para) > limit:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return [c[:limit] for c in chunks]


def extract_pdf_text(data: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


class LibraryFull(Exception):
    pass


class DocumentLibrary:
    def __init__(self, embeddings: Embeddings, seed_file: Path | None = None) -> None:
        self._embeddings = embeddings
        self._seed_file = seed_file
        self._lock = threading.Lock()
        self._records: dict[str, DocRecord] = {}
        self._store = InMemoryVectorStore(embedding=embeddings)
        self.reset()

    def _seed(self) -> list[tuple[str, str]]:
        if self._seed_file and self._seed_file.exists():
            data = json.loads(self._seed_file.read_text(encoding="utf-8"))
            return [(d["text"], d.get("source", "sample")) for d in data]
        return []

    def reset(self) -> None:
        with self._lock:
            self._records = {}
            self._store = InMemoryVectorStore(embedding=self._embeddings)
        for text, source in self._seed():
            self.add(text, source)

    def add(self, text: str, source: str = "pasted") -> list[DocRecord]:
        chunks = chunk_text(text[:MAX_UPLOAD_CHARS])
        if not chunks:
            return []
        with self._lock:
            if len(self._records) + len(chunks) > MAX_DOCS:
                raise LibraryFull(f"The library holds at most {MAX_DOCS} passages.")
            ids = [uuid.uuid4().hex[:10] for _ in chunks]
            self._store.add_texts(
                chunks, metadatas=[{"doc_id": i} for i in ids], ids=ids
            )
            records = [DocRecord(i, c, source) for i, c in zip(ids, chunks)]
            for r in records:
                self._records[r.id] = r
        return records

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._records:
                return False
            self._store.delete(ids=[doc_id])
            del self._records[doc_id]
        return True

    def list(self) -> list[dict]:
        with self._lock:
            return [asdict(r) for r in self._records.values()]

    def __len__(self) -> int:
        return len(self._records)

    def search(self, query: str, k: int) -> list[dict]:
        with self._lock:
            if not self._records:
                return []
            hits = self._store.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in hits:
            doc_id = doc.metadata.get("doc_id", "")
            record = self._records.get(doc_id)
            results.append(
                {
                    "id": doc_id,
                    "text": doc.page_content,
                    "source": record.source if record else "unknown",
                    "score": round(float(score), 4),
                }
            )
        return results
