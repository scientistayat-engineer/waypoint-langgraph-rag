"""Wires settings, library, LLM and graph together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import Settings
from .demo import DemoLLM, KeywordEmbeddings
from .graph import build_graph, describe_graph
from .library import DocumentLibrary


@dataclass
class Engine:
    settings: Settings
    library: DocumentLibrary
    graph: object
    llm_factory: Callable[[str | None], object]

    @property
    def layout(self) -> dict:
        return describe_graph(self.graph)


def build_engine(settings: Settings) -> Engine:
    # "keyword" retrieval needs no PyTorch, so it starts instantly and stays small.
    if settings.demo_mode or settings.embedding_backend == "keyword":
        embeddings = KeywordEmbeddings()
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model, model_kwargs={"device": "cpu"}
        )

    if settings.demo_mode:
        demo_llm = DemoLLM()

        def llm_factory(_key):
            return demo_llm

    else:
        from langchain_groq import ChatGroq

        def llm_factory(key):
            return ChatGroq(model=settings.groq_model, temperature=0, api_key=key or None)

    library = DocumentLibrary(embeddings, settings.seed_file)
    graph = build_graph(library, llm_factory, settings.min_score)
    return Engine(settings, library, graph, llm_factory)
