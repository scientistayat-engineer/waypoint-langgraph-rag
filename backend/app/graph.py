"""The LangGraph workflow: START -> retrieve -> generate -> END."""

from __future__ import annotations

from typing import Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .library import DocumentLibrary

SYSTEM_PROMPT = (
    "Answer only using the supplied context. "
    "Cite the passages you use as [1], [2] and so on. "
    "If it lacks the answer, say you don't know.\n\nContext:\n{context}"
)

NODE_INFO = {
    "retrieve": "Find the passages most similar to the question",
    "generate": "Ask the language model to answer from those passages only",
    "fallback": "Used when nothing in the library is close enough: no model call",
}


class State(TypedDict):
    """Data shared between nodes."""

    question: str
    k: int
    sources: list[dict]
    context: str
    answer: str


FALLBACK_ANSWER = (
    "I couldn't find anything close to that in the library. "
    "Try rephrasing, or add a passage about it."
)


def build_graph(
    library: DocumentLibrary, llm_factory: Callable[[str | None], object], min_score: float = 0.15
):
    def retrieve(state: State):
        sources = library.search(state["question"], state["k"])
        return {
            "sources": sources,
            "context": "\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, 1)),
        }

    def generate(state: State, config: RunnableConfig):
        key = (config.get("configurable") or {}).get("groq_api_key")
        llm = llm_factory(key)
        reply = llm.invoke(
            [
                ("system", SYSTEM_PROMPT.format(context=state["context"])),
                ("human", state["question"]),
            ]
        )
        return {"answer": str(reply.content).strip()}

    def fallback(state: State):
        return {"answer": FALLBACK_ANSWER}

    def route(state: State) -> str:
        best = max((s["score"] for s in state["sources"]), default=0.0)
        return "generate" if best >= min_score else "fallback"

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("fallback", fallback)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve", route, {"generate": "generate", "fallback": "fallback"}
    )
    builder.add_edge("generate", END)
    builder.add_edge("fallback", END)
    return builder.compile()


def describe_graph(graph) -> dict:
    """Nodes and edges of the compiled graph, in a frontend-friendly shape."""
    drawable = graph.get_graph()
    rename = {"__start__": "START", "__end__": "END"}
    nodes = [
        {
            "id": rename.get(n, n),
            "kind": "terminal" if n in rename else "node",
            "description": NODE_INFO.get(n, ""),
        }
        for n in drawable.nodes
    ]
    edges = [
        {
            "source": rename.get(e.source, e.source),
            "target": rename.get(e.target, e.target),
            "conditional": bool(e.conditional),
        }
        for e in drawable.edges
    ]
    return {"nodes": nodes, "edges": edges}
