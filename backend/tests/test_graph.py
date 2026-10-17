from app.engine import build_engine
from app.library import chunk_text
from conftest import make_settings


def test_graph_branches_after_retrieve():
    layout = build_engine(make_settings(demo=True)).layout
    assert {n["id"] for n in layout["nodes"]} == {"START", "retrieve", "generate", "fallback", "END"}
    pairs = {(e["source"], e["target"]) for e in layout["edges"]}
    assert pairs == {
        ("START", "retrieve"), ("retrieve", "generate"), ("retrieve", "fallback"),
        ("generate", "END"), ("fallback", "END"),
    }


def test_retrieve_returns_the_most_relevant_passage_first():
    engine = build_engine(make_settings(demo=True))
    hits = engine.library.search("What does RAG do?", k=2)
    assert "RAG" in hits[0]["text"]
    assert hits[0]["score"] >= hits[1]["score"]


def test_graph_answers_from_context_and_admits_ignorance():
    engine = build_engine(make_settings(demo=True))
    base = {"k": 2, "sources": [], "context": "", "answer": ""}
    good = engine.graph.invoke({**base, "question": "What does RAG do?"})
    assert "retrieves relevant documents" in good["answer"]
    bad = engine.graph.invoke({**base, "question": "Who won the 2018 World Cup?"})
    assert "couldn't find anything close" in bad["answer"]


def test_chunk_text_respects_limit_and_keeps_content():
    text = "First paragraph.\n\nSecond paragraph.\n\n" + ("Long sentence here. " * 80)
    chunks = chunk_text(text, limit=200)
    assert chunks and all(len(c) <= 200 for c in chunks)
    assert "First paragraph." in chunks[0]
