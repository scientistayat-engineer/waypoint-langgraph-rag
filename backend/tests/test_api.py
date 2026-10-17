import json


def test_health_reports_demo_mode(demo_client):
    body = demo_client.get("/api/health").json()
    assert body["status"] == "ok" and body["demo_mode"] is True
    assert body["documents"] == 6


def test_graph_endpoint(demo_client):
    body = demo_client.get("/api/graph").json()
    assert {n["id"] for n in body["nodes"]} == {"START", "retrieve", "generate", "fallback", "END"}


def test_document_crud(demo_client):
    added = demo_client.post("/api/documents", json={"text": "Firebase Hosting serves static sites."})
    assert added.status_code == 201
    doc_id = added.json()["added"][0]["id"]
    assert len(demo_client.get("/api/documents").json()["documents"]) == 7
    assert demo_client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert demo_client.delete(f"/api/documents/{doc_id}").status_code == 404
    reset = demo_client.post("/api/documents/reset").json()
    assert len(reset["documents"]) == 6


def test_new_document_becomes_searchable(demo_client):
    demo_client.post("/api/documents", json={"text": "Vercel deploys frontend projects from GitHub."})
    result = demo_client.post("/api/ask", json={"question": "What does Vercel deploy?"}).json()
    assert "Vercel" in result["answer"]
    assert any("Vercel" in s["text"] for s in result["sources"])


def test_ask_validates_input(demo_client):
    assert demo_client.post("/api/ask", json={"question": "a"}).status_code == 422


def test_stream_event_order(demo_client):
    res = demo_client.post("/api/ask/stream", json={"question": "What does RAG do?"})
    events = [json.loads(line) for line in res.text.strip().splitlines()]
    kinds = [(e["event"], e.get("node")) for e in events]
    assert kinds == [
        ("start", None),
        ("node_start", "retrieve"),
        ("node_end", "retrieve"),
        ("node_start", "generate"),
        ("node_end", "generate"),
        ("done", None),
    ]
    assert events[2]["update"]["sources"]
    assert "retrieves" in events[-1]["answer"]
    assert events[-1]["path"] == ["retrieve", "generate"]


def test_missing_key_is_a_clean_401_in_real_mode(keyed_client):
    res = keyed_client.post("/api/ask", json={"question": "What does RAG do?"})
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "missing_api_key"


def test_header_key_is_accepted_in_real_mode(keyed_client):
    res = keyed_client.post(
        "/api/ask", json={"question": "What does RAG do?"}, headers={"X-Groq-Key": "test"}
    )
    assert res.status_code == 200


def test_unrelated_question_takes_the_fallback_route(demo_client):
    res = demo_client.post("/api/ask/stream", json={"question": "Who won the 2018 World Cup?"})
    events = [json.loads(line) for line in res.text.strip().splitlines()]
    assert [e["node"] for e in events if e["event"] == "node_start"] == ["retrieve", "fallback"]
    assert events[-1]["path"] == ["retrieve", "fallback"]


def _tiny_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 20 50 Td ({text}) Tj ET"
    objs = [
        "<</Type/Catalog/Pages 2 0 R>>",
        "<</Type/Pages/Kids[3 0 R]/Count 1>>",
        "<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        f"<</Length {len(stream)}>>\nstream\n{stream}\nendstream",
        "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF"
    return out.encode()


def test_pdf_upload_is_searchable_and_cited(demo_client):
    pdf = _tiny_pdf("Firebase Hosting serves static websites worldwide.")
    res = demo_client.post("/api/documents/upload", files={"file": ("notes.pdf", pdf, "application/pdf")})
    assert res.status_code == 201 and res.json()["added"][0]["source"] == "notes.pdf"
    answer = demo_client.post("/api/ask", json={"question": "What does Firebase Hosting serve?"}).json()
    assert "Firebase" in answer["answer"] and "[1]" in answer["answer"]


def test_empty_upload_is_rejected(demo_client):
    res = demo_client.post("/api/documents/upload", files={"file": ("a.txt", b"   ", "text/plain")})
    assert res.status_code == 422


def test_keyword_backend_with_real_llm_mode_needs_no_pytorch():
    from dataclasses import replace

    from app.engine import build_engine
    from conftest import make_settings

    settings = replace(make_settings(demo=False), embedding_backend="keyword")
    engine = build_engine(settings)
    assert engine.library.search("What does RAG do?", k=1)[0]["score"] > 0
