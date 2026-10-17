"""FastAPI application: JSON API, streaming endpoint, and the static frontend."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings, get_settings
from .engine import Engine, build_engine
from .library import LibraryFull, extract_pdf_text
from .schemas import AddDocumentRequest, AskRequest


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = {"engine": engine}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        get_engine()  # load the embedding model once, at startup
        yield

    app = FastAPI(title="Waypoint", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    def get_engine() -> Engine:
        if state["engine"] is None:
            state["engine"] = build_engine(settings)
        return state["engine"]

    def resolve_key(header_key: str | None) -> str | None:
        key = header_key or settings.groq_api_key
        if not key and not settings.demo_mode:
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "missing_api_key",
                    "message": "Add your Groq API key to ask a question.",
                },
            )
        return key

    # ---- meta ----
    @app.get("/api/health")
    def health():
        eng = get_engine()
        return {
            "status": "ok",
            "version": __version__,
            "demo_mode": settings.demo_mode,
            "llm": "demo" if settings.demo_mode else settings.groq_model,
            "embeddings": "keyword"
            if settings.demo_mode or settings.embedding_backend == "keyword"
            else settings.embedding_model,
            "documents": len(eng.library),
            "server_has_key": bool(settings.groq_api_key),
            "top_k": settings.top_k,
        }

    @app.get("/api/graph")
    def graph_layout():
        return get_engine().layout

    # ---- library ----
    @app.get("/api/documents")
    def list_documents():
        return {"documents": get_engine().library.list()}

    @app.post("/api/documents", status_code=201)
    def add_document(body: AddDocumentRequest):
        try:
            records = get_engine().library.add(body.text, body.source)
        except LibraryFull as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        if not records:
            raise HTTPException(status_code=422, detail="No text to add.")
        return {"added": [r.__dict__ for r in records]}

    @app.post("/api/documents/upload", status_code=201)
    async def upload_document(file: UploadFile):
        data = await file.read()
        if len(data) > 5_000_000:
            raise HTTPException(status_code=413, detail="Keep files under 5 MB.")
        name = (file.filename or "upload")[:60]
        try:
            text = (
                extract_pdf_text(data)
                if name.lower().endswith(".pdf")
                else data.decode("utf-8", errors="ignore")
            )
        except Exception:
            raise HTTPException(status_code=422, detail="Couldn't read that PDF.")
        if not text.strip():
            raise HTTPException(
                status_code=422, detail="No selectable text found. Scanned PDFs aren't supported."
            )
        try:
            records = get_engine().library.add(text, name)
        except LibraryFull as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"added": [r.__dict__ for r in records]}

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str):
        if not get_engine().library.remove(doc_id):
            raise HTTPException(status_code=404, detail="Passage not found.")
        return {"deleted": doc_id}

    @app.post("/api/documents/reset")
    def reset_documents():
        eng = get_engine()
        eng.library.reset()
        return {"documents": eng.library.list()}

    # ---- asking ----
    def _inputs(body: AskRequest, eng: Engine) -> dict:
        return {
            "question": body.question.strip(),
            "k": body.k or settings.top_k,
            "sources": [],
            "context": "",
            "answer": "",
        }

    @app.post("/api/ask")
    def ask(body: AskRequest, x_groq_key: str | None = Header(default=None)):
        eng = get_engine()
        key = resolve_key(x_groq_key)
        started = time.perf_counter()
        try:
            result = eng.graph.invoke(
                _inputs(body, eng), config={"configurable": {"groq_api_key": key}}
            )
        except Exception as exc:  # surfaced to the client as a clean message
            raise HTTPException(
                status_code=502,
                detail={"code": "llm_error", "message": _clean_error(exc)},
            )
        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    @app.post("/api/ask/stream")
    def ask_stream(body: AskRequest, x_groq_key: str | None = Header(default=None)):
        eng = get_engine()
        key = resolve_key(x_groq_key)

        def events() -> Iterator[str]:
            def emit(payload: dict) -> str:
                return json.dumps(payload) + "\n"

            run_started = time.perf_counter()
            yield emit({"event": "start", "question": body.question})
            step_started = time.perf_counter()
            final: dict = {}
            path: list[str] = []
            try:
                for update in eng.graph.stream(
                    _inputs(body, eng),
                    config={"configurable": {"groq_api_key": key}},
                    stream_mode=["tasks", "updates"],
                ):
                    mode, payload = update
                    if mode == "tasks":
                        if "result" not in payload:  # a task is starting
                            step_started = time.perf_counter()
                            yield emit({"event": "node_start", "node": payload["name"]})
                        continue
                    for node, delta in payload.items():
                        now = time.perf_counter()
                        path.append(node)
                        final.update(delta)
                        yield emit(
                            {
                                "event": "node_end",
                                "node": node,
                                "update": delta,
                                "elapsed_ms": round((now - step_started) * 1000),
                            }
                        )
                yield emit(
                    {
                        "event": "done",
                        "answer": final.get("answer", ""),
                        "sources": final.get("sources", []),
                        "path": path,
                        "total_ms": round((time.perf_counter() - run_started) * 1000),
                    }
                )
            except Exception as exc:
                yield emit(
                    {"event": "error", "code": "llm_error", "message": _clean_error(exc)}
                )

        return StreamingResponse(events(), media_type="application/x-ndjson")

    if settings.frontend_dir.exists():
        app.mount(
            "/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend"
        )
    return app


def _clean_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if "401" in text or "invalid_api_key" in text.lower():
        return "Groq rejected the API key. Check it and try again."
    return text[:300]


app = create_app()
