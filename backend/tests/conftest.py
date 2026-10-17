import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ROOT_DIR, Settings  # noqa: E402
from app.engine import build_engine  # noqa: E402
from app.main import create_app  # noqa: E402


def make_settings(demo: bool) -> Settings:
    return Settings(
        groq_api_key=None,
        groq_model="test-model",
        embedding_model="test-embeddings",
        demo_mode=demo,
        embedding_backend="huggingface",
        top_k=2,
        min_score=0.15,
        cors_origins=["*"],
        frontend_dir=ROOT_DIR / "frontend",
        seed_file=ROOT_DIR / "data" / "sample_docs.json",
    )


@pytest.fixture()
def demo_client():
    settings = make_settings(demo=True)
    return TestClient(create_app(settings, build_engine(settings)))


@pytest.fixture()
def keyed_client():
    """Real-mode key handling, but with the demo components so no network is used."""
    demo_settings = make_settings(demo=True)
    engine = build_engine(demo_settings)
    return TestClient(create_app(make_settings(demo=False), engine))
