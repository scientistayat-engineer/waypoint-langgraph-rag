"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT_DIR = Path(__file__).resolve().parents[2]


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None
    groq_model: str
    embedding_model: str
    demo_mode: bool
    embedding_backend: str
    top_k: int
    min_score: float
    cors_origins: list[str]
    frontend_dir: Path
    seed_file: Path


def get_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", "*")
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        demo_mode=_flag("DEMO_MODE", False),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "huggingface").strip().lower(),
        min_score=float(os.getenv("MIN_SCORE", "0.15")),
        top_k=max(1, min(int(os.getenv("TOP_K", "2")), 10)),
        cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        frontend_dir=Path(os.getenv("FRONTEND_DIR", ROOT_DIR / "frontend")),
        seed_file=Path(os.getenv("SEED_FILE", ROOT_DIR / "data" / "sample_docs.json")),
    )
