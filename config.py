"""Central configuration for the Finance RAG system.

Every tunable value lives here so ingest.py, rag.py, app.py and the FastAPI
backend all share one source of truth. Values can be overridden with
environment variables (loaded from a .env file) where it makes sense.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file sitting next to this file (if present).
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Paths -----------------------------------------------------------------
# Folder where the persisted ChromaDB lives. Because it is an absolute path,
# the store is found no matter which directory you launch the app from.
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
# Where uploaded / downloaded PDFs are kept.
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_reports")

# --- Models ----------------------------------------------------------------
# Provider backend: "openai" (default, assignment stack) or "ollama" (local, free).
PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
# Keep the answering model close to deterministic (assignment asks for 0–0.2).
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Models (Ollama, local) ------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def active_embed_model() -> str:
    """Name of the embedding model actually in use, per PROVIDER."""
    return OLLAMA_EMBED_MODEL if PROVIDER == "ollama" else EMBEDDING_MODEL


def active_llm_model() -> str:
    """Name of the answering model actually in use, per PROVIDER."""
    return OLLAMA_LLM_MODEL if PROVIDER == "ollama" else LLM_MODEL

# --- Chunking --------------------------------------------------------------
# 1000 chars with 150 overlap sits in the middle of the allowed range and, per
# the assignment hint, is large enough to keep most financial tables inside a
# single chunk while staying cheap to embed.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval -------------------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))

# --- API key ---------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- FastAPI backend URL (used only when the UI runs in "API mode") --------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def require_api_key() -> str:
    """Return the OpenAI key or raise a clear error if it is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or export OPENAI_API_KEY in your shell."
        )
    return OPENAI_API_KEY
