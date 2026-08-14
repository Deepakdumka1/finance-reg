"""Optional FastAPI backend (bonus marks).

Exposes the same ingestion/answering logic as three HTTP endpoints:

    POST /ingest   multipart PDF upload      -> {"files": 3, "chunks": 214}
    POST /ask      {"question", "top_k"}     -> {"answer", "sources": [...]}
    GET  /stats    -                         -> collection + model info

Run it from the project root:

    uvicorn api.main:app --reload

...or from inside the api/ folder:

    uvicorn main:app --reload

Then open http://localhost:8000/docs to try every endpoint.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the project root importable whether uvicorn is launched from the root
# (`uvicorn api.main:app`) or from inside api/ (`uvicorn main:app`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402
from ingest import collection_stats, ingest_paths  # noqa: E402
from rag import answer_question  # noqa: E402

app = FastAPI(
    title="Finance RAG API",
    description="Retrieval-Augmented Q&A over quarterly-report PDFs.",
    version="1.0.0",
)


# --- Schemas ---------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., examples=["What was total revenue last quarter?"])
    top_k: int = Field(config.DEFAULT_TOP_K, ge=1, le=20)


class Source(BaseModel):
    file: str
    page: object  # int in practice, but kept loose for "?" fallbacks


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    files: int
    chunks: int
    skipped: list[str] = []


class StatsResponse(BaseModel):
    provider: str
    collection: str
    total_chunks: int
    embedding_model: str
    llm_model: str


# --- Endpoints -------------------------------------------------------------
@app.get("/stats", response_model=StatsResponse)
def stats() -> dict:
    """Collection name, total chunks, and the models in use."""
    return collection_stats()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> dict:
    """Accept one or more PDF uploads, index them, and report counts."""
    saved_paths: list[str] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="finance_rag_ingest_"))
    for upload in files:
        dest = tmp_dir / (upload.filename or "upload.pdf")
        dest.write_bytes(await upload.read())
        saved_paths.append(str(dest))

    result = ingest_paths(saved_paths)
    return result


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> dict:
    """Answer a question from the indexed documents, with sources."""
    result = answer_question(payload.question, top_k=payload.top_k)
    return {"answer": result["answer"], "sources": result["sources"]}


@app.get("/")
def root() -> dict:
    return {"service": "Finance RAG API", "docs": "/docs"}
