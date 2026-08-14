"""Ingestion pipeline: load PDF -> chunk -> embed -> store in ChromaDB.

Run from the command line to index everything in the data/ folder:

    python ingest.py                 # index every PDF in data/
    python ingest.py data/q1.pdf ... # index specific files
    python ingest.py --reset         # wipe the store first, then index data/

Or import ingest_paths()/ingest_data_dir() from the UI and the API.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from vector_store import get_collection, reset_collection


def load_pdf_pages(path: str | Path) -> list[tuple[int, str]]:
    """Extract text from a PDF, one entry per page.

    Returns a list of (page_number, text) tuples. Page numbers are 1-based so
    they match what an analyst sees in a PDF viewer. Pages whose text cannot be
    extracted (blank or scanned images) are skipped.
    """
    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((index + 1, text))
    return pages


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive character splitter with the configured size and overlap.

    It tries to break on paragraph, then line, then sentence, then word
    boundaries, which keeps chunks readable and avoids cutting mid-number.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_pages(file_name: str, pages: list[tuple[int, str]]) -> list[dict]:
    """Split each page into chunks, carrying the file name and page number."""
    splitter = _build_splitter()
    chunks: list[dict] = []
    for page_number, text in pages:
        for chunk_text in splitter.split_text(text):
            cleaned = chunk_text.strip()
            if cleaned:
                chunks.append(
                    {"text": cleaned, "file": file_name, "page": page_number}
                )
    return chunks


def _chunk_id(file_name: str, page: int, position: int, text: str) -> str:
    """Deterministic id so re-indexing the same file updates instead of duplicating."""
    raw = f"{file_name}:{page}:{position}:{text[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def ingest_paths(paths: Iterable[str | Path]) -> dict:
    """Load, chunk, embed and store the given PDFs.

    Returns {"files": <int>, "chunks": <int>, "skipped": [names]} where files
    and chunks count only what was successfully indexed this call.
    """
    collection = get_collection()
    files_processed = 0
    total_chunks = 0
    skipped: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            skipped.append(f"{path.name} (not found)")
            continue

        pages = load_pdf_pages(path)
        if not pages:
            # No selectable text -> almost certainly a scanned image.
            skipped.append(f"{path.name} (no extractable text – scanned image?)")
            continue

        chunks = chunk_pages(path.name, pages)
        if not chunks:
            skipped.append(f"{path.name} (produced no chunks)")
            continue

        ids, documents, metadatas = [], [], []
        for position, chunk in enumerate(chunks):
            ids.append(_chunk_id(chunk["file"], chunk["page"], position, chunk["text"]))
            documents.append(chunk["text"])
            metadatas.append({"file": chunk["file"], "page": chunk["page"]})

        # upsert = insert or replace, so re-indexing a file is idempotent.
        # Batch to stay well under Chroma / OpenAI request limits.
        batch = 100
        for start in range(0, len(ids), batch):
            end = start + batch
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        files_processed += 1
        total_chunks += len(chunks)

    return {"files": files_processed, "chunks": total_chunks, "skipped": skipped}


def ingest_data_dir() -> dict:
    """Index every PDF currently sitting in the data/ folder."""
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    return ingest_paths(pdfs)


def collection_stats() -> dict:
    """Small summary used by the /stats endpoint and the UI sidebar."""
    collection = get_collection()
    return {
        "provider": config.PROVIDER,
        "collection": config.COLLECTION_NAME,
        "total_chunks": collection.count(),
        "embedding_model": config.active_embed_model(),
        "llm_model": config.active_llm_model(),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Index quarterly-report PDFs into ChromaDB.")
    parser.add_argument("paths", nargs="*", help="PDF files to index (default: everything in data/).")
    parser.add_argument("--reset", action="store_true", help="Delete the existing index before ingesting.")
    args = parser.parse_args()

    if args.reset:
        reset_collection()
        print("Cleared existing collection.")

    result = ingest_paths(args.paths) if args.paths else ingest_data_dir()

    print(f"{result['files']} files processed, {result['chunks']} chunks stored.")
    if result["skipped"]:
        print("Skipped:")
        for item in result["skipped"]:
            print(f"  - {item}")


if __name__ == "__main__":
    _cli()
