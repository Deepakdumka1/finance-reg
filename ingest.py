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
import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from vector_store import get_collection, reset_collection


def load_pdf_pages(path: str | Path) -> list[tuple[int, str]]:
    """Extract text from a PDF, one entry per page.

    Returns (page_number, text) tuples. Page numbers are 1-based to match a PDF
    viewer. Pages with no extractable text (blank or scanned) are skipped.
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
    """Recursive character splitter with the configured size and overlap."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _derive_quarter(file_name: str) -> str:
    """Best-effort quarter label from a file name, e.g. 'Q1 FY26'.

    Falls back to the file stem when no Q#/FY## pattern is present. The label is
    stored as metadata AND prefixed into each chunk's text, so the quarter is
    embedded and becomes part of what retrieval matches on - the key fix for
    near-identical quarterly press releases (Stage 6 of the guide).
    """
    stem = Path(file_name).stem
    q = re.search(r"Q\s*([1-4])", stem, re.IGNORECASE)
    fy = re.search(r"FY\s*'?\s*(\d{2,4})", stem, re.IGNORECASE)
    if q and fy:
        return f"Q{q.group(1)} FY{fy.group(1)}"
    if q:
        return f"Q{q.group(1)}"
    return stem.replace("_", " ")


def chunk_pages(file_name: str, pages: list[tuple[int, str]], quarter: str) -> list[dict]:
    """Split each page into chunks, carrying file name, page number and quarter.

    Each chunk is prefixed with a source label so the file and quarter are
    embedded with the content and retrieval can tell quarters apart.
    """
    splitter = _build_splitter()
    chunks: list[dict] = []
    for page_number, text in pages:
        for piece in splitter.split_text(text):
            cleaned = piece.strip()
            if cleaned:
                labelled = f"[Source: {file_name} | {quarter} | page {page_number}]\n{cleaned}"
                chunks.append(
                    {
                        "text": labelled,
                        "file": file_name,
                        "page": page_number,
                        "quarter": quarter,
                    }
                )
    return chunks


def _chunk_id(file_name: str, page: int, position: int, text: str) -> str:
    """Deterministic id so re-indexing a file updates instead of duplicating."""
    raw = f"{file_name}:{page}:{position}:{text[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def ingest_paths(paths: Iterable[str | Path]) -> dict:
    """Load, chunk, embed and store the given PDFs.

    Returns {"files", "chunks", "skipped"} counting only what was indexed now.
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
            skipped.append(f"{path.name} (no extractable text - scanned image?)")
            continue

        quarter = _derive_quarter(path.name)
        chunks = chunk_pages(path.name, pages, quarter)
        if not chunks:
            skipped.append(f"{path.name} (produced no chunks)")
            continue

        ids, documents, metadatas = [], [], []
        for position, chunk in enumerate(chunks):
            ids.append(_chunk_id(chunk["file"], chunk["page"], position, chunk["text"]))
            documents.append(chunk["text"])
            metadatas.append(
                {"file": chunk["file"], "page": chunk["page"], "quarter": chunk["quarter"]}
            )

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
    """Summary used by the /stats endpoint and the UI sidebar."""
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
