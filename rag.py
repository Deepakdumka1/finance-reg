"""Retrieval-Augmented answering: retrieve chunks -> prompt -> GPT-4o.

The public entry point is answer_question(). It returns the answer text plus
the list of sources (file name + page number) so the UI can show where every
answer came from and an analyst can verify it against the original PDF.
"""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

import config
from vector_store import get_collection

# The refusal instruction lives in the system prompt. This is how the app
# stays honest: the model is told to answer ONLY from the supplied context and
# to say so plainly when the answer is not there.
SYSTEM_PROMPT = (
    "You are a financial research assistant for an investment advisory desk. "
    "Answer the analyst's question using ONLY the context passages provided "
    "below, which are extracts from company quarterly-report PDFs. "
    "Follow these rules strictly:\n"
    "1. If the answer is not contained in the context, reply exactly: "
    "\"The information is not available in the uploaded documents.\" Do not "
    "guess, and do not use any outside knowledge.\n"
    "2. When you give figures, quote them exactly as they appear and mention "
    "the quarter/period they belong to.\n"
    "3. Be concise and factual. Cite the page number(s) you used in-line, like "
    "(p. 7), when helpful.\n"
    "4. If the context is contradictory or unclear, say what you can support "
    "and note the uncertainty."
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Ollama exposes an OpenAI-compatible API, so the same client works for
    # both providers -- we just point it at the local server with a dummy key.
    if config.PROVIDER == "ollama":
        return OpenAI(base_url=f"{config.OLLAMA_BASE_URL}/v1", api_key="ollama")
    return OpenAI(api_key=config.require_api_key())


def retrieve(question: str, top_k: int = config.DEFAULT_TOP_K) -> list[dict]:
    """Return the top_k most similar chunks for a question.

    Each item: {"text", "file", "page", "distance"}. Sorted closest-first.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    top_k = max(1, min(top_k, collection.count()))
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: list[dict] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "file": meta.get("file", "unknown"),
                "page": meta.get("page", "?"),
                "distance": distance,
            }
        )
    return chunks


def _format_context(chunks: list[dict]) -> str:
    """Number each passage and label it with its source for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Passage {i} | {chunk['file']} | page {chunk['page']}]"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def _dedupe_sources(chunks: list[dict]) -> list[dict]:
    """Collapse sources to unique (file, page) pairs, preserving order."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["file"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": chunk["file"], "page": chunk["page"]})
    return sources


def answer_question(question: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
    """Answer a question from the indexed documents.

    Returns {"answer", "sources": [{"file", "page"}], "chunks": [...]}.
    If nothing has been indexed, returns the standard refusal so the UI never
    fabricates an answer.
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": [], "chunks": []}

    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return {
            "answer": "The information is not available in the uploaded documents.",
            "sources": [],
            "chunks": [],
        }

    context = _format_context(chunks)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    response = _client().chat.completions.create(
        model=config.active_llm_model(),
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()

    return {
        "answer": answer,
        "sources": _dedupe_sources(chunks),
        "chunks": chunks,
    }


if __name__ == "__main__":
    # Quick manual test: python rag.py "What was total revenue last quarter?"
    import sys

    q = " ".join(sys.argv[1:]) or "What was total revenue in the most recent quarter?"
    result = answer_question(q)
    print("Q:", q)
    print("A:", result["answer"])
    print("Sources:", result["sources"])
