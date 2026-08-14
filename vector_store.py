"""Shared access to the persisted ChromaDB collection.

Both ingest.py (writing) and rag.py (reading) import get_collection() from
here so they always talk to the same on-disk store, configured with the same
OpenAI embedding function. Using one embedding function for indexing *and*
querying is what makes similarity search meaningful.
"""
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

import config


@lru_cache(maxsize=1)
def get_client() -> "chromadb.ClientAPI":
    """Return a process-wide PersistentClient pointed at CHROMA_DIR.

    PersistentClient writes the index to disk, so documents indexed in one run
    are still searchable after the app is stopped and started again.
    """
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


@lru_cache(maxsize=1)
def get_embedding_function():
    """Embedding function for every chunk and every query, chosen by PROVIDER.

    Ollama runs locally (no key); OpenAI is the default. The same function is
    used for indexing and querying so the vectors are always comparable.
    """
    if config.PROVIDER == "ollama":
        return embedding_functions.OllamaEmbeddingFunction(
            url=config.OLLAMA_BASE_URL,
            model_name=config.OLLAMA_EMBED_MODEL,
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.require_api_key(),
        model_name=config.EMBEDDING_MODEL,
    )


def get_collection():
    """Get (or create) the financial_reports collection.

    cosine distance is a good default for OpenAI embeddings, which are
    normalised, so cosine and dot-product rank identically.
    """
    return get_client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    """Delete every vector in the collection (used by the 'Clear index' button)."""
    client = get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet; that is fine.
        pass
