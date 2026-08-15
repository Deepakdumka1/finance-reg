"""Streamlit interface for the Finance RAG system.

Mandatory features: upload PDFs; an Index button that reports files + chunks; a
question box; the answer; the sources (file + page) under each answer; honest
refusal (handled in rag.py); persistence via on-disk ChromaDB; and a visible
question/answer history that survives across questions.

Bonus: a sidebar toggle runs the UI in "API mode", calling the FastAPI backend
over HTTP instead of doing the work in-process.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import requests
import streamlit as st

import config
from ingest import collection_stats, ingest_paths
from rag import answer_question

st.set_page_config(page_title="Finance RAG", page_icon=":bar_chart:", layout="wide")


def do_ingest(saved_paths: list[str], use_api: bool) -> dict:
    if use_api:
        files = [("files", (p.split("/")[-1], open(p, "rb"), "application/pdf")) for p in saved_paths]
        resp = requests.post(f"{config.API_BASE_URL}/ingest", files=files, timeout=600)
        resp.raise_for_status()
        return resp.json()
    return ingest_paths(saved_paths)


def do_ask(question: str, top_k: int, use_api: bool) -> dict:
    if use_api:
        resp = requests.post(
            f"{config.API_BASE_URL}/ask",
            json={"question": question, "top_k": top_k},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    return answer_question(question, top_k=top_k)


def do_stats(use_api: bool) -> dict:
    if use_api:
        resp = requests.get(f"{config.API_BASE_URL}/stats", timeout=30)
        resp.raise_for_status()
        return resp.json()
    return collection_stats()


def save_uploads(uploaded_files) -> list[str]:
    """Persist uploaded files into data/ so they can be re-indexed later."""
    paths = []
    for uf in uploaded_files:
        dest = config.DATA_DIR / uf.name
        with open(dest, "wb") as fh:
            fh.write(uf.getbuffer())
        paths.append(str(dest))
    return paths


# Sidebar -------------------------------------------------------------------
st.sidebar.title("Finance RAG")
st.sidebar.caption("Ask questions across quarterly-report PDFs.")

use_api = st.sidebar.toggle(
    "Use FastAPI backend (bonus)",
    value=False,
    help="When on, the UI calls the FastAPI service instead of running the "
    "pipeline in-process. Start it with: uvicorn api.main:app --reload",
)

top_k = st.sidebar.slider("Chunks to retrieve (top_k)", 1, 10, config.DEFAULT_TOP_K)

st.sidebar.divider()
st.sidebar.subheader("Index status")
try:
    stats = do_stats(use_api)
    st.sidebar.metric("Chunks stored", stats.get("total_chunks", "?"))
    st.sidebar.write(f"**Provider:** {stats.get('provider', '?')}")
    st.sidebar.write(f"**Collection:** {stats.get('collection', '?')}")
    st.sidebar.write(f"**Embeddings:** {stats.get('embedding_model', '?')}")
    st.sidebar.write(f"**LLM:** {stats.get('llm_model', '?')}")
except Exception as exc:  # noqa: BLE001
    st.sidebar.warning(f"Could not read stats: {exc}")

if config.PROVIDER == "openai" and not config.OPENAI_API_KEY and not use_api:
    st.sidebar.error("OPENAI_API_KEY not found. Add it to .env before indexing or asking.")


# Main ----------------------------------------------------------------------
st.title("Quarterly Report Q&A")
st.write(
    "Upload quarterly-result PDFs, index them, then ask questions in plain "
    "English. Every answer shows the source file and page so you can verify it."
)

# 1) Upload + Index
st.header("1. Upload & index")
uploaded = st.file_uploader(
    "Upload one or more quarterly-report PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Index uploaded files", type="primary", disabled=not uploaded):
        with st.spinner("Loading, chunking, embedding and storing..."):
            try:
                paths = save_uploads(uploaded)
                result = do_ingest(paths, use_api)
                st.success(
                    f"{result['files']} files processed, {result['chunks']} chunks stored."
                )
                for skipped in result.get("skipped", []):
                    st.warning(f"Skipped {skipped}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Indexing failed: {exc}")

with col_b:
    if st.button("Re-index everything in data/"):
        with st.spinner("Re-indexing the data/ folder..."):
            try:
                pdfs = [str(p) for p in sorted(config.DATA_DIR.glob("*.pdf"))]
                if not pdfs:
                    st.info("No PDFs found in the data/ folder yet.")
                else:
                    result = do_ingest(pdfs, use_api)
                    st.success(
                        f"{result['files']} files processed, {result['chunks']} chunks stored."
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Indexing failed: {exc}")

# 2) Ask
st.header("2. Ask a question")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input(
    "Your question",
    placeholder="e.g. What was total revenue in the most recent quarter?",
)

if st.button("Get answer", type="primary", disabled=not question):
    try:
        indexed = do_stats(use_api).get("total_chunks", 0)
    except Exception:  # noqa: BLE001
        indexed = 0

    if not indexed:
        st.warning("Nothing is indexed yet. Upload PDFs (or use data/) and click Index first.")
    else:
        with st.spinner("Retrieving relevant chunks and asking the model..."):
            try:
                result = do_ask(question, top_k, use_api)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Query failed: {exc}")
                result = None
        if result:
            st.session_state.history.insert(0, {"question": question, "result": result})

if st.session_state.history and st.button("Clear history"):
    st.session_state.history = []
    st.rerun()

for item in st.session_state.history:
    result = item["result"]
    with st.container(border=True):
        st.markdown(f"**Q:** {item['question']}")
        st.markdown("**Answer**")
        st.write(result["answer"])

        sources = result.get("sources", [])
        st.markdown("**Sources**")
        if sources:
            for src in sources:
                st.markdown(f"- **{src['file']}** - page {src['page']}")
        else:
            st.caption("No sources (the answer was not found in the documents).")

        chunks = result.get("chunks", [])
        if chunks:
            with st.expander("Show retrieved chunks (debug)"):
                for j, ch in enumerate(chunks, start=1):
                    dist = ch.get("distance")
                    dist_str = f" (distance {dist:.4f})" if isinstance(dist, (int, float)) else ""
                    st.markdown(f"**Passage {j} - {ch['file']} p.{ch['page']}{dist_str}**")
                    st.text(ch["text"])

# 3) Optional: share price (yfinance)
with st.expander("Optional: look up a live share price (yfinance)"):
    st.caption("Nice-to-have only. Does not affect the PDF-based answers above.")
    ticker = st.text_input("Ticker symbol", placeholder="e.g. INFY, AAPL, MSFT")
    if st.button("Fetch price", disabled=not ticker):
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).fast_info
            price = getattr(info, "last_price", None) or info.get("lastPrice")
            currency = getattr(info, "currency", None) or info.get("currency", "")
            if price:
                st.metric(f"{ticker.upper()} last price", f"{price:.2f} {currency}")
            else:
                st.info("No price returned for that ticker.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not fetch price: {exc}")
