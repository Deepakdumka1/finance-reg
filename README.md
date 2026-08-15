# Finance RAG — Q&A over Quarterly Financial Reports

A Retrieval-Augmented Generation (RAG) system for the ET AI Academy / HCLTech
Assignment 1. It reads a company's quarterly-result PDFs, splits them into
chunks, embeds them, stores the vectors in a **persistent ChromaDB**, and
answers plain-English questions with an LLM that is **restricted to the
retrieved text**. Every answer shows the **source file and page**, and the app
**refuses to answer** when the information is not in the documents.

## Company and documents

**Infosys Limited** — 4 consecutive quarters, FY2025-26 (IFRS, INR press
releases, which include management commentary).

| # | File | Quarter | Source |
|---|------|---------|--------|
| 1 | `Infosys_Q1_FY26.pdf` | Q1 FY26 (Jun 2025) | [link](https://www.infosys.com/investors/reports-filings/quarterly-results/2025-2026/q1/documents/ifrs-inr-press-release.pdf) |
| 2 | `Infosys_Q2_FY26.pdf` | Q2 FY26 (Sep 2025) | [link](https://www.infosys.com/investors/reports-filings/quarterly-results/2025-2026/q2/documents/ifrs-inr-press-release.pdf) |
| 3 | `Infosys_Q3_FY26.pdf` | Q3 FY26 (Dec 2025) | [link](https://www.infosys.com/investors/reports-filings/quarterly-results/2025-2026/q3/documents/ifrs-inr-press-release.pdf) |
| 4 | `Infosys_Q4_FY26.pdf` | Q4 FY26 (Mar 2026) | [link](https://www.infosys.com/investors/reports-filings/quarterly-results/2025-2026/q4/documents/ifrs-inr-press-release.pdf) |

All four pass the text-selection test (real text, not scanned images).

## Model provider (OpenAI or Ollama)

The assignment's default stack is OpenAI (`text-embedding-3-small` + `gpt-4o`).
A working OpenAI key was not available, and the academy approved alternatives,
so this project runs on **Ollama** (fully local, free, no API key). The
provider is a one-line switch in `.env`:

| PROVIDER | Embeddings | Answering model |
|----------|-----------|-----------------|
| `ollama` (used here) | `nomic-embed-text` | `llama3.2` (3B) |
| `openai` | `text-embedding-3-small` | `gpt-4o` |

To switch to OpenAI: set `PROVIDER=openai` and `OPENAI_API_KEY=sk-...` in `.env`.

## Structure

```
finance-rag/
|-- app.py            # Streamlit UI (upload, index, ask, sources, history)
|-- ingest.py         # load PDFs -> chunk -> embed -> store in ChromaDB
|-- rag.py            # retrieve -> prompt -> answer with sources
|-- vector_store.py   # shared ChromaDB collection + embedding function
|-- config.py         # all settings (provider, models, chunk size)
|-- api/main.py       # optional FastAPI backend (/ingest, /ask, /stats)
|-- data/             # the 4 Infosys quarterly PDFs
|-- chroma_db/        # persisted vector store (git-ignored; rebuilt by ingest)
|-- requirements.txt
|-- .env.example
`-- .gitignore
```

## Setup

Python 3.10+ recommended. Install Ollama from https://ollama.com, then:

```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2
cp .env.example .env                 # PROVIDER=ollama by default
```

## Run

```
python ingest.py --reset             # index the PDFs in data/
streamlit run app.py                 # http://localhost:8501
```

Optional FastAPI backend (bonus):

```
uvicorn api.main:app --reload        # http://localhost:8000/docs
```
- `POST /ingest` -> `{"files": N, "chunks": M}`
- `POST /ask` `{"question","top_k"}` -> `{"answer","sources":[{"file","page"}]}`
- `GET /stats` -> provider, collection, chunk count, models

## How it works

1. **Load** — `pypdf` extracts text page by page (page numbers kept).
2. **Chunk** — recursive character splitting, 1000 chars / 150 overlap. Each
   chunk is prefixed with a `[Source: file | quarter | page]` label so the
   quarter is embedded and retrieval can tell the near-identical quarterly
   press releases apart.
3. **Embed** — `nomic-embed-text` via Ollama (same model for chunks + queries).
4. **Store** — vectors + `{file, page, quarter}` metadata in a persistent
   ChromaDB. Survives restarts.
5. **Ask** — the question is embedded, the top-k closest chunks retrieved, and
   the model answers using only those chunks. Sources (file + page) are shown.

**Chunk size:** 1000 / 150 sits mid-range of the allowed 800-1200 / 100-200 —
big enough to keep most of a table or paragraph in one chunk (financial tables
lose alignment as plain text), while staying cheap to embed.

**Indexing:** run once on demand; a stable per-chunk id (file + page + position)
means re-running overwrites rather than duplicating.

### System prompt (grounding + honest refusal), temperature 0.0

```
You are a financial research assistant for an investment advisory desk.
Answer the analyst's question using ONLY the context passages provided below,
which are extracts from company quarterly-report PDFs. Rules:
1. If the answer is not in the context, reply exactly: "The information is not
   available in the uploaded documents." Do not guess or use outside knowledge.
2. Quote figures exactly as they appear and mention the quarter/period.
3. Be concise and factual; cite page numbers in-line like (p. 7).
4. If the context is unclear or contradictory, say what you can support.
```

## The 10 test questions (answers produced by the app, top_k=6)

1. **Total revenue, latest quarter (Q4 FY26)** — ₹46,402 crore. **Correct** (verified against the PDF).
2. **Net profit compared across quarters; highest?** — Listed Q1-Q4; said Q4 highest. **Partly wrong** — it used Q4's full-year net profit (~₹36,254 cr) instead of the quarterly figure; the quarterly comparison is off.
3. **Revenue YoY vs same quarter last year** — Reported Q3 FY26 ₹45,479 cr, +8.9% YoY. **Partly off** — answered for Q3 rather than the latest quarter (Q4).
4. **Management commentary on demand** — Quoted the CEO on strong deal wins and AI-driven demand (Q2). **Reasonable.**
5. **Fastest-growing segment/geography** — "The information is not available in the uploaded documents." **Defensible** — the INR press release does not break this out cleanly.
6. **Operating margin per quarter; trend** — Q1 20.8%, Q3 20.0%, Q4 20.9%, Q2 not found. **Partly wrong** — Q4 is actually 20.3% (reported), and Q2 (21.0%) was missed.
7. **Dividend declared, amount, record date** — "Not available." **Wrong (miss)** — dividends exist (Q2 interim ₹23/share; Q4 final ₹25/share); retrieval did not surface the line.
8. **Risks / headwinds / challenges** — Full list from the safe-harbour section (talent, wages, Generative AI disruption, regulation/immigration, McCamish cyber incident, US H-1B, litigation, etc.). **Correct and thorough.**
9. **Three-line summary of the latest quarter** — Produced a 3-line summary, but drew on Q3 figures for the "latest" quarter. **Partly off.**
10. **Trap: CEO's personal shareholding in 2015** — "The information is not available in the uploaded documents." **Correct refusal.**

## Manual verification (by hand, against the PDFs)

| Figure | App said | PDF says | Match? |
|--------|----------|----------|--------|
| Q4 FY26 revenue | ₹46,402 cr | ₹46,402 cr | Yes |
| Q1 FY26 operating margin | 20.8% | 20.8% | Yes |
| Q4 FY26 operating margin | 20.9% | 20.3% (reported) | No |

## Honest notes / what did not work well

- **Provider:** runs on Ollama `llama3.2` (3B), not GPT-4o. Answers are solid on
  single-fact and list questions (revenue, risks, margins) but weaker on
  cross-quarter reasoning and table figures — hence the errors in Q2, Q6, Q9.
- **Tables:** Q4's press release contains both quarterly and full-year columns;
  the model sometimes picked the full-year number (Q2 net-profit answer). Larger
  chunks help but do not fully solve plain-text table ambiguity.
- **"Latest quarter" drift:** with four near-identical documents the model
  occasionally answered for Q3 instead of Q4 (Q3, Q9). The per-chunk quarter
  label reduced but did not eliminate this.
- **Dividend miss (Q7):** the dividend appears inside prose-heavy chunks and did
  not rank in the top-6 for that query. Raising `top_k` or a dedicated dividend
  query would likely recover it.
- **Likely improvement:** switching to GPT-4o (set `PROVIDER=openai`) or a larger
  local model (`llama3.1:8b`) should improve the numeric/table answers.

## Notes
- `.env`, `.venv/`, and `chroma_db/` are git-ignored. Run `python ingest.py --reset` after cloning to rebuild the index.
