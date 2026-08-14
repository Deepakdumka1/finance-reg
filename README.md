# Finance RAG — Retrieval-Augmented Q&A over PDF documents

A small, self-contained RAG (Retrieval-Augmented Generation) app. It reads PDFs,
splits them into chunks, embeds the chunks, stores the vectors in a persistent
ChromaDB, and answers natural-language questions using an LLM restricted to the
retrieved text. Every answer shows the **source file and page**, and the app
**refuses to answer** when the information is not in the documents.

Built for the ET AI Academy / HCLTech RAG project.

## Providers: OpenAI or Ollama (local)

The assignment's default stack is OpenAI (`text-embedding-3-small` for embeddings,
`gpt-4o` for answers). This project also supports **Ollama**, which runs entirely
on your machine with no API key and no cost (the academy approved this
alternative). Pick the backend with `PROVIDER` in `.env`:

| PROVIDER               | Embeddings            | Answering model |
|------------------------|-----------------------|-----------------|
| `ollama` (default here)| `nomic-embed-text`    | `llama3.2`      |
| `openai`               | `text-embedding-3-small` | `gpt-4o`     |

## Structure

```
finance-rag/
|-- app.py            # Streamlit UI (upload, index, ask, sources)
|-- ingest.py         # load PDFs -> chunk -> embed -> store in ChromaDB
|-- rag.py            # retrieve -> prompt -> LLM answer with sources
|-- vector_store.py   # shared ChromaDB collection + embedding function
|-- config.py         # all settings (models, chunk size, provider)
|-- api/main.py       # optional FastAPI backend (/ingest, /ask, /stats)
|-- data/             # the PDFs to index
|-- chroma_db/        # persisted vector store (git-ignored; rebuilt by ingest)
|-- requirements.txt
|-- .env.example
`-- .gitignore
```

## Setup

Python 3.10+ recommended (works on 3.9). For the local provider, install Ollama
from https://ollama.com and pull the models.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Local (Ollama) provider:
ollama pull nomic-embed-text
ollama pull llama3.2

cp .env.example .env               # then edit if needed
```

Example `.env` (local default):

```
PROVIDER=ollama
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

To use OpenAI instead: set `PROVIDER=openai` and `OPENAI_API_KEY=sk-...`.

## Run

```
python ingest.py --reset     # index everything in data/
streamlit run app.py         # open http://localhost:8501
```

In the UI: upload PDFs (or use the ones already in `data/`), click **Index**,
ask a question, and read the answer with its sources.

Optional FastAPI backend (bonus):

```
uvicorn api.main:app --reload   # http://localhost:8000/docs
```

- `POST /ingest` -> `{"files": N, "chunks": M}`
- `POST /ask` `{"question","top_k"}` -> `{"answer","sources":[{"file","page"}]}`
- `GET /stats` -> provider, collection, chunk count, models

## How it works

1. **Load** — `pypdf` extracts text page by page (page numbers preserved).
2. **Chunk** — recursive character splitting, 1000 chars with 150 overlap.
3. **Embed** — each chunk is embedded (Ollama `nomic-embed-text` or OpenAI).
4. **Store** — vectors + `{file, page}` metadata go into a persistent ChromaDB;
   it survives restarts.
5. **Ask** — the question is embedded, the top-k closest chunks are retrieved,
   and the LLM answers using only those chunks. Sources (file + page) are shown.

### Chunk-size choice
1000 characters with 150 overlap sits mid-range of the allowed 800-1200 / 100-200.
It is large enough to keep most paragraphs and small tables inside one chunk
(better answers) while staying cheap to embed.

### Honest refusal
The system prompt tells the model to answer only from the retrieved context and
otherwise reply "The information is not available in the uploaded documents.",
so it does not invent answers.

## Data indexed
This repo currently indexes the provided Meridian documents in `data/`:
- `Meridian_Procurement_Policy_Handbook_v4.2.pdf`
- `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf`

## Example runs (Ollama: llama3.2 + nomic-embed-text)

**Q: What were the main supply chain risks highlighted in the Q1 review?**
A: Single-source dependency for microcontrollers (Shenzhen Rui Electronics);
port congestion at Nhava Sheva adding ~2.9 days per ocean lane; copper up ~14%
YoY pressuring wire-harness margins; a new BIS certification requirement from
Oct 2025; and customer concentration (top 3 customers = 71% of revenue).
Sources: Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf (pages 1, 3)

**Q (trap): What is the annual salary of Meridian's Head of Procurement?**
A: The information is not available in the uploaded documents.

## Limitations / honest notes
- Local `llama3.2` (3B) is smaller than GPT-4o; answers are solid but sometimes
  less precise. A larger local model (e.g., `llama3.1:8b`) improves quality if
  your machine has enough RAM.
- The first query after startup is slow (~20s) while the model loads into memory.
- PDF tables lose alignment when extracted to text; larger chunks mitigate this.
- The vector store is git-ignored, so run `python ingest.py --reset` after cloning.
