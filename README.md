# RAG-based Education AI · Interactive Learning Assistant

Course materials (PDF, slides, notes, lecture videos) are indexed into a local vector database. The
assistant answers questions grounded in them, generates practice questions, evaluates student
answers and measures hallucination.

- `backend/`: FastAPI service (ingestion → chunking → BGE-m3 embeddings → ChromaDB → retrieval → LLM)
- `rag_ui/`: Streamlit UI; it talks to the backend only over HTTP (`docs/api_contract.md`)

## Setup (once)

```bash
conda create -n rag python=3.12 -y
conda activate rag
pip install torch --index-url https://download.pytorch.org/whl/cu126   # RTX 50xx: cu128; no GPU: .../whl/cpu
pip install -r requirements.txt
pip install -r rag_ui/requirements.txt
cp .env.example .env          # add GROQ_API_KEY or OPENAI_API_KEY (or use Ollama)
```

## Run

Two terminals, both in the project folder:

```bash
uvicorn backend.main:app                                    # API on http://localhost:8000 (docs: /docs)
streamlit run rag_ui/app.py --server.fileWatcherType none  # UI
```

Start the backend first; the sidebar shows whether the UI can reach it. Set `RAG_API_URL` if the
backend runs elsewhere.

## Tests

```bash
python -m pytest -q
```
