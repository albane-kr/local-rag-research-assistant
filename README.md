# Local NotebookLM‑Style Research Assistant

A local, private, extensible research assistant that ingests documents, extracts structured representations, and answers questions using retrieval-augmented generation (RAG) with local LLMs.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

### Web UI (recommended)
```powershell
uvicorn src.api.app:app --reload
# Open http://localhost:8000/ui in your browser
```

### CLI
```powershell
# Ingest a document
python -m src.cli.main ingest path/to/document.pdf

# List all resources
python -m src.cli.main resources

# List versions of a resource
python -m src.cli.main versions <resource-id>

# Chat with RAG
python -m src.cli.main chat "What is machine learning?"
```

### API Endpoints
- `POST /ingest` — Upload a document (triggers ETL → chunking → indexing)
- `POST /query` — Retrieve chunks (no LLM)
- `POST /chat` — Full RAG + LLM pipeline
- `GET /resources` — List all resources
- `GET /resources/<id>/versions` — List versions of a resource

## Testing

```powershell
python -m pytest -v
```

## Architecture

- **Ingest**: Local files, versioned storage (UUID resource_id + version number)
- **ETL**: Markdown conversion via markitdown, stored under `./data/markdown/`
- **Chunking**: Heading-based splitter with overlap, stored under `./data/raw/`
- **Embeddings**: all-MiniLM-L6-v2 (384-dim, CPU-friendly)
- **Vector DB**: Chroma with persistent storage (`./chroma_db/`), metadata filtering for version isolation
- **RAG**: Retrieve latest-version chunks only, build prompts with fallback
- **LLM**: Ollama local server with query-type routing (summary→gemma, compare→nemotron, analyze→granite, default→llama)
- **Database**: SQLite (`./data.db`) for resource metadata and versioning
