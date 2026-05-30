# Local NotebookLM‑Style Research Assistant — MVP scaffold

This repository contains an initial scaffold for the Local NotebookLM‑style Research Assistant (Python + Chroma MVP).

Setup (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run development server:

```powershell
uvicorn src.api.app:app --reload
```

Run tests:

```powershell
pytest -q
```

Files created in scaffold:
- `src/api/app.py` (FastAPI health endpoint)
- `src/db/models.py` (SQLite metadata models)
- `src/ingest/` (ingest stubs)
- `src/etl/` (ETL stubs)
- `src/chunking/` (splitter)
- `src/vectorstore/` (Chroma adapter stub)
- `src/llm/` (Ollama client stub)
- `tests/test_health.py` (basic health test)
