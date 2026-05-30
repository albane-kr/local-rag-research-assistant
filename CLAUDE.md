# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands assume the `.venv` is active. On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Alternatively, invoke via the venv directly: `.\.venv\Scripts\python.exe -m <command>`.

```powershell
# Install dependencies
pip install -r requirements.txt

# Run dev server
uvicorn src.api.app:app --reload

# Run all tests
python -m pytest -q

# Run a single test
python -m pytest tests/test_ingest.py::test_ingest_flow -v
```

Never run `pytest` bare — it will resolve to the system Python where dependencies are not installed. Always use `python -m pytest` or `.\.venv\Scripts\pytest`.

## Architecture

### Core pipeline (in execution order)

```
Upload → /ingest endpoint
  → ingest_file()       stores raw file under ./data/raw/, writes Resource row to SQLite
  → run_markitdown()    converts raw file to Markdown, saves under ./data/markdown/
  → [step 5, pending]  chunk Markdown → embed → upsert into Chroma with version metadata
  → [step 6, pending]  RAG: retrieve latest-version chunks → build prompt → Ollama LLM
```

### Versioning contract

Every stored artifact belongs to a `(resource_id, version)` pair. A `UniqueConstraint` on this pair is enforced in the DB. `ingest_file()` calls `get_next_version()` to auto-increment; new uploads of the same logical resource must pass the same `resource_id`. The RAG layer must only retrieve chunks where `is_latest=True` — this is the mechanism for CHUNK-03 / VER-03 / GOV-02.

### Database

Single `Resource` table in SQLite (`./data.db`). Schema is in `src/db/models.py`. `init_db()` is called once at app startup (`app.py:11`) and returns a `SessionLocal` factory. All endpoints receive a session via `Depends(get_db)` — do not instantiate sessions manually inside handlers.

### Test isolation

`tests/conftest.py` overrides `get_db` with an in-memory SQLite using `StaticPool` (required so all sessions in a request share the same connection). Tests receive `client` and `test_session` fixtures — never import `SessionLocal` from `app` in tests.

### Stubs still to implement

- `src/vectorstore/chroma_client.py` — `add()` raises `NotImplementedError`; `query()` returns `[]`
- `src/llm/ollama_client.py` — `generate()` returns a hardcoded stub string
- `src/chunking/splitter.py` — heading splitter exists but is not wired to embedding or Chroma

### Key design decisions

- **markitdown** (Python API, not CLI) is the canonical ETL tool. Use `MarkItDown().convert(path)` — see `src/etl/markitdown_adapter.py`.
- **Chroma** is the chosen vector store (native metadata filtering required for version isolation). FAISS is in `requirements.txt` but is not the target.
- **sentence-transformers** is the chosen embedding backend (local CPU, no Ollama dependency for embeddings).
- ETL status and `markdown_path` must be written back to the `Resource` row after conversion — the DB is the source of truth for pipeline state.
- All data paths (`./data/raw`, `./data/markdown`, `./chroma_db`) are currently relative to CWD — a known issue to address before production.

### Control tests (from prd.md)

Each PRD control test maps to a test file. Current coverage:

| ID | File | Status |
|----|------|--------|
| INGEST-01 | `tests/test_ingest.py` | passing |
| ETL-01 | `tests/test_ingest.py` (etl_status check) | passing |
| INGEST-02, INGEST-03, VER-*, CHUNK-*, KG-*, RAG-*, LLM-*, UI-* | not yet written | pending |
