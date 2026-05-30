from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
from src.db.models import init_db, Resource
from src.ingest.ingest import ingest_file
from src.etl.markitdown_adapter import run_markitdown
from src.chunking.pipeline import index_resource, _get_chroma
from src.vectorstore.chroma_client import ChromaClient
from src.rag.retriever import retrieve_context, build_context_prompt
from pathlib import Path
from sqlalchemy.orm import Session


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    fallback: bool = True

app = FastAPI()
SessionLocal = init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_chroma() -> ChromaClient:
    return _get_chroma()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Welcome to Local RAG Research Assistant"}


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    chroma: ChromaClient = Depends(get_chroma),
):
    """Ingest a document file and trigger ETL and chunking."""
    suffix = os.path.splitext(file.filename)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        # File handle is now closed — safe to copy on Windows

        result = ingest_file(tmp_path, session)

        md_dir = Path("./data/markdown")
        md_dir.mkdir(parents=True, exist_ok=True)
        md_path = md_dir / f"{result['resource_id']}_v{result['version']}.md"

        try:
            run_markitdown(result["raw_path"], str(md_path))
            result["markdown_path"] = str(md_path)
            result["etl_status"] = "success"
        except Exception as e:
            result["etl_status"] = "failed"
            result["etl_error"] = str(e)

        # Persist ETL result back to the DB record
        resource = session.query(Resource).filter_by(resource_id=result["resource_id"], version=result["version"]).first()
        if resource:
            resource.markdown_path = result.get("markdown_path")
            resource.status = "etl_success" if result["etl_status"] == "success" else "etl_failed"
            session.commit()

        # Chunk, embed, and index if ETL succeeded
        if result["etl_status"] == "success":
            try:
                chunk_count = index_resource(
                    result["resource_id"], result["version"], str(md_path), session, chroma=chroma
                )
                result["chunk_count"] = chunk_count
                result["status"] = "indexed"
            except Exception as e:
                result["index_error"] = str(e)

        return JSONResponse(result)
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/query")
def query_endpoint(req: QueryRequest, chroma: ChromaClient = Depends(get_chroma)):
    """RAG-01/02/03: Retrieve context from latest-version chunks and build prompt."""
    try:
        chunks = retrieve_context(req.query, chroma, top_k=req.top_k)
        prompt = build_context_prompt(req.query, chunks, fallback=req.fallback)

        return JSONResponse({
            "query": req.query,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "resource_id": c.resource_id,
                    "version": c.version,
                    "heading": c.heading,
                    "distance": float(c.distance),
                }
                for c in chunks
            ],
            "prompt": prompt,
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
