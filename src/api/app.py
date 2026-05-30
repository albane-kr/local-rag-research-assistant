from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import tempfile
import os
from src.db.models import init_db, Resource
from src.ingest.ingest import ingest_file
from src.etl.markitdown_adapter import run_markitdown
from pathlib import Path
from sqlalchemy.orm import Session

app = FastAPI()
SessionLocal = init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Welcome to Local RAG Research Assistant"}


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...), session: Session = Depends(get_db)):
    """Ingest a document file and trigger ETL."""
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

        return JSONResponse(result)
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
