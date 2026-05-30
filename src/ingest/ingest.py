import uuid
import shutil
from pathlib import Path
from src.db.models import Resource


def get_next_version(resource_id: str, session) -> int:
    """Get the next version number for a resource_id."""
    latest = session.query(Resource).filter_by(resource_id=resource_id).order_by(Resource.version.desc()).first()
    return (latest.version + 1) if latest else 1


def ingest_file(file_path: str, session) -> dict:
    """Ingest a file and store raw copy."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(file_path)
    
    resource_id = str(uuid.uuid4())
    version = get_next_version(resource_id, session)
    
    storage_dir = Path("./data/raw")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = storage_dir / f"{resource_id}_v{version}{p.suffix}"
    shutil.copy(p, raw_path)
    
    resource = Resource(
        resource_id=resource_id,
        version=version,
        raw_path=str(raw_path),
        status="ingested"
    )
    session.add(resource)
    session.commit()
    
    return {
        "resource_id": resource_id,
        "version": version,
        "raw_path": str(raw_path),
        "status": "ingested"
    }
