"""Resource management endpoints for version listing and status."""
from fastapi import HTTPException
from sqlalchemy.orm import Session
from src.db.models import Resource


def get_all_resources(session: Session) -> list[dict]:
    """Get all resources with their latest version info."""
    # Get unique resource_ids
    resource_ids = session.query(Resource.resource_id).distinct().all()
    result = []

    for (rid,) in resource_ids:
        latest = (
            session.query(Resource)
            .filter_by(resource_id=rid)
            .order_by(Resource.version.desc())
            .first()
        )
        if latest:
            result.append({
                "resource_id": rid,
                "latest_version": latest.version,
                "status": latest.status,
                "chunk_count": 0,  # Would need a chunks table to populate
            })

    return result


def get_resource_versions(resource_id: str, session: Session) -> list[dict]:
    """Get all versions of a resource."""
    versions = (
        session.query(Resource)
        .filter_by(resource_id=resource_id)
        .order_by(Resource.version.asc())
        .all()
    )

    if not versions:
        raise HTTPException(status_code=404, detail=f"Resource {resource_id} not found")

    return [
        {
            "version": v.version,
            "status": v.status,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "chunk_count": 0,
        }
        for v in versions
    ]
