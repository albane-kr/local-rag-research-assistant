import os
import tempfile

from src.db.models import Resource


def test_ingest_flow(client):
    """INGEST-01: New resource creates version 1 stored in DB."""
    test_client, TestSessionLocal = client

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("# Test Document\n\nThis is a test.")
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            response = test_client.post("/ingest", files={"file": ("test.txt", f, "text/plain")})

        assert response.status_code == 200, response.text
        data = response.json()
        assert "resource_id" in data
        assert data["version"] == 1
        assert data["status"] == "ingested"
        assert data["etl_status"] in ("success", "failed")

        session = TestSessionLocal()
        resource = session.query(Resource).filter_by(resource_id=data["resource_id"]).first()
        assert resource is not None
        assert resource.version == 1
        session.close()
    finally:
        os.unlink(tmp_path)
