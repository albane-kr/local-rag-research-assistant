import os
import tempfile
from pathlib import Path

from src.db.models import Resource
from src.chunking.embedder import embed_chunks


def test_ingest_flow(client):
    """INGEST-01 + ETL-01 + CHUNK-01: Full pipeline — ingest → ETL → chunking."""
    test_client, TestSessionLocal, chroma = client

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("# Introduction\n\nThis is test content for the introduction section.\n\n"
                "## Background\n\nAdditional background details go here.")
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            response = test_client.post("/ingest", files={"file": ("test.txt", f, "text/plain")})

        assert response.status_code == 200, response.text
        data = response.json()

        # INGEST-01: resource created with version 1
        assert "resource_id" in data
        assert data["version"] == 1
        assert data["status"] == "indexed"  # now "indexed" after full pipeline

        # ETL-01: markdown was generated
        assert data["etl_status"] == "success"
        assert "markdown_path" in data
        md_path = Path(data["markdown_path"])
        assert md_path.exists()
        markdown_content = md_path.read_text()
        assert "# Introduction" in markdown_content or "Introduction" in markdown_content

        # CHUNK-01: chunks were created and indexed
        assert "chunk_count" in data
        assert data["chunk_count"] > 0

        # Verify Resource record in DB
        session = TestSessionLocal()
        resource = session.query(Resource).filter_by(resource_id=data["resource_id"]).first()
        assert resource is not None
        assert resource.version == 1
        assert resource.status == "indexed"
        assert resource.markdown_path == data["markdown_path"]
        session.close()

        # Verify chunks are in Chroma and only latest version is returned
        query_vec = embed_chunks(["Introduction"])[0]
        results = chroma.query(query_vec, top_k=10)
        assert len(results) > 0, "expected chunks to be indexed in Chroma"
        returned_versions = {r["metadata"]["version"] for r in results}
        assert returned_versions == {1}, f"expected only v1, got {returned_versions}"
        returned_resource_ids = {r["metadata"]["resource_id"] for r in results}
        assert data["resource_id"] in returned_resource_ids

    finally:
        os.unlink(tmp_path)
        if Path(data.get("markdown_path", "")).exists():
            Path(data["markdown_path"]).unlink()
