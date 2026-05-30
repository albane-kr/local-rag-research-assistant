"""Tests for UI endpoints (UI-01, UI-02 from PRD)."""
import os
import tempfile
from pathlib import Path


def test_resources_endpoint(client):
    """UI-01: GET /resources returns list with current version info."""
    test_client, TestSessionLocal, chroma = client

    # Initially empty
    response = test_client.get("/resources")
    assert response.status_code == 200
    data = response.json()
    assert data["resources"] == []

    # Ingest a document
    doc_text = "# Test Document\n\nContent here."
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(doc_text)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            ingest_response = test_client.post(
                "/ingest",
                files={"file": ("test.txt", f, "text/plain")}
            )
        assert ingest_response.status_code == 200
        resource_id = ingest_response.json()["resource_id"]

        # Now list resources
        response = test_client.get("/resources")
        assert response.status_code == 200
        data = response.json()
        assert len(data["resources"]) == 1
        res = data["resources"][0]
        assert res["resource_id"] == resource_id
        assert res["latest_version"] == 1
        assert "status" in res
    finally:
        os.unlink(tmp_path)


def test_versions_endpoint(client):
    """Get all versions of a resource."""
    test_client, TestSessionLocal, chroma = client

    doc_text = "# Document\n\nVersion one."
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(doc_text)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            ingest_response = test_client.post(
                "/ingest",
                files={"file": ("doc.txt", f, "text/plain")}
            )
        assert ingest_response.status_code == 200
        resource_id = ingest_response.json()["resource_id"]

        # Get versions
        response = test_client.get(f"/resources/{resource_id}/versions")
        assert response.status_code == 200
        data = response.json()
        assert len(data["versions"]) == 1
        assert data["versions"][0]["version"] == 1
    finally:
        os.unlink(tmp_path)


def test_versions_endpoint_not_found(client):
    """GET /resources/{nonexistent}/versions returns 404."""
    test_client, TestSessionLocal, chroma = client

    response = test_client.get("/resources/nonexistent-id/versions")
    assert response.status_code == 404


def test_ui_chat_with_provenance(client):
    """UI-02: /chat endpoint returns answer with provenance metadata."""
    test_client, TestSessionLocal, chroma = client
    from unittest.mock import Mock
    from src.api.app import get_ollama

    mock_ollama = Mock()
    mock_ollama.generate.return_value = {
        "response": "This is a test answer.",
        "model": "test-model",
    }

    from src.api.app import app
    app.dependency_overrides[get_ollama] = lambda: mock_ollama

    # Ingest document first
    doc_text = "# Document\n\nContent for testing."
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(doc_text)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            test_client.post(
                "/ingest",
                files={"file": ("test.txt", f, "text/plain")}
            )

        # Now test chat with provenance
        response = test_client.post("/chat", json={
            "query": "What is in the document?",
            "top_k": 5,
        })
        assert response.status_code == 200
        data = response.json()

        # Verify UI-02 requirements: answer + model used + chunks + versions
        assert "answer" in data
        assert "model_used" in data
        assert "chunk_ids" in data
        assert "resource_ids" in data
        assert "versions" in data
        assert "query_type" in data
        assert "chunk_count" in data
    finally:
        os.unlink(tmp_path)
        app.dependency_overrides.clear()
