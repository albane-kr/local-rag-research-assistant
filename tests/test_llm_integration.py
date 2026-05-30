import os
import tempfile
from unittest.mock import Mock
from pathlib import Path

from src.api.app import get_ollama


def test_chat_endpoint_with_context(client):
    """Integration test: /chat endpoint returns LLM response with provenance."""
    test_client, TestSessionLocal, chroma = client
    mock_ollama = Mock()
    mock_ollama.generate.return_value = {
        "response": "Climate change is driven by greenhouse gas emissions from human activities.",
        "model": "test-model",
    }

    # Override the Ollama dependency
    from src.api.app import app, get_ollama as original_get_ollama
    app.dependency_overrides[original_get_ollama] = lambda: mock_ollama

    # Ingest a document first
    doc_text = """\
# Climate Science

## Greenhouse Effect

The greenhouse effect is the process by which Earth's atmosphere traps heat.
Carbon dioxide and methane are key greenhouse gases.

## Climate Change

Climate change is caused by human activities that emit greenhouse gases.
These emissions trap more heat in the atmosphere, warming the planet.
"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(doc_text)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as f:
            ingest_response = test_client.post(
                "/ingest",
                files={"file": ("climate.txt", f, "text/plain")}
            )
        assert ingest_response.status_code == 200

        # Now make a chat query
        chat_response = test_client.post("/chat", json={
            "query": "What causes climate change?",
            "top_k": 5,
        })
        assert chat_response.status_code == 200
        data = chat_response.json()

        # Verify response structure (GOV-01: provenance)
        assert "query" in data
        assert "answer" in data
        assert "model_used" in data
        assert "chunk_ids" in data
        assert "resource_ids" in data
        assert "versions" in data
        assert "chunk_count" in data

        # Verify provenance metadata is present
        assert data["chunk_count"] > 0
        assert len(data["chunk_ids"]) > 0
        assert len(data["resource_ids"]) > 0
        assert len(data["versions"]) > 0

        # Verify the LLM response
        assert "climate" in data["answer"].lower()

    finally:
        os.unlink(tmp_path)
        app.dependency_overrides.clear()


def test_chat_endpoint_empty_query(client):
    """Test /chat with no indexed documents."""
    test_client, TestSessionLocal, chroma = client
    mock_ollama = Mock()

    from src.api.app import app, get_ollama as original_get_ollama
    app.dependency_overrides[original_get_ollama] = lambda: mock_ollama

    try:
        response = test_client.post("/chat", json={
            "query": "Tell me about ancient Rome",
            "top_k": 5,
        })
        assert response.status_code == 200
        data = response.json()

        assert data["chunk_count"] == 0
        assert "No relevant documents found" in data["answer"]

    finally:
        app.dependency_overrides.clear()
