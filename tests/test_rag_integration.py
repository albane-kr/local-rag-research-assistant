import os
import tempfile
from pathlib import Path

from src.db.models import Resource


def test_query_endpoint_with_indexed_content(client):
    """Integration test: /query returns chunks from indexed documents."""
    test_client, TestSessionLocal, chroma = client

    # First, ingest a document
    doc_text = """\
# Climate Science

## Greenhouse Effect

The greenhouse effect is the process by which Earth's atmosphere traps heat.
Carbon dioxide and methane are key greenhouse gases.

## Global Warming

Global warming refers to the long-term increase in Earth's average temperature
caused by human activities, primarily the emission of greenhouse gases.
"""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(doc_text)
        tmp_path = f.name

    try:
        # Ingest the document
        with open(tmp_path, "rb") as f:
            ingest_response = test_client.post(
                "/ingest",
                files={"file": ("climate.txt", f, "text/plain")}
            )
        assert ingest_response.status_code == 200

        # Now query the indexed content
        query_response = test_client.post("/query", json={
            "query": "What causes global warming?",
            "top_k": 5,
            "fallback": False,
        })
        assert query_response.status_code == 200
        data = query_response.json()

        # Verify chunks were retrieved
        assert "chunk_count" in data
        assert data["chunk_count"] > 0

        # Verify chunks contain relevant content
        chunk_texts = "\n".join([c["text"] for c in data["chunks"]])
        assert "warming" in chunk_texts.lower() or "greenhouse" in chunk_texts.lower()

        # Verify prompt was built
        assert "prompt" in data
        assert len(data["prompt"]) > 0
        assert "What causes global warming?" in data["prompt"]

    finally:
        os.unlink(tmp_path)


def test_query_endpoint_empty_fallback(client):
    """RAG-03: Empty retrieval with fallback returns zero-shot prompt."""
    test_client, TestSessionLocal, chroma = client

    response = test_client.post("/query", json={
        "query": "Discuss ancient Roman philosophy",
        "top_k": 5,
        "fallback": True,  # Enable fallback for empty results
    })
    assert response.status_code == 200
    data = response.json()

    assert data["chunk_count"] == 0
    assert "prompt" in data
    assert len(data["prompt"]) > 0
    assert "Discuss ancient Roman philosophy" in data["prompt"]


def test_query_endpoint_no_fallback(client):
    """If fallback=False and no chunks, prompt should be empty."""
    test_client, TestSessionLocal, chroma = client

    response = test_client.post("/query", json={
        "query": "Something not in any document",
        "top_k": 5,
        "fallback": False,
    })
    assert response.status_code == 200
    data = response.json()

    assert data["chunk_count"] == 0
    assert data["prompt"] == ""
