from unittest.mock import Mock
from src.llm.orchestrator import (
    classify_query,
    select_model,
    generate_response,
    QueryType,
    MODEL_ROUTING,
)
from src.rag.retriever import RetrievalResult


def test_query_classification():
    """Test query type classification."""
    assert classify_query("summarize this paper") == QueryType.SUMMARY
    assert classify_query("give me a brief overview") == QueryType.SUMMARY

    assert classify_query("compare method A vs B") == QueryType.COMPARE
    assert classify_query("what is the difference?") == QueryType.COMPARE

    assert classify_query("explain the process") == QueryType.ANALYZE
    assert classify_query("discuss the findings") == QueryType.ANALYZE

    assert classify_query("what is this?") == QueryType.DEFAULT


def test_model_selection():
    """LLM-01/02/03: Select appropriate model for query type."""
    assert select_model(QueryType.SUMMARY) == "gemma"
    assert select_model(QueryType.COMPARE) == "nemotron"
    assert select_model(QueryType.ANALYZE) == "neural-chat"
    assert select_model(QueryType.DEFAULT) == "llama2"


def test_generate_response_with_mock_ollama():
    """Test response generation with mocked Ollama client."""
    mock_ollama = Mock()
    mock_ollama.generate.return_value = {
        "response": "This is a summary of the document.",
        "model": "gemma",
    }

    chunks = [
        RetrievalResult(
            chunk_id="r1_v1_0",
            text="Document content here.",
            resource_id="r1",
            version=1,
            heading="## Section",
            distance=0.1,
        ),
        RetrievalResult(
            chunk_id="r1_v1_1",
            text="More content.",
            resource_id="r1",
            version=1,
            heading="## Another",
            distance=0.2,
        ),
    ]

    response = generate_response(
        query="summarize this",
        prompt="[context]...\n\nSummarize: this",
        chunks=chunks,
        ollama=mock_ollama,
    )

    assert response.answer == "This is a summary of the document."
    assert response.model_used == "gemma"
    assert response.query_type == QueryType.SUMMARY
    assert response.chunk_ids == ["r1_v1_0", "r1_v1_1"]
    assert response.resource_ids == ["r1"]
    assert response.versions == [1]

    mock_ollama.generate.assert_called_once_with(
        "[context]...\n\nSummarize: this",
        model="gemma",
    )


def test_provenance_tracking():
    """Test that provenance metadata is correctly tracked."""
    mock_ollama = Mock()
    mock_ollama.generate.return_value = {"response": "Answer", "model": "test"}

    chunks = [
        RetrievalResult("id1", "text1", "doc_a", 2, "H1", 0.1),
        RetrievalResult("id2", "text2", "doc_a", 2, "H2", 0.15),
        RetrievalResult("id3", "text3", "doc_b", 1, "H3", 0.2),
    ]

    response = generate_response("query", "prompt", chunks, mock_ollama)

    assert set(response.chunk_ids) == {"id1", "id2", "id3"}
    assert set(response.resource_ids) == {"doc_a", "doc_b"}
    assert set(response.versions) == {1, 2}
