import tempfile
from src.rag.retriever import retrieve_context, build_context_prompt, RetrievalResult
from src.chunking.splitter import split_by_headings
from src.chunking.embedder import embed_chunks
from src.vectorstore.chroma_client import ChromaClient


def test_latest_version_retrieval():
    """RAG-01: Only latest-version chunks are returned on query."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        chroma = ChromaClient(persist_dir=tmp_dir)

        # Index v1
        v1_chunks = split_by_headings(
            "# Machine Learning\n\nVersion 1: Classic approaches using decision trees.",
            resource_id="ml-paper",
            version=1,
            overlap_chars=0,
        )
        chroma.upsert(v1_chunks, embed_chunks([c.text for c in v1_chunks]))

        # Index v2 (deactivates v1)
        v2_chunks = split_by_headings(
            "# Machine Learning\n\nVersion 2: Modern deep learning approaches.",
            resource_id="ml-paper",
            version=2,
            overlap_chars=0,
        )
        chroma.deactivate_old_versions("ml-paper", 2)
        chroma.upsert(v2_chunks, embed_chunks([c.text for c in v2_chunks]))

        results = retrieve_context("machine learning approaches", chroma, top_k=5)
        assert len(results) > 0
        for r in results:
            assert r.version == 2, f"expected v2, got v{r.version}"


def test_empty_retrieval_fallback():
    """RAG-03: Empty retrieval with fallback produces zero-shot prompt."""
    empty_chunks: list[RetrievalResult] = []

    prompt_with_fallback = build_context_prompt(
        "What is the meaning of life?",
        empty_chunks,
        fallback=True,
    )
    assert "Question:" in prompt_with_fallback
    assert "What is the meaning of life?" in prompt_with_fallback
    assert "Please answer" in prompt_with_fallback or "ability" in prompt_with_fallback

    prompt_without_fallback = build_context_prompt(
        "What is the meaning of life?",
        empty_chunks,
        fallback=False,
    )
    assert prompt_without_fallback == ""


def test_context_prompt_building():
    """Test that retrieved chunks are formatted into a prompt correctly."""
    chunks = [
        RetrievalResult(
            chunk_id="r1_v1_0",
            text="Deep learning uses neural networks with many layers.",
            resource_id="r1",
            version=1,
            heading="## Deep Learning",
            distance=0.1,
        ),
        RetrievalResult(
            chunk_id="r2_v1_0",
            text="Transformers revolutionized NLP by using attention mechanisms.",
            resource_id="r2",
            version=1,
            heading="## Transformers",
            distance=0.15,
        ),
    ]

    prompt = build_context_prompt("Explain neural networks", chunks, fallback=False)
    assert "Deep Learning" in prompt
    assert "neural networks" in prompt
    assert "Transformers" in prompt
    assert "attention mechanisms" in prompt
    assert "Explain neural networks" in prompt
