import tempfile
from src.chunking.splitter import split_by_headings
from src.chunking.embedder import embed_chunks
from src.vectorstore.chroma_client import ChromaClient

SAMPLE_MD = """\
# Introduction

This is the introduction with enough text to form a standalone chunk.

## Background

This section covers relevant prior work and context for the study.

## Methods

Here we describe the experimental methodology used in detail.
"""


def test_chunk_boundaries():
    """CHUNK-01: chunk boundaries align with heading boundaries."""
    chunks = split_by_headings(SAMPLE_MD, resource_id="r1", version=1, overlap_chars=0)
    assert len(chunks) >= 3
    headings = [c.heading for c in chunks]
    assert "# Introduction" in headings
    assert "## Background" in headings
    assert "## Methods" in headings


def test_chunk_ids_unique():
    chunks = split_by_headings(SAMPLE_MD, resource_id="r1", version=1)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_overlap_prepended():
    chunks = split_by_headings(SAMPLE_MD, resource_id="r1", version=1, overlap_chars=50)
    if len(chunks) >= 2:
        tail_of_first = chunks[0].text[-50:]
        assert tail_of_first in chunks[1].text


def test_long_section_sub_split():
    long_body = "\n\n".join([f"Paragraph {i}. " + "word " * 50 for i in range(10)])
    md = f"# Long Section\n\n{long_body}"
    chunks = split_by_headings(md, resource_id="r1", version=1, max_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.heading == "# Long Section"


def test_embedding_generation():
    """CHUNK-02: embed_chunks returns a valid float vector per input text."""
    vecs = embed_chunks(["Hello world", "Another sentence."])
    assert len(vecs) == 2
    for vec in vecs:
        assert len(vec) == 384  # all-MiniLM-L6-v2 output dimension
        assert all(isinstance(v, float) for v in vec)


def test_version_isolation():
    """CHUNK-03: after re-indexing with v2, only v2 chunks are returned by query."""
    # ignore_cleanup_errors: Chroma holds HNSW file handles via background threads on Windows
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        chroma = ChromaClient(persist_dir=tmp_dir)

        v1_chunks = split_by_headings(
            "# Alpha\n\nVersion one content here.",
            resource_id="res-iso",
            version=1,
            overlap_chars=0,
        )
        chroma.upsert(v1_chunks, embed_chunks([c.text for c in v1_chunks]))

        v2_chunks = split_by_headings(
            "# Alpha\n\nVersion two content here.",
            resource_id="res-iso",
            version=2,
            overlap_chars=0,
        )
        chroma.deactivate_old_versions("res-iso", 2)
        chroma.upsert(v2_chunks, embed_chunks([c.text for c in v2_chunks]))

        query_vec = embed_chunks(["Alpha content"])[0]
        results = chroma.query(query_vec, top_k=10)
        assert results, "expected at least one result"
        returned_versions = {r["metadata"]["version"] for r in results}
        assert returned_versions == {2}, f"expected only v2, got versions {returned_versions}"
