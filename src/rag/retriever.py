from dataclasses import dataclass
from src.chunking.embedder import embed_chunks
from src.vectorstore.chroma_client import ChromaClient


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    resource_id: str
    version: int
    heading: str
    distance: float


def retrieve_context(
    query: str,
    chroma: ChromaClient,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    RAG-01: Retrieve latest-version chunks only.
    Embed the query and search Chroma, which filters by is_latest=1.
    """
    query_embedding = embed_chunks([query])[0]
    results = chroma.query(query_embedding, top_k=top_k)
    return [
        RetrievalResult(
            chunk_id=r["chunk_id"],
            text=r["text"],
            resource_id=r["metadata"]["resource_id"],
            version=r["metadata"]["version"],
            heading=r["metadata"]["heading"],
            distance=r["distance"],
        )
        for r in results
    ]


def build_context_prompt(
    query: str,
    retrieved_chunks: list[RetrievalResult],
    fallback: bool = False,
) -> str:
    """
    Build a prompt with retrieved context.
    If no chunks and fallback=True, return a zero-shot prompt.
    """
    if not retrieved_chunks:
        if fallback:
            return f"Question: {query}\n\nPlease answer the question to the best of your ability."
        else:
            return ""

    context = "\n\n".join([
        f"[{r.resource_id} v{r.version} - {r.heading}]\n{r.text}"
        for r in retrieved_chunks
    ])

    return f"""Context from documents:

{context}

---

Question: {query}

Based on the context above, please provide a comprehensive answer."""
