from pathlib import Path
from src.chunking.splitter import split_by_headings
from src.chunking.embedder import embed_chunks
from src.vectorstore.chroma_client import ChromaClient
from src.db.models import Resource

_chroma: ChromaClient | None = None


def _get_chroma() -> ChromaClient:
    """Lazily initialize the default Chroma client (used in production)."""
    global _chroma
    if _chroma is None:
        _chroma = ChromaClient()
    return _chroma


def index_resource(
    resource_id: str,
    version: int,
    markdown_path: str,
    session,
    chroma: ChromaClient | None = None,
) -> int:
    """
    Load markdown → chunk → embed → deactivate old version chunks → upsert new chunks.
    Updates Resource.status to 'indexed' on success.
    Returns the number of chunks indexed.
    Accepts an optional chroma instance for test injection; defaults to singleton.
    """
    if chroma is None:
        chroma = _get_chroma()

    text = Path(markdown_path).read_text(encoding="utf-8")
    chunks = split_by_headings(text, resource_id=resource_id, version=version)
    if not chunks:
        return 0

    embeddings = embed_chunks([c.text for c in chunks])
    chroma.deactivate_old_versions(resource_id, version)
    chroma.upsert(chunks, embeddings)

    resource = session.query(Resource).filter_by(
        resource_id=resource_id, version=version
    ).first()
    if resource:
        resource.status = "indexed"
        session.commit()

    return len(chunks)
