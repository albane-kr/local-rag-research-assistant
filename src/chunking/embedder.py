from sentence_transformers import SentenceTransformer

# 90 MB, 384-dim vectors, max 256 tokens — good CPU performance for local use
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_chunks(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Return one float vector per text. Loads the model lazily on first call."""
    vectors = _get_model().encode(texts, batch_size=batch_size, show_progress_bar=False)
    return vectors.tolist()
