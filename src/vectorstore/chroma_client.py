class ChromaClient:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir

    def add(self, embeddings):
        # stub: persist embeddings to Chroma instance
        raise NotImplementedError()

    def query(self, q_embedding, top_k: int = 5):
        # stub: query Chroma
        return []
