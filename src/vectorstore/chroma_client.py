import chromadb
from src.chunking.splitter import Chunk


class ChromaClient:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "resource_id": c.resource_id,
                    "version": c.version,
                    "chunk_index": c.chunk_index,
                    "heading": c.heading,
                    "is_latest": c.is_latest,
                }
                for c in chunks
            ],
        )

    def deactivate_old_versions(self, resource_id: str, current_version: int) -> None:
        """
        Set is_latest=0 on all chunks for resource_id whose version != current_version.
        Passes full metadata back to update() because Chroma replaces (not merges) metadata.
        """
        results = self.collection.get(
            where={"$and": [
                {"resource_id": {"$eq": resource_id}},
                {"is_latest": {"$eq": 1}},
            ]}
        )
        old_ids = []
        old_metadatas = []
        for id_, meta in zip(results["ids"], results["metadatas"]):
            if meta["version"] != current_version:
                old_ids.append(id_)
                old_metadatas.append({**meta, "is_latest": 0})

        if old_ids:
            self.collection.update(ids=old_ids, metadatas=old_metadatas)

    def query(self, q_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return up to top_k latest-version chunks nearest to q_embedding."""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
            where={"is_latest": {"$eq": 1}},
        )
        return [
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]
