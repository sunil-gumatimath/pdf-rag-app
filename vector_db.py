from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import get_settings


class QdrantStorage:
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()
        self.collection = collection or settings.qdrant_collection
        self.dim = dim or settings.embedding_dim
        self.client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
            timeout=30,
        )

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids, vectors, and payloads must have the same length.")

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> dict[str, list[str]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )

        contexts: list[str] = []
        sources: set[str] = set()

        for point in results.points:
            payload = getattr(point, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if isinstance(text, str) and text.strip():
                contexts.append(text.strip())

            if isinstance(source, str) and source.strip():
                sources.add(source.strip())

        return {"contexts": contexts, "sources": sorted(sources)}
