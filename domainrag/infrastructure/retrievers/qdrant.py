"""使用现有 BGE 编码查询并从 Qdrant Local 召回 Chunk。"""

from __future__ import annotations

from ...models import SearchResult
from ..storage.qdrant import QdrantStore
from .embedding import Encoder, encode_texts


class QdrantRetriever:
    def __init__(
        self,
        store: QdrantStore,
        encoder: Encoder,
        *,
        query_instruction: str,
        batch_size: int,
        vector_size: int,
    ):
        self.store = store
        self.encoder = encoder
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.vector_size = vector_size

    def close(self) -> None:
        self.store.close()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = encode_texts(self.encoder, [f"{self.query_instruction}{query}"], self.batch_size)[0]
        self.store.validate_vector(vector, self.vector_size)
        return self.store.search(vector, top_k)
