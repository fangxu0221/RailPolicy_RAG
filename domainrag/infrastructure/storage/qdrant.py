"""Qdrant Local 的 Collection、索引校验与显式构建。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from weakref import finalize

from ...models import Chunk, Document, SearchResult


OWNER = "railpolicy_rag/qdrant"


class QdrantIndexError(RuntimeError):
    """索引缺失、失效或无法安全地替换。"""


def _digest(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def snapshot_manifest(documents: list[Document], chunks: list[Chunk]) -> dict:
    """为 Document 快照及切块结果生成与读取顺序无关的指纹。"""

    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    chunk_ids = [chunk.chunk_id for chunk in ordered]
    if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Qdrant requires nonempty chunks with unique chunk IDs")
    return {
        "format_version": 1,
        "document_sha256": _digest(sorted((doc.source, doc.text) for doc in documents)),
        "chunk_sha256": _digest([asdict(chunk) for chunk in ordered]),
        "chunk_count": len(ordered),
        "chunk_ids": chunk_ids,
    }


def _point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{OWNER}/{chunk_id}"))


class QdrantStore:
    def __init__(self, path: Path, collection: str, *, create: bool = False):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}", collection):
            raise ValueError("qdrant.collection must contain only letters, digits, '_' or '-'")
        self.collection = collection
        if not create and not path.is_dir():
            raise self._invalid(f"database directory missing at {path}")
        try:
            import portalocker
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError(
                'Qdrant Local requires python -m pip install -e ".[qdrant]"'
            ) from exc
        if create:
            path.mkdir(parents=True, exist_ok=True)
        self.models = models
        with ExitStack() as cleanup:
            # Qdrant Local loads disk state before taking its own lock.
            # Serialize our opens before that read as well as during rebuild.
            try:
                cleanup.enter_context(portalocker.Lock(
                    path / ".railpolicy.lock", mode="a", timeout=0, fail_when_locked=True
                ))
            except portalocker.exceptions.LockException as exc:
                raise QdrantIndexError(f"Qdrant Local directory already in use: {path}") from exc
            self.client = QdrantClient(path=str(path))
            cleanup.callback(self.client.close)
            self._close = finalize(self, cleanup.pop_all().close)

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._close()

    def _invalid(self, reason: str) -> QdrantIndexError:
        return QdrantIndexError(
            f"Qdrant index {self.collection!r}: {reason}. "
            "Run index-build --config <config>; add --rebuild to replace a stale managed index."
        )

    def _collection_config(self):
        if not self.client.collection_exists(self.collection):
            raise self._invalid("collection missing")
        names = {item.name for item in self.client.get_collections().collections}
        if self.collection not in names:
            raise self._invalid("configured name is an alias; choose a direct collection name")
        return self.client.get_collection(self.collection).config

    def _validate_points(self, manifest: dict) -> None:
        expected = {_point_id(chunk_id) for chunk_id in manifest["chunk_ids"]}
        if self.client.count(self.collection, exact=True).count != manifest["chunk_count"]:
            raise self._invalid("point count differs from the manifest")
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, offset=offset, limit=256, with_payload=False, with_vectors=False
            )
            for point in points:
                if str(point.id) not in expected:
                    raise self._invalid("point IDs differ from the manifest")
                expected.remove(str(point.id))
            if offset is None:
                break
        if expected:
            raise self._invalid("point IDs differ from the manifest")

    def validate(self, manifest: dict) -> None:
        config = self._collection_config()
        metadata = config.metadata or {}
        if metadata.get("owner") != OWNER:
            raise self._invalid("collection is not managed by RailPolicy_RAG")
        saved = metadata.get("manifest")
        if not isinstance(saved, dict):
            raise self._invalid("index build did not complete")
        if saved != manifest:
            fields = sorted(key for key in saved.keys() | manifest.keys()
                            if saved.get(key) != manifest.get(key))
            raise self._invalid(f"manifest mismatch ({', '.join(fields)})")
        vectors = config.params.vectors
        if (not isinstance(vectors, self.models.VectorParams)
                or vectors.size != manifest["vector_size"]
                or vectors.distance != self.models.Distance.DOT):
            raise self._invalid("collection vector dimension or distance differs from the manifest")
        self._validate_points(manifest)

    @staticmethod
    def validate_vector(vector: list[float], size: int) -> None:
        if len(vector) != size or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"encoder must return finite vectors of dimension {size}")

    def build(
        self,
        manifest: dict,
        chunks: list[Chunk],
        encode: Callable[[list[str]], list[list[float]]],
        batch_size: int,
        *,
        rebuild: bool = False,
    ) -> bool:
        """构建新索引。相同快照直接复用，替换限显式指定的项目 Collection。"""

        if self.client.collection_exists(self.collection):
            config = self._collection_config()
            if (config.metadata or {}).get("owner") != OWNER:
                raise self._invalid("refusing to replace a collection owned by another application")
            try:
                self.validate(manifest)
            except QdrantIndexError:
                if not rebuild:
                    raise
                # Refuse implicit side effects on aliases even during explicit rebuilds.
                if any(alias.collection_name == self.collection
                       for alias in self.client.get_aliases().aliases):
                    raise self._invalid("remove external aliases or choose a fresh collection")
                # Local delete_collection drops its reference before removing the directory.
                # Close SQLite explicitly so Windows can remove it before reusing the name.
                self.client._client.collections[self.collection].close()
                if not self.client.delete_collection(self.collection):
                    raise self._invalid("failed to remove the stale managed collection")
            else:
                return False
        size = manifest["vector_size"]
        if not self.client.create_collection(
            self.collection,
            vectors_config=self.models.VectorParams(size=size, distance=self.models.Distance.DOT),
            metadata={"owner": OWNER},
        ):
            raise self._invalid("failed to create collection")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = encode([chunk.text for chunk in batch])
            points = []
            for chunk, vector in zip(batch, vectors, strict=True):
                self.validate_vector(vector, size)
                points.append(self.models.PointStruct(
                    id=_point_id(chunk.chunk_id), vector=vector, payload=asdict(chunk)
                ))
            self.client.upsert(self.collection, points=points, wait=True)
        self._validate_points(manifest)
        # Publish the ready manifest only after every write and the inventory check succeed.
        if not self.client.update_collection(self.collection, metadata={"manifest": manifest}):
            raise self._invalid("failed to publish the index manifest")
        return True

    def search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        hits = self.client.query_points(
            self.collection, query=vector, limit=top_k, with_payload=True, with_vectors=False
        ).points
        results = [SearchResult(Chunk(**(hit.payload or {})), float(hit.score)) for hit in hits]
        return sorted(results, key=lambda item: (-item.score, item.chunk.chunk_id))
