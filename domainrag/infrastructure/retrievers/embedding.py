"""Embedding 稠密检索：把问题和 Chunk 映射到同一个向量空间后比较相似度。

初始化时一次性编码全部 Chunk；每次查询只编码问题，然后用归一化向量的点积
（等价于余弦相似度）排序。内存基线与 Qdrant 共用模型加载和向量归一化。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from ...models import Chunk, SearchResult


DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
NORMALIZATION = "encoder_normalize_embeddings_then_l2_preserve_zero_v1"


class Encoder(Protocol):
    """SentenceTransformer 及测试用假 Encoder 共同遵守的接口。"""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


def _as_vector(value: object) -> list[float]:
    """把 numpy 数组或普通序列统一转换为 Python 浮点列表。"""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("encoder output must be a sequence of vectors")
    return [float(item) for item in value]


def _normalize(vector: list[float]) -> list[float]:
    """进行 L2 归一化，使向量长度为 1。"""

    # ||v||₂ = sqrt(v1² + v2² + ... + vn²)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def load_encoder(
    model_name: str, *, prefer_cached_model: bool = False, revision: str | None = None
) -> tuple[Encoder, Path | None]:
    """加载 SentenceTransformer，并返回实际使用的本地目录（可用时）。"""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding retrieval requires sentence-transformers. "
            'Install it with: python -m pip install -e ".[embedding]"'
        ) from exc
    model_path = Path(model_name)
    if model_path.is_dir():
        return SentenceTransformer(model_name), model_path
    revision_options = {"revision": revision} if revision else {}
    if prefer_cached_model:
        from huggingface_hub import try_to_load_from_cache

        cached_modules = try_to_load_from_cache(model_name, "modules.json", **revision_options)
        if isinstance(cached_modules, str):
            cached_path = Path(cached_modules).parent
            try:
                return SentenceTransformer(str(cached_path), local_files_only=True), cached_path
            except OSError:
                # 缓存不完整时，按既有流程补齐模型文件。
                pass
    return SentenceTransformer(model_name, **revision_options), None


def encode_texts(encoder: Encoder, texts: list[str], batch_size: int) -> list[list[float]]:
    """沿用 BGE 的批编码和 L2 归一化，包括零向量处理。"""

    encoded = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, Sequence):
        raise TypeError("encoder output must contain one vector per input text")
    vectors = [_normalize(_as_vector(vector)) for vector in encoded]
    if len(vectors) != len(texts):
        raise ValueError("encoder returned a different number of vectors than inputs")
    return vectors


def encoder_version(encoder: Encoder, path: Path | None, revision: str | None) -> str:
    """使用实际的 HF 快照版本，普通本地模型目录则按文件内容生成指纹。"""

    if path is not None:
        if (path.parent.name == "snapshots"
                and path.parent.parent.name.startswith("models--")
                and re.fullmatch(r"[0-9a-f]{40}", path.name)):
            return path.name
        digest = hashlib.sha256()
        for file in sorted(path.rglob("*")):
            if not file.is_file():
                continue
            file_digest = hashlib.sha256()
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(block)
            digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
            digest.update(file_digest.digest())
        return f"sha256:{digest.hexdigest()}"
    first_module = getattr(encoder, "_first_module", lambda: None)()
    config = getattr(getattr(first_module, "auto_model", None), "config", None)
    commit = getattr(config, "_commit_hash", None) or revision
    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
        return commit
    raise ValueError("Cannot identify the embedding model revision; set embedding.model_revision to a commit SHA")


def embedding_dimension(encoder: Encoder) -> int:
    get_dimension = getattr(encoder, "get_embedding_dimension", None)
    if get_dimension is None:
        get_dimension = getattr(encoder, "get_sentence_embedding_dimension", None)
    dimension = get_dimension() if get_dimension else None
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Qdrant requires a model with a positive embedding dimension")
    return dimension


class EmbeddingRetriever:
    """使用本地 Sentence Transformers 模型执行稠密语义检索。"""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DEFAULT_MODEL,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 32,
        encoder: Encoder | None = None,
        *,
        prefer_cached_model: bool = False,
        revision: str | None = None,
    ):
        if not chunks:
            raise ValueError("EmbeddingRetriever requires at least one chunk")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if encoder is None:
            encoder, _ = load_encoder(
                model_name, prefer_cached_model=prefer_cached_model, revision=revision
            )

        self.chunks = chunks
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.encoder = encoder
        # 文档向量只在初始化时计算一次，后续多个问题可以复用。
        self.chunk_embeddings = self._encode([chunk.text for chunk in chunks])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return encode_texts(self.encoder, texts, self.batch_size)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """编码问题、计算它与所有 Chunk 的余弦相似度，并取 Top-K。"""

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # BGE 检索模型建议只给“问题”添加检索指令，文档 Chunk 不添加。
        query_text = f"{self.query_instruction}{query}" if self.query_instruction else query
        query_vector = self._encode([query_text])[0]
        scored: list[SearchResult] = []
        for chunk, chunk_vector in zip(self.chunks, self.chunk_embeddings):
            if len(chunk_vector) != len(query_vector):
                raise ValueError("query and document embedding dimensions do not match")
            # 两个向量都已归一化，因此点积就是 cosine similarity；越大越相关。
            score = sum(
                query_value * chunk_value
                for query_value, chunk_value in zip(query_vector, chunk_vector)
            )
            scored.append(SearchResult(chunk=chunk, score=score))

        # 相同分数使用 chunk_id 打破平局，保证每次运行顺序一致。
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[: min(top_k, len(scored))]
