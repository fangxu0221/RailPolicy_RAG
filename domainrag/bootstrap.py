"""配置、冻结校验和检索器装配。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from .infrastructure.retrievers.bm25 import BM25Retriever
from .chunking import chunk_documents
from .infrastructure.retrievers.embedding import (
    DEFAULT_MODEL,
    DEFAULT_QUERY_INSTRUCTION,
    NORMALIZATION,
    EmbeddingRetriever,
    embedding_dimension,
    encoder_version,
    encode_texts,
    load_encoder,
)
from .infrastructure.retrievers.hybrid import HybridRetriever
from .infrastructure.retrievers.qdrant import QdrantRetriever
from .infrastructure.storage.qdrant import QdrantStore, snapshot_manifest
from .io import load_config, load_documents
from .models import Chunk, Document
from .infrastructure.retrievers.reranker import (
    DEFAULT_RERANKER_BACKEND,
    DEFAULT_RERANKER_FILE,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
)
from .domain.protocols import Retriever
from .frozen import verify_frozen_dataset


def project_path(config_path: Path, value: str) -> Path:
    """把配置中的项目相对路径解析为不依赖当前终端位置的绝对路径。"""

    # 配置放在 config/ 目录中，因此其父目录的父目录就是项目根目录。
    project_root = config_path.resolve().parent.parent
    return (project_root / value).resolve()


def build_bm25(chunks: list[Chunk], config: dict[str, object]) -> BM25Retriever:
    """从配置读取 BM25 超参数并构造稀疏检索器。"""

    bm25_config = dict(config.get("bm25", {}))
    return BM25Retriever(
        chunks,
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )


def _embedding_options(config: dict[str, object]) -> tuple[str, str, int, str | None]:
    embedding_config = dict(config.get("embedding", {}))
    revision = embedding_config.get("model_revision")
    return (
        str(embedding_config.get("model", DEFAULT_MODEL)),
        str(embedding_config.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)),
        int(embedding_config.get("batch_size", 32)),
        str(revision) if revision is not None else None,
    )


def build_embedding(
    chunks: list[Chunk], config: dict[str, object], *, prefer_cached_model: bool = False
) -> EmbeddingRetriever:
    """从配置读取模型、查询指令和批大小并构造稠密检索器。"""

    model, instruction, batch_size, revision = _embedding_options(config)
    return EmbeddingRetriever(
        chunks, model, instruction, batch_size,
        prefer_cached_model=prefer_cached_model, revision=revision,
    )


def _chunk_snapshot(documents: list[Document], config: dict[str, object]) -> list[Chunk]:
    return chunk_documents(documents, int(config["chunk_size"]), int(config["chunk_overlap"]))


def _qdrant_location(config: dict[str, object], config_path: Path | None) -> tuple[Path, str]:
    options = dict(config["qdrant"])
    path = str(options["path"])
    resolved = project_path(config_path, path) if config_path is not None else Path(path).resolve()
    return resolved, str(options["collection"])


def _prepare_qdrant(documents: list[Document], chunks: list[Chunk], config: dict[str, object]):
    manifest = snapshot_manifest(documents, chunks)
    model, instruction, batch_size, revision = _embedding_options(config)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoder, loaded_path = load_encoder(model, prefer_cached_model=bool(revision), revision=revision)
    manifest.update(
        chunk_size=int(config["chunk_size"]),
        chunk_overlap=int(config["chunk_overlap"]),
        model_name=model,
        model_revision=encoder_version(encoder, loaded_path, revision),
        vector_size=embedding_dimension(encoder),
        distance="dot",
        normalization=NORMALIZATION,
        query_instruction=instruction,
    )
    return manifest, encoder, batch_size


def build_qdrant_index_from_documents(
    documents: list[Document], config: dict[str, object], config_path: Path, *, rebuild: bool = False
) -> dict[str, object]:
    if str(config.get("retriever", "")).lower() not in {"qdrant", "hybrid_qdrant"}:
        raise ValueError("index-build requires a 'qdrant' or 'hybrid_qdrant' config")
    path, collection = _qdrant_location(config, config_path)
    chunks = _chunk_snapshot(documents, config)
    with QdrantStore(path, collection, create=True) as store:
        manifest, encoder, batch_size = _prepare_qdrant(documents, chunks, config)
        encode = partial(encode_texts, encoder, batch_size=batch_size)
        created = store.build(manifest, chunks, encode, batch_size, rebuild=rebuild)
    return {
        "status": "built" if created else "reused",
        "path": str(path),
        "collection": collection,
        "chunk_count": manifest["chunk_count"],
        "model_revision": manifest["model_revision"],
        "vector_size": manifest["vector_size"],
    }


def build_qdrant_index(config_path: Path, *, rebuild: bool = False) -> dict[str, object]:
    config, documents = load_retrieval_snapshot(config_path)
    return build_qdrant_index_from_documents(documents, config, config_path, rebuild=rebuild)


def _open_qdrant(
    documents: list[Document], chunks: list[Chunk], config: dict[str, object], config_path: Path | None
) -> QdrantRetriever:
    store = QdrantStore(*_qdrant_location(config, config_path))
    try:
        manifest, encoder, batch_size = _prepare_qdrant(documents, chunks, config)
        store.validate(manifest)
        return QdrantRetriever(
            store, encoder, query_instruction=manifest["query_instruction"],
            batch_size=batch_size, vector_size=manifest["vector_size"],
        )
    except BaseException:
        store.close()
        raise


def load_retrieval_snapshot(config_path: Path) -> tuple[dict[str, object], list[Document]]:
    """先校验冻结清单，再读取供检索、工具和评测共同使用的文档快照。"""
    config = load_config(config_path)
    verify_frozen_dataset(config_path)
    documents = load_documents(project_path(config_path, str(config["docs_path"])))
    return config, documents


def build_retriever(config_path: Path) -> tuple[Retriever, dict[str, object]]:
    """从配置装配检索器，并保留原配置供报告引用。"""
    config, documents = load_retrieval_snapshot(config_path)
    return build_retriever_from_documents(documents, config, config_path=config_path), config


def build_retriever_from_documents(
    documents: list[Document], config: dict[str, object], *,
    embedding_builder: Callable[[list[Chunk], dict[str, object]], Retriever] | None = None,
    config_path: Path | None = None,
) -> Retriever:
    """从调用方已加载的快照装配；文件加载和冻结检查由入口负责。"""

    chunks = _chunk_snapshot(documents, config)
    build_dense = embedding_builder or build_embedding
    retriever_name = str(config.get("retriever", "bm25")).lower()
    qdrant = None
    if retriever_name in {"qdrant", "hybrid_qdrant"}:
        qdrant = _open_qdrant(documents, chunks, config, config_path)
    try:
        if retriever_name == "bm25":
            retriever: Retriever = build_bm25(chunks, config)
        elif retriever_name in {"embedding", "qdrant"}:
            retriever = qdrant or build_dense(chunks, config)
        elif retriever_name in {"hybrid", "hybrid_qdrant"}:
            hybrid_config = dict(config.get("hybrid", {}))
            retriever = HybridRetriever(
                [build_bm25(chunks, config), qdrant or build_dense(chunks, config)],
                candidate_k=int(hybrid_config.get("candidate_k", 10)),
                rrf_k=int(hybrid_config.get("rrf_k", 60)),
            )
        else:
            raise ValueError(
                f"unsupported retriever {retriever_name!r}; "
                "choose 'bm25', 'embedding', 'hybrid', 'qdrant', or 'hybrid_qdrant'"
            )
        return _with_reranker(retriever, config)
    except BaseException:
        if qdrant is not None:
            qdrant.close()
        raise


def _with_reranker(retriever: Retriever, config: dict[str, object]) -> Retriever:
    reranker_config = dict(config.get("reranker", {}))
    if bool(reranker_config.get("enabled", False)):
        reranker_backend = str(
            reranker_config.get("backend", DEFAULT_RERANKER_BACKEND)
        )
        configured_model_file = reranker_config.get("model_file")
        # ONNX 使用配置文件或默认量化模型；其他后端不需要 ONNX 文件。
        reranker_model_file = (
            str(configured_model_file)
            if configured_model_file
            else DEFAULT_RERANKER_FILE if reranker_backend == "onnx" else None
        )
        retriever = CrossEncoderReranker(
            retriever,
            model_name=str(
                reranker_config.get("model", DEFAULT_RERANKER_MODEL)
            ),
            candidate_k=int(reranker_config.get("candidate_k", 10)),
            batch_size=int(reranker_config.get("batch_size", 4)),
            max_length=int(reranker_config.get("max_length", 512)),
            backend=reranker_backend,
            model_file=reranker_model_file,
        )
    return retriever
