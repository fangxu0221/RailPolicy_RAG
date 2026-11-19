"""用同一 BGE 快照对比内存检索与临时 Qdrant Local 索引。"""

from __future__ import annotations

import argparse
from functools import partial
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from domainrag.bootstrap import (
    build_embedding,
    build_qdrant_index_from_documents,
    build_retriever_from_documents,
    load_retrieval_snapshot,
    project_path,
)
from domainrag.domain.protocols import Retriever
from domainrag.evaluation import evaluate_retrieval
from domainrag.frozen import text_sha256
from domainrag.io import load_evaluation_cases
from domainrag.models import SearchResult


class TimedRetriever:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.query_ms: list[float] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        started = perf_counter()
        results = self.retriever.search(query, top_k)
        self.query_ms.append((perf_counter() - started) * 1000)
        return results


def evaluate(retriever, cases, top_k, documents):
    retriever.search(cases[0].question, top_k)
    timed = TimedRetriever(retriever)
    report = evaluate_retrieval(timed, cases, top_k=top_k, documents=documents)
    report["query_ms"] = {
        "median": round(median(timed.query_ms), 3),
        "max": round(max(timed.query_ms), 3),
        "by_case": {case.case_id: round(elapsed, 3) for case, elapsed in zip(cases, timed.query_ms)},
    }
    return report


def compare_rankings(memory_report, qdrant_report):
    rows = []
    for memory, qdrant in zip(memory_report["cases"], qdrant_report["cases"], strict=True):
        memory_ids = [item["chunk_id"] for item in memory["retrieved"]]
        qdrant_ids = [item["chunk_id"] for item in qdrant["retrieved"]]
        rows.append({
            "id": memory["id"],
            "same_order": memory_ids == qdrant_ids,
            "shared_chunks": len(set(memory_ids) & set(qdrant_ids)),
            "memory": memory_ids,
            "qdrant": qdrant_ids,
        })
    return rows


def run(config_path: Path, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    # Compare initialized runtimes. The measurements still include loading model weights.
    import_module("sentence_transformers")
    import_module("qdrant_client")
    config, documents = load_retrieval_snapshot(config_path)
    if config.get("retriever") != "qdrant" or config.get("reranker", {}).get("enabled"):
        raise ValueError("performance comparison requires a dense-only qdrant config")
    if not config.get("evidence_evaluation"):
        raise ValueError("performance comparison requires an Evidence evaluation config")
    eval_path = project_path(config_path, str(config["eval_path"]))
    cases = load_evaluation_cases(eval_path)
    top_k = int(config.get("top_k", 3))

    started = perf_counter()
    memory = build_retriever_from_documents(
        documents, {**config, "retriever": "embedding"},
        embedding_builder=partial(build_embedding, prefer_cached_model=True),
    )
    memory_start = perf_counter() - started
    memory_report = evaluate(memory, cases, top_k, documents)
    del memory

    with TemporaryDirectory(prefix="railpolicy-qdrant-bench-") as folder:
        local_config = {**config, "qdrant": {**config["qdrant"], "path": folder}}
        started = perf_counter()
        index = build_qdrant_index_from_documents(documents, local_config, config_path)
        first_build = perf_counter() - started
        started = perf_counter()
        qdrant = build_retriever_from_documents(documents, local_config, config_path=config_path)
        qdrant_start = perf_counter() - started
        try:
            qdrant_report = evaluate(qdrant, cases, top_k, documents)
        finally:
            qdrant.close()

    return {
        "config": config,
        "config_sha256": text_sha256(config_path),
        "eval_sha256": text_sha256(eval_path),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "qdrant-client": version("qdrant-client"),
            "sentence-transformers": version("sentence-transformers"),
            "torch": torch.__version__,
            "torch_threads": threads,
        },
        "documents": len(documents),
        "chunks": index["chunk_count"],
        "model_revision": index["model_revision"],
        "vector_size": index["vector_size"],
        "top_k": top_k,
        "startup_seconds": {
            "in_memory_build": round(memory_start, 3),
            "qdrant_first_build": round(first_build, 3),
            "qdrant_reopen": round(qdrant_start, 3),
        },
        "memory": memory_report,
        "qdrant": qdrant_report,
        "rankings": compare_rankings(memory_report, qdrant_report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a fresh path")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    report = run(args.config, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "startup_seconds": report["startup_seconds"],
        "same_rankings": sum(row["same_order"] for row in report["rankings"]),
        "cases": len(report["rankings"]),
        "memory_summary": report["memory"]["summary"],
        "qdrant_summary": report["qdrant"]["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
