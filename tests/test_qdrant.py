"""在真实 Qdrant Local 临时目录中测试持久化，模型使用离线替身。"""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch
from uuid import uuid4

from domainrag import bootstrap
from domainrag.application.knowledge_service import KnowledgeTools
from domainrag.chunking import chunk_documents
from domainrag.infrastructure.retrievers.bm25 import BM25Retriever
from domainrag.infrastructure.retrievers.embedding import EmbeddingRetriever
from domainrag.infrastructure.retrievers.hybrid import HybridRetriever
from domainrag.infrastructure.retrievers.qdrant import QdrantRetriever
from domainrag.infrastructure.retrievers.reranker import CrossEncoderReranker
from domainrag.infrastructure.storage.qdrant import QdrantIndexError, QdrantStore
from domainrag.models import Document


REVISION = "a" * 40


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.dimension = 2
        self.fail_at = None

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, *, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
        assert not show_progress_bar and convert_to_numpy and normalize_embeddings
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("encoder interrupted")
        padding = [0.0] * (self.dimension - 2)
        return [([3.0, 0.0] if "改签" in text else [0.0, 4.0]) + padding for text in texts]


@unittest.skipUnless(importlib.util.find_spec("qdrant_client"), "install the qdrant extra")
class QdrantTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "configs").mkdir()
        (self.root / "docs").mkdir()
        self.config_path = self.root / "configs/dev.json"
        self.db = self.root / ".qdrant/test"
        self.config = {
            "retriever": "qdrant", "docs_path": "docs",
            "qdrant": {"path": ".qdrant/test", "collection": "test_chunks"},
            "chunk_size": 8, "chunk_overlap": 2,
            "embedding": {"model": "fake/bge", "model_revision": REVISION,
                          "query_instruction": "检索：", "batch_size": 1},
        }
        self.documents = [
            Document("a.md", "Ａ  可改签\n车票"),
            Document("b.md", "退票收费"),
        ]
        for doc in self.documents:
            (self.root / "docs" / doc.source).write_text(doc.text, encoding="utf-8")
        self.write_config()
        self.encoder = FakeEncoder()
        loader = patch.object(bootstrap, "load_encoder", return_value=(self.encoder, None))
        self.loader = loader.start()
        self.addCleanup(loader.stop)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def build(self, **kwargs):
        return bootstrap.build_qdrant_index(self.config_path, **kwargs)

    def load(self):
        retriever, _ = bootstrap.build_retriever(self.config_path)
        self.addCleanup(retriever.close)
        return retriever

    def test_missing_index_never_builds_or_loads_model(self):
        with self.assertRaisesRegex(QdrantIndexError, "index-build"):
            bootstrap.build_retriever(self.config_path)
        self.loader.assert_not_called()
        self.assertFalse(self.db.exists())

    def test_build_reopen_and_search_match_memory_and_restore_payload(self):
        expected = chunk_documents(self.documents, 8, 2)
        created = self.build()
        self.assertEqual((created["status"], created["chunk_count"], created["vector_size"]),
                         ("built", len(expected), 2))
        self.assertEqual(self.encoder.calls, [[chunk.text] for chunk in expected])
        self.encoder.calls.clear()
        opened = self.load()
        self.assertEqual(self.encoder.calls, [])
        self.assertEqual(opened.search("改签", 1)[0].chunk, expected[0])
        opened.close()
        reopened = self.load()
        self.assertEqual(self.encoder.calls, [["检索：改签"]])
        hits = reopened.search("改签", 10)
        self.assertEqual(self.encoder.calls, [["检索：改签"], ["检索：改签"]])
        memory = EmbeddingRetriever(expected, encoder=FakeEncoder(), query_instruction="检索：")
        self.assertEqual(hits, memory.search("改签", 10))
        self.assertEqual([hit.chunk for hit in hits], expected)
        with self.assertRaisesRegex(ValueError, "top_k"):
            reopened.search("改签", 0)

    def test_repeated_build_reuses_vectors_and_stable_ids(self):
        self.build()
        with QdrantStore(self.db, "test_chunks") as store:
            first, _ = store.client.scroll(store.collection)
        self.encoder.calls.clear()
        self.assertEqual(self.build()["status"], "reused")
        self.assertEqual(self.build(rebuild=True)["status"], "reused")
        self.assertEqual(self.encoder.calls, [])
        with QdrantStore(self.db, "test_chunks") as store:
            second, _ = store.client.scroll(store.collection)
        self.assertEqual(first, second)

    def test_original_document_change_invalidates_even_if_normalized_chunks_match(self):
        self.build()
        (self.root / "docs/a.md").write_text("Ａ 可改签 车票", encoding="utf-8")
        with self.assertRaisesRegex(QdrantIndexError, "document_sha256"):
            bootstrap.build_retriever(self.config_path)
        with self.assertRaisesRegex(QdrantIndexError, "--rebuild"):
            self.build()

    def test_dimension_and_config_mismatches_fail_without_embedding_documents(self):
        self.build()
        self.encoder.calls.clear()
        alternatives = [
            ({"chunk_size": 9}, "chunk_size"),
            ({"chunk_overlap": 1}, "chunk_overlap"),
            ({"embedding": {**self.config["embedding"], "model": "fake/other"}}, "model_name"),
            ({"embedding": {**self.config["embedding"], "model_revision": "b" * 40}}, "model_revision"),
            ({"embedding": {**self.config["embedding"], "query_instruction": ""}}, "query_instruction"),
        ]
        for changes, field in alternatives:
            with self.subTest(field=field), self.assertRaisesRegex(QdrantIndexError, field):
                bootstrap.build_retriever_from_documents(
                    self.documents, {**self.config, **changes}, config_path=self.config_path
                )
        self.encoder.dimension = 3
        with self.assertRaisesRegex(QdrantIndexError, "vector_size"):
            bootstrap.build_retriever(self.config_path)
        self.assertEqual(self.encoder.calls, [])

    def test_explicit_rebuild_removes_deleted_chunks(self):
        self.build()
        (self.root / "docs/a.md").unlink()
        with self.assertRaisesRegex(QdrantIndexError, "manifest mismatch"):
            bootstrap.build_retriever(self.config_path)
        with self.assertRaisesRegex(QdrantIndexError, "manifest mismatch"):
            self.build()
        self.assertEqual(self.build(rebuild=True)["chunk_count"], 1)
        hits = self.load().search("改签", 10)
        self.assertEqual([hit.chunk.source for hit in hits], ["b.md"])

    def test_explicit_rebuild_can_change_vector_dimension(self):
        self.build()
        self.encoder.dimension = 3
        self.assertEqual(self.build(rebuild=True)["vector_size"], 3)
        self.assertEqual(self.load().search("改签", 1)[0].chunk.source, "a.md")

    def test_point_inventory_detects_corruption_with_unchanged_count(self):
        self.build()
        with QdrantStore(self.db, "test_chunks") as store:
            points, _ = store.client.scroll(store.collection, with_vectors=True)
            first = points[0]
            store.client.delete(store.collection, points_selector=[first.id])
            store.client.upsert(store.collection, [store.models.PointStruct(
                id=str(uuid4()), payload=first.payload, vector=first.vector
            )])
        with self.assertRaisesRegex(QdrantIndexError, "point IDs"):
            bootstrap.build_retriever(self.config_path)

    def test_partial_build_fails_closed_and_can_be_replaced(self):
        self.encoder.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.build()
        with self.assertRaisesRegex(QdrantIndexError, "did not complete"):
            bootstrap.build_retriever(self.config_path)
        with self.assertRaisesRegex(QdrantIndexError, "did not complete"):
            self.build()
        self.assertEqual(self.build(rebuild=True)["status"], "built")
        self.assertEqual(len(self.load().search("改签", 3)), 2)

    def test_explicit_rebuild_refuses_unmanaged_collection(self):
        with QdrantStore(self.db, "test_chunks", create=True) as store:
            store.client.create_collection(
                store.collection, store.models.VectorParams(size=2, distance=store.models.Distance.COSINE)
            )
        with self.assertRaisesRegex(QdrantIndexError, "refusing to replace"):
            self.build(rebuild=True)
        with QdrantStore(self.db, "test_chunks") as store:
            self.assertTrue(store.client.collection_exists(store.collection))

    def test_live_reader_blocks_rebuild_of_the_same_directory(self):
        self.build()
        opened = self.load()
        with self.assertRaisesRegex(QdrantIndexError, "already in use"):
            self.build(rebuild=True)
        opened.close()
        self.assertEqual(self.build()["status"], "reused")

    def test_hybrid_rrf_reranker_and_knowledge_tools(self):
        self.build()
        self.config.update(retriever="hybrid_qdrant",
                           hybrid={"candidate_k": 2, "rrf_k": 60},
                           reranker={"enabled": True, "backend": "torch", "candidate_k": 2})
        self.write_config()
        scored = []

        class FakeScorer:
            def predict(self, pairs, **kwargs):
                scored.extend(pairs)
                return list(range(len(pairs)))

        model = ModuleType("sentence_transformers")
        model.CrossEncoder = lambda *args, **kwargs: FakeScorer()
        self.encoder.calls.clear()
        with patch.dict("sys.modules", {"sentence_transformers": model}):
            tools = KnowledgeTools.from_config(self.config_path)
        reranker = tools._retriever
        self.assertIsInstance(reranker, CrossEncoderReranker)
        self.assertIsInstance(reranker.retriever, HybridRetriever)
        bm25, dense = reranker.retriever.retrievers
        self.assertIsInstance(bm25, BM25Retriever)
        self.assertIsInstance(dense, QdrantRetriever)
        self.addCleanup(dense.close)
        self.assertEqual(self.encoder.calls, [])
        result = tools.search_knowledge("改签", 1)["results"][0]
        self.assertEqual(result["source"], "b.md")
        self.assertEqual(scored, [("改签", "A 可改签 车票"), ("改签", "退票收费")])
        self.assertEqual(self.encoder.calls, [["检索：改签"]])

    def test_build_checks_frozen_files_before_opening_qdrant(self):
        with patch.object(bootstrap, "verify_frozen_dataset", side_effect=ValueError("tampered")):
            with self.assertRaisesRegex(ValueError, "tampered"):
                self.build(rebuild=True)
        self.loader.assert_not_called()
        self.assertFalse(self.db.exists())
