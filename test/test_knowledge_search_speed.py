"""Knowledge search speed fixes keep their answers.

Three changes cut a knowledge search on a 22k-item, 59k-entity library from
~0.9 s to ~0.2 s: the alias fallback in ``find_entity`` skips rows that have no
aliases, two indexes serve the graph leg, and the vector leg reuses one cached
matrix until the database files change. These tests pin that each change
returns what the code returned before it.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from kiro_crew.knowledge import retrieval
from kiro_crew.knowledge.embedder import floats_to_bytes
from kiro_crew.knowledge.retrieval import ANY_EMBEDDING_SPACE, HybridRetriever
from kiro_crew.knowledge.store import KnowledgeStore

pytest.importorskip("numpy")

_DIM = 8


@pytest.fixture()
def store(tmp_path: Path):
    retrieval.clear_vector_matrix_cache()
    s = KnowledgeStore(str(tmp_path / "kb.db"))
    yield s
    s.close()
    retrieval.clear_vector_matrix_cache()


def _vec(rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(_DIM)]


def _retriever(store: KnowledgeStore, query_vec: list[float]) -> HybridRetriever:
    return HybridRetriever(store, embedder=lambda _q: query_vec, embed_sig=ANY_EMBEDDING_SPACE)


class TestFindEntity:
    def test_alias_still_resolves_beside_alias_less_rows(self, store):
        for i in range(5):
            store.add_entity(f"plain {i}", "concept")
        eid = store.add_entity("Kubernetes", "tool", aliases=["k8s"])
        assert store.find_entity("K8S")["id"] == eid

    def test_name_match_is_case_insensitive(self, store):
        eid = store.add_entity("FastAPI", "tool")
        assert store.find_entity("fastapi")["id"] == eid

    def test_miss_returns_none(self, store):
        store.add_entity("FastAPI", "tool")
        assert store.find_entity("django") is None

    def test_graph_indexes_exist(self, store):
        names = {
            r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert {"idx_entities_name_lower", "idx_mentions_entity_item"} <= names


class TestGraphTieOrder:
    def test_equal_counts_rank_by_item_id(self, store):
        eid = store.add_entity("widget", "concept")
        items = [store.add_item(f"doc {i}", "c", "doc") for i in range(4)]
        for item in items:
            store.add_mention(item, eid)
        ranked = HybridRetriever(store)._graph_search("widget")
        assert [i for i, _ in ranked] == sorted(items)


class TestVectorMatrixCache:
    def test_cached_ranking_matches_reference_scorer(self, store):
        rng = random.Random(7)
        for i in range(40):
            store.add_item(f"doc {i}", "c", "doc", embedding=floats_to_bytes(_vec(rng)))
        q = _vec(rng)
        r = _retriever(store, q)
        first = r._vector_search("q", limit=40)
        assert len(retrieval._matrix_cache) == 1
        second = r._vector_search("q", limit=40)
        rows = store.db.execute(
            "SELECT id, embedding FROM items WHERE embedding IS NOT NULL AND status = 'active'"
        ).fetchall()
        scored, _ = HybridRetriever._score_rows_python(rows, q)
        scored.sort(key=lambda x: x[1], reverse=True)
        reference = [(i, n + 1) for n, (i, _) in enumerate(scored[:40])]
        assert first == second == reference

    def test_write_invalidates_the_cache(self, store):
        rng = random.Random(3)
        for i in range(10):
            store.add_item(f"doc {i}", "c", "doc", embedding=floats_to_bytes(_vec(rng)))
        q = _vec(rng)
        r = _retriever(store, q)
        r._vector_search("q")
        new_id = store.add_item("exact", "c", "doc", embedding=floats_to_bytes(q))
        assert r._vector_search("q")[0][0] == new_id

    def test_a_wal_reset_changes_the_fingerprint_beyond_the_stat(self, store):
        # After a checkpoint, the next write rewrites the WAL from its start, so
        # its size can repeat and a coarse mtime can too; the header salts cannot.
        rng = random.Random(4)
        store.add_item("a", "c", "doc", embedding=floats_to_bytes(_vec(rng)))
        before = retrieval._db_fingerprint(store._db_path)
        assert before is not None and before[2] is not None
        store.db.execute("PRAGMA wal_checkpoint(RESTART)").fetchall()
        store.add_item("b", "c", "doc", embedding=floats_to_bytes(_vec(rng)))
        after = retrieval._db_fingerprint(store._db_path)
        assert after is not None and after[2] is not None
        assert after[2] != before[2]

    def test_mismatched_width_uses_reference_path(self, store):
        rng = random.Random(5)
        q = _vec(rng)
        store.add_item("ok", "c", "doc", embedding=floats_to_bytes(q))
        store.add_item("short", "c", "doc", embedding=floats_to_bytes([1.0, 2.0, 3.0, 4.0]))
        r = _retriever(store, q)
        assert len(r._vector_search("q")) == 1
        assert retrieval._matrix_cache == {}

    def test_in_memory_store_is_not_cached(self):
        retrieval.clear_vector_matrix_cache()
        s = KnowledgeStore(":memory:")
        try:
            rng = random.Random(1)
            s.add_item("a", "c", "doc", embedding=floats_to_bytes(_vec(rng)))
            assert _retriever(s, _vec(rng))._vector_search("q")
            assert retrieval._matrix_cache == {}
        finally:
            s.close()
