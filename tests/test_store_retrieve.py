import numpy as np

from conformal_rag.retrieve import retrieve, rrf_fuse, vector_search


def test_bm25_finds_topical_chunk(store):
    hits = store.bm25("oil pressure idle", k=5)
    assert hits
    top = store.get_chunks([hits[0][0]])[0]
    assert "oil" in top["text"].lower()


def test_bm25_survives_fts_operators(store):
    # raw '"' or 'NEAR(' would raise inside FTS5 MATCH; sanitizer must cope
    assert isinstance(store.bm25('oil AND "pressure" NEAR(idle)', k=3), list)
    assert store.bm25("!!! ???", k=3) == []


def test_embeddings_roundtrip(store):
    ids, mat = store.all_embeddings()
    assert len(ids) == 5
    assert mat.dtype == np.float32
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_vector_search_ranks_by_similarity(store, embedder):
    hits = vector_search(store, embedder, "generator battery fuel lines", k=3)
    top = store.get_chunks([hits[0][0]])[0]
    assert top["doc"] == "tm-generator.pdf"


def test_rrf_rewards_agreement():
    fused = rrf_fuse({"a": [(1, 9.0), (2, 5.0)], "b": [(1, 0.9), (3, 0.5)]}, rrf_k=60)
    assert fused[0][0] == 1  # found by both → top
    assert set(fused[0][2]) == {"a", "b"}


def test_hybrid_retrieve_end_to_end(store, embedder, cfg):
    hits = retrieve(store, embedder, "low oil pressure at idle", k_final=3)
    assert hits
    assert hits[0].doc == "tm-engine.pdf"
    assert hits[0].page == 12
    assert hits[0].score > 0
