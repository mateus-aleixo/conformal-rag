"""Hybrid retrieval: FTS5 BM25 + brute-force cosine, fused by reciprocal rank.

RRF (Cormack et al., 2009) needs no score normalisation across heterogeneous
retrievers, which is exactly the BM25-vs-cosine situation. score = Σ 1/(k + rank).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .store import Store


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    doc: str
    page: int
    text: str
    score: float          # fused RRF score — bigger is better
    sources: tuple[str, ...]  # which retrievers found it: ("bm25", "vec")


def vector_search(store: Store, embedder: Embedder, query: str, k: int) -> list[tuple[int, float]]:
    ids, mat = store.all_embeddings()
    if len(ids) == 0:
        return []
    q = embedder.encode([query])[0]
    sims = mat @ q  # both sides L2-normalised → cosine
    top = np.argsort(-sims)[:k]
    return [(int(ids[i]), float(sims[i])) for i in top]


def rrf_fuse(
    ranked_lists: dict[str, list[tuple[int, float]]], rrf_k: int = 60
) -> list[tuple[int, float, tuple[str, ...]]]:
    scores: dict[int, float] = {}
    found_by: dict[int, list[str]] = {}
    for name, ranking in ranked_lists.items():
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            found_by.setdefault(cid, []).append(name)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(cid, s, tuple(found_by[cid])) for cid, s in fused]


def retrieve(
    store: Store,
    embedder: Embedder,
    query: str,
    k_bm25: int = 20,
    k_vec: int = 20,
    k_final: int = 5,
    rrf_k: int = 60,
) -> list[Hit]:
    fused = rrf_fuse(
        {
            "bm25": store.bm25(query, k_bm25),
            "vec": vector_search(store, embedder, query, k_vec),
        },
        rrf_k=rrf_k,
    )[:k_final]
    chunks = {c["id"]: c for c in store.get_chunks([cid for cid, _, _ in fused])}
    return [
        Hit(
            chunk_id=cid,
            doc=chunks[cid]["doc"],
            page=chunks[cid]["page"],
            text=chunks[cid]["text"],
            score=score,
            sources=srcs,
        )
        for cid, score, srcs in fused
        if cid in chunks
    ]
