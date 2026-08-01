"""Score every question with the logprob support score, then calibrate.

One short call per question (max_tokens=1), so this is far cheaper than any
earlier scoring pass. Losses are reused from the runs already judged, which
keeps the comparison honest — only the score changes.

    python scripts/score_logprob.py --model qwen2.5:14b-instruct
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from conformal_rag.config import DEFAULT
from conformal_rag.conformal import calibrate_threshold, selective_risk
from conformal_rag.embed import get_embedder
from conformal_rag.llm import OpenAICompatClient
from conformal_rag.retrieve import retrieve
from conformal_rag.scores import support_score_logprob
from conformal_rag.store import Store

ROOT = Path(__file__).parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--base", default="http://localhost:11434/v1")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "logprob_scores.json")
    a = ap.parse_args(argv)

    # Losses come from runs already judged by the same 14B, one cache per batch
    # of questions. Only the score changes here, which is what keeps the
    # comparison against support_v1 honest. A missing cache is skipped rather
    # than fatal, so this still runs before a new batch has been judged.
    prior = {}
    for p in ["gate_scores_14b.json", "gate_scores_v2_14b.json",
              "gate_scores_v3_14b.json"]:
        path = ROOT / "runs" / p
        if not path.exists():
            print(f"  (no {p} yet; skipping)")
            continue
        for r in json.loads(path.read_text()):
            prior[r["id"]] = r
    golden = {}
    for p in ["golden.jsonl", "golden_generated.jsonl", "golden_v2.jsonl",
              "golden_v3.jsonl"]:
        for line in (ROOT / "evals" / p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d["id"] in prior:
                    golden[d["id"]] = d

    cfg = replace(DEFAULT, llm_provider="openai", openai_base=a.base,
                  openai_model=a.model, openai_key="ollama")
    store, emb, llm = Store(cfg.db_path), get_embedder("bge"), OpenAICompatClient(cfg)

    out = json.loads(a.out.read_text()) if a.out.exists() else []
    done = {r["id"] for r in out}
    todo = [g for g in golden.values() if g["id"] not in done]
    print(f"scoring {len(todo)} questions ({len(done)} cached) with {a.model}")

    t0 = time.perf_counter()
    for n, g in enumerate(todo, start=1):
        hits = retrieve(store, emb, g["question"], cfg.k_bm25, cfg.k_vec, cfg.k_final, cfg.rrf_k)
        s = support_score_logprob(g["question"], hits, llm)
        p = prior[g["id"]]
        out.append({"id": g["id"], "answerable": g["answerable"], "loss": p["loss"],
                    "verdict": p["verdict"], "logprob_score": s, "support_v1": p["score"]})
        if n % 20 == 0 or n == len(todo):
            a.out.write_text(json.dumps(out, indent=2))
            print(f"  {len(out)}/{len(golden)}  ({time.perf_counter()-t0:.0f}s)")
    a.out.write_text(json.dumps(out, indent=2))

    ls = np.array([r["logprob_score"] for r in out], float)
    vs = np.array([r["support_v1"] for r in out], float)
    ll = np.array([r["loss"] for r in out], float)
    ans = np.array([r["answerable"] for r in out])

    def auc(s, pos):
        p, q = s[pos == 1], s[pos == 0]
        d = p[:, None] - q[None, :]
        return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)

    corr = 1 - ll[ans]
    print(f"\n{'':<16}{'distinct':>9}{'AUC corr':>10}{'ans':>8}{'unans':>8}")
    for name, s in [("support_v1", vs), ("logprob", ls)]:
        print(f"  {name:<14}{len(set(np.round(s,4))):>9}{auc(s[ans], corr):>10.3f}"
              f"{s[ans].mean():>8.3f}{s[~ans].mean():>8.3f}")

    print(f"\n  risk / coverage across thresholds (all {len(out)} questions)")
    print(f"  {'threshold':>10}{'coverage':>10}{'risk':>8}")
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        k = ls >= t
        if k.sum():
            print(f"  {t:>10.2f}{k.mean():>9.0%}{ll[k].mean():>8.3f}")

    idx = list(range(len(out)))
    random.Random(7).shuffle(idx)
    h = len(idx) // 2
    cal, test = [out[i] for i in idx[:h]], [out[i] for i in idx[h:]]
    print(f"\n  gate at alpha = {a.alpha}   (calibration {len(cal)} / test {len(test)})")
    print(f"  {'score':<14}{'thr':>7}{'risk':>8}{'coverage':>10}  met?")
    for name, key in [("support_v1", "support_v1"), ("logprob", "logprob_score")]:
        cs = np.array([r[key] for r in cal], float)
        cl = np.array([r["loss"] for r in cal], float)
        ts = np.array([r[key] for r in test], float)
        tl = np.array([r["loss"] for r in test], float)
        thr = calibrate_threshold(cs, cl, a.alpha)
        hh = selective_risk(ts, tl, thr)
        met = hh["risk"] <= a.alpha and hh["n_answered"] > 0
        print(f"  {name:<14}{thr:>7.3f}{hh['risk']:>8.3f}{hh['answer_rate']:>9.0%}"
              f"  {'YES' if met else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
