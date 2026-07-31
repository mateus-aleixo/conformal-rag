"""Re-score a run's answers with one fixed judge, so generators can be compared.

Comparing generators is only meaningful when the grader is held constant. The
7 B experiment showed why: the 7 B judge turned out *stricter* than the 3 B one,
so judging each model with itself understated the bigger model's gain.

This takes a scored run, replays its answers past a chosen judge, and writes a
new file with the same shape. Answers are reused when the run stored them, and
regenerated with the original generator when it did not (the earliest runs
predate answer storage).

    python scripts/rejudge.py --scores runs/gate_scores_7b.json \
        --judge-model qwen2.5:14b-instruct --out runs/gate_scores_7b_j14.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from conformal_rag.answer import answer
from conformal_rag.config import DEFAULT
from conformal_rag.embed import get_embedder
from conformal_rag.judge import judge_answer, loss_from_verdict
from conformal_rag.llm import get_llm
from conformal_rag.store import Store

ROOT = Path(__file__).parent.parent


def load_golden() -> dict[str, dict]:
    g = {}
    for p in ["golden.jsonl", "golden_generated.jsonl"]:
        for line in (ROOT / "evals" / p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                g[d["id"]] = d
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--gen-model", default=None,
                    help="only needed when the run did not store its answers")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = json.loads(a.scores.read_text())
    golden = load_golden()
    jud = get_llm(replace(DEFAULT, llm_provider="ollama", ollama_model=a.judge_model))

    need_gen = any(r["answerable"] and not r.get("answer") for r in rows)
    gen = store = emb = None
    if need_gen:
        if not a.gen_model:
            sys.exit("this run has no stored answers; pass --gen-model to regenerate them")
        print(f"regenerating missing answers with {a.gen_model}")
        gcfg = replace(DEFAULT, llm_provider="ollama", ollama_model=a.gen_model)
        gen, store, emb = get_llm(gcfg), Store(gcfg.db_path), get_embedder("bge")

    out = []
    if a.out.exists():                     # resume; these runs are expensive
        out = json.loads(a.out.read_text())
        print(f"resuming from {len(out)} already re-judged")
    done = {r["id"] for r in out}

    for n, r in enumerate([x for x in rows if x["id"] not in done], start=1):
        if not r["answerable"]:
            out.append({**r, "verdict": "UNANSWERABLE", "loss": 1.0,
                        "judge": a.judge_model})
        else:
            text = r.get("answer")
            if not text:
                gcfg = replace(DEFAULT, llm_provider="ollama", ollama_model=a.gen_model)
                text = answer(golden[r["id"]]["question"], store, emb, gen, gcfg).text or ""
            v = judge_answer(golden[r["id"]]["question"], text,
                             golden[r["id"]].get("note", ""), jud)
            out.append({**r, "answer": text[:1200], "verdict": v,
                        "loss": loss_from_verdict(v), "judge": a.judge_model})
        if n % 10 == 0:
            a.out.write_text(json.dumps(out, indent=2))
            print(f"  {len(out)}/{len(rows)}")

    a.out.write_text(json.dumps(out, indent=2))
    ans = [r for r in out if r["answerable"]]
    err = sum(r["loss"] for r in ans) / len(ans)
    print(f"\n{a.scores.name} judged by {a.judge_model}: base error {err:.3f}"
          f"  ({len(ans)} answerable)\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
