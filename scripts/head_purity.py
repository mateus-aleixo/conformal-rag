"""Search combination rules by the thing the gate actually needs: head purity.

M5 picked a score by AUC and got one that cannot meet alpha = 0.2. This scores
candidate rules by the right objective instead:

    maximise coverage, subject to held-out selective risk <= alpha

Runs entirely on the cached bakeoff signals, so it costs no model calls and many
rules can be compared honestly.

**Guarding against picking a winner by luck.** Comparing ~a dozen rules on the
same test half and reporting the best would be selection bias — with enough
rules one wins by noise. So the rule is chosen on the **calibration** half only,
and its **test** number is reported as the honest estimate. The full table is
printed too, with the caveat attached, because hiding it would be worse.

    python scripts/head_purity.py --alpha 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from conformal_rag.conformal import calibrate_threshold, selective_risk

ROOT = Path(__file__).parent.parent
BASE = ["support_v1", "support_v2", "groundedness", "self_consistency"]


def rules():
    """Candidate combination rules, each mapping a row to a score in [0, 1]."""
    def g(k):
        return lambda r: float(r[k])

    out = {k: g(k) for k in BASE}
    # geometric mean — what M5 used; best AUC, contaminated head
    out["geo(gr,sc)"] = lambda r: float(np.sqrt(r["groundedness"] * r["self_consistency"]))
    out["mean(su,gr)"] = lambda r: (r["support_v1"] + r["groundedness"]) / 2
    # conjunctions: an answer must clear EVERY check
    out["min(su,gr)"] = lambda r: min(r["support_v1"], r["groundedness"])
    out["min(su,sc)"] = lambda r: min(r["support_v1"], r["self_consistency"])
    out["min(gr,sc)"] = lambda r: min(r["groundedness"], r["self_consistency"])
    out["min(su,gr,sc)"] = lambda r: min(r["support_v1"], r["groundedness"], r["self_consistency"])
    out["min(su,gr,v2)"] = lambda r: min(r["support_v1"], r["groundedness"], r["support_v2"])
    # conjunction plus the free deterministic veto
    out["min(su,gr)+veto"] = lambda r: 0.0 if r.get("refused") else min(
        r["support_v1"], r["groundedness"])
    out["su+veto"] = lambda r: 0.0 if r.get("refused") else float(r["support_v1"])
    return out


def head_purity(scores, losses, keep=0.30):
    """Risk inside the top `keep` fraction — the head, measured directly."""
    n = max(1, int(len(scores) * keep))
    top = np.argsort(-scores)[:n]
    return float(losses[top].mean())


def evaluate(rows, fn, alpha, seed=7):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    h = len(idx) // 2
    cal, test = [rows[i] for i in idx[:h]], [rows[i] for i in idx[h:]]
    cs = np.array([fn(r) for r in cal], float)
    cl = np.array([r["loss"] for r in cal], float)
    ts = np.array([fn(r) for r in test], float)
    tl = np.array([r["loss"] for r in test], float)
    thr = calibrate_threshold(cs, cl, alpha)
    cal_h = selective_risk(cs, cl, thr)
    test_h = selective_risk(ts, tl, thr)
    return {
        "threshold": thr,
        "cal_risk": cal_h["risk"], "cal_cov": cal_h["answer_rate"],
        "cal_met": cal_h["risk"] <= alpha and cal_h["n_answered"] > 0,
        "risk": test_h["risk"], "cov": test_h["answer_rate"],
        "met": test_h["risk"] <= alpha and test_h["n_answered"] > 0,
        "head30": head_purity(ts, tl),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff", type=Path, default=ROOT / "runs" / "bakeoff_14b.json")
    ap.add_argument("--scores", type=Path, default=ROOT / "runs" / "gate_scores_14b.json")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "head_purity.json")
    a = ap.parse_args(argv)

    rows = json.loads(a.bakeoff.read_text())
    # the refusal flag lives in the scoring run, not the bakeoff
    refused = {r["id"]: r.get("refused_by_prompt", False)
               for r in json.loads(a.scores.read_text())}
    for r in rows:
        r["refused"] = refused.get(r["id"], False)

    results = {name: evaluate(rows, fn, a.alpha) for name, fn in rules().items()}

    print(f"alpha = {a.alpha}   |   {len(rows)} questions, 50/50 calibration/test split\n")
    print(f"  {'rule':<18} {'thr':>5} {'CAL cov':>8} {'met':>4}  |"
          f" {'TEST risk':>9} {'TEST cov':>9} {'met':>4} {'head30':>7}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["cal_cov"] * kv[1]["cal_met"]):
        print(f"  {name:<18} {r['threshold']:>5.2f} {r['cal_cov']:>7.0%} "
              f"{'yes' if r['cal_met'] else 'no':>4}  | {r['risk']:>9.3f} {r['cov']:>8.0%} "
              f"{'yes' if r['met'] else 'no':>4} {r['head30']:>7.3f}")

    # honest pick: chosen on calibration only
    eligible = {k: v for k, v in results.items() if v["cal_met"]}
    if eligible:
        pick = max(eligible, key=lambda k: eligible[k]["cal_cov"])
        p = results[pick]
        print(f"\n  selected on the CALIBRATION half only: {pick}")
        print(f"  its held-out numbers: risk {p['risk']:.3f}  coverage {p['cov']:.0%}"
              f"  ({'meets' if p['met'] else 'MISSES'} alpha = {a.alpha})")
        print("\n  The table above is shown in full, but picking its best TEST row would be\n"
              "  selection bias across a dozen rules. The line above is the honest estimate.")
    else:
        pick = None
        print("\n  no rule met alpha on the calibration half")

    a.out.write_text(json.dumps({"alpha": a.alpha, "selected": pick,
                                 "results": results}, indent=2))
    print(f"\n  written -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
