"""Does more calibration data actually buy anything, and for which score?

`calibration_size.py` asked this once, on `support_v1`, from a single fixed
split, and answered no: 20, 52 and 72 hand-written questions all produced the
identical threshold 0.50. That was not a property of the sample size. It was the
score being quantised onto three values, so extra points landed on ties the
threshold search could not separate.

The logprob score is near-continuous (109 distinct values on this set), so the
question is live again. This
time hold the *evaluation* fixed and vary only the calibration set, over many
random draws rather than one:

    fixed test size, fixed validation size, n_cal swept
    for each n_cal, many repeats -> distribution, not a point

The nested protocol from `nested_gate.py` is preserved exactly: cal fits the
threshold, val picks the safety margin, test is scored once.

    python scripts/calibration_curve.py --alpha 0.2 --repeats 300
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
MARGINS = [0.20, 0.175, 0.15, 0.125, 0.10, 0.075, 0.05]


def one_trial(rows, key, alpha, n_cal, n_val, n_test, rng):
    """One draw.

    Test and validation sizes are held FIXED and any remaining questions are
    discarded, so the only thing varying across the sweep is n_cal. Letting the
    test set absorb the remainder instead (n_test = N - n_cal - n_val) makes the
    test set shrink as calibration grows, and the resulting fall in "met alpha"
    is mostly the test estimate getting noisier. That confound produced an
    apparent "more data hurts" result on the first run of this script.
    """
    if n_cal + n_val + n_test > len(rows):
        return None
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    cal = [rows[i] for i in idx[:n_cal]]
    val = [rows[i] for i in idx[n_cal:n_cal + n_val]]
    test = [rows[i] for i in idx[n_cal + n_val:n_cal + n_val + n_test]]

    def arr(part):
        return (np.array([r[key] for r in part], float),
                np.array([r["loss"] for r in part], float))

    cs, cl = arr(cal)
    vs, vl = arr(val)
    ts, tl = arr(test)

    best, best_cov = None, -1.0
    for m in MARGINS:
        thr = calibrate_threshold(cs, cl, m)
        h = selective_risk(vs, vl, thr)
        if h["n_answered"] > 0 and h["risk"] <= alpha and h["answer_rate"] > best_cov:
            best, best_cov = m, h["answer_rate"]
    if best is None:
        return None

    thr = calibrate_threshold(cs, cl, best)
    h = selective_risk(ts, tl, thr)
    if h["n_answered"] == 0:
        return None
    return {"threshold": thr, "risk": h["risk"],
            "coverage": h["answer_rate"], "met": h["risk"] <= alpha}


def sweep(rows, key, alpha, sizes, n_val, n_test, repeats, seed):
    out = {}
    for n_cal in sizes:
        rng = random.Random(seed)          # same draws across scores and sizes
        trials = [t for t in (one_trial(rows, key, alpha, n_cal, n_val, n_test, rng)
                              for _ in range(repeats)) if t]
        if not trials:
            out[n_cal] = None
            continue
        cov = np.array([t["coverage"] for t in trials])
        risk = np.array([t["risk"] for t in trials])
        thr = np.array([t["threshold"] for t in trials])
        out[n_cal] = {
            "n_trials": len(trials),
            "met_rate": float(np.mean([t["met"] for t in trials])),
            # Binomial standard error on the hold rate, so a trend across the
            # sweep can be told apart from draw noise.
            "met_se": float(np.sqrt(np.mean([t["met"] for t in trials])
                                    * (1 - np.mean([t["met"] for t in trials]))
                                    / len(trials))),
            "coverage_mean": float(cov.mean()),
            "coverage_p25": float(np.percentile(cov, 25)),
            "coverage_p75": float(np.percentile(cov, 75)),
            "risk_mean": float(risk.mean()),
            # How much the fitted threshold itself moves. If extra calibration
            # data is doing anything, this is what should shrink.
            "threshold_sd": float(thr.std()),
            "distinct_thresholds": int(len(set(np.round(thr, 4)))),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path,
                    default=ROOT / "runs" / "logprob_scores.json")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 30, 40, 50, 62])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "runs" / "calibration_curve.json")
    a = ap.parse_args(argv)

    rows = json.loads(a.scores.read_text())
    print(f"{len(rows)} questions   alpha = {a.alpha}   "
          f"val = {a.n_val} held fixed   {a.repeats} draws per point\n")

    results = {}
    for key, label in [("support_v1", "support_v1"), ("logprob_score", "logprob")]:
        vals = sorted({r[key] for r in rows})
        print(f"  {label}  ({len(vals)} distinct score values)")
        print(f"    {'n_cal':>6}{'n_test':>8}{'met a':>8}{'coverage':>20}"
              f"{'risk':>8}{'thr sd':>9}")
        results[key] = sweep(rows, key, a.alpha, a.sizes, a.n_val, a.n_test,
                             a.repeats, a.seed)
        for n_cal, s in results[key].items():
            if not s:
                print(f"    {n_cal:>6}   never met")
                continue
            print(f"    {n_cal:>6}{a.n_test:>8}"
                  f"{s['met_rate']:>6.0%}+-{s['met_se']:.0%}"
                  f"{s['coverage_mean']:>12.0%} (IQR {s['coverage_p25']:.0%}"
                  f"-{s['coverage_p75']:.0%}){s['risk_mean']:>8.3f}"
                  f"{s['threshold_sd']:>9.3f}")
        print()

    a.out.write_text(json.dumps(
        {"alpha": a.alpha, "repeats": a.repeats, "n_val": a.n_val, "n_test": a.n_test,
         "n_questions": len(rows), "results": results}, indent=2))
    print(f"  written -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
