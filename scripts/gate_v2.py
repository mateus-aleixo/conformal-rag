"""Recalibrate the gate on each candidate score, and on a combination.

The bakeoff ranked candidates by AUC over correct-vs-incorrect. This asks the
question that actually matters: with each score as the nonconformity signal, can
conformal risk control bound the **any-mistake** risk that M4 could not?

Also tries `groundedness x self_consistency`, because the bakeoff showed the two
are complementary rather than redundant — one ranks answerability well and
correctness poorly, the other does the reverse.

    python scripts/gate_v2.py --plot docs/figures/gate_v2.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from conformal_rag.conformal import calibrate_threshold, selective_risk

ROOT = Path(__file__).parent.parent
KEYS = ["support_v1", "support_v2", "groundedness", "self_consistency", "combined"]


def add_combined(rows):
    for r in rows:
        # geometric mean: both signals must agree before the gate answers
        r["combined"] = float(np.sqrt(max(r["groundedness"], 0) * max(r["self_consistency"], 0)))
    return rows


def split(rows, seed=7):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    h = len(idx) // 2
    return [rows[i] for i in idx[:h]], [rows[i] for i in idx[h:]]


def auc(scores, positive):
    pos, neg = scores[positive == 1], scores[positive == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff", type=Path, default=ROOT / "runs" / "bakeoff.json")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate_v2.json")
    a = ap.parse_args(argv)

    rows = add_combined(json.loads(a.bakeoff.read_text()))
    cal, test = split(rows)
    tl = np.array([r["loss"] for r in test], dtype=float)
    ungated = float(tl.mean())

    print(f"items {len(rows)} | calibration {len(cal)} / test {len(test)}")
    print(f"ungated any-mistake risk on the test half: {ungated:.3f}\n")

    ans = [r for r in rows if r["answerable"]]
    corr = np.array([1 - r["loss"] for r in ans])
    print("  AUC over correct-vs-incorrect (answerable only)")
    for k in KEYS:
        print(f"    {k:<18} {auc(np.array([r[k] for r in ans], float), corr):.3f}")

    results = {}
    print("\n  can it bound the ANY-MISTAKE risk?")
    print(f"    {'score':<18} {'alpha':>6} {'thr':>6} {'risk':>7} {'answered':>9}  met?")
    for k in KEYS:
        cs = np.array([r[k] for r in cal], float)
        cl = np.array([r["loss"] for r in cal], float)
        ts = np.array([r[k] for r in test], float)
        rowset = []
        for al in a.alphas:
            thr = calibrate_threshold(cs, cl, al)
            h = selective_risk(ts, tl, thr)
            met = h["risk"] <= al and h["n_answered"] > 0
            rowset.append({"alpha": al, "threshold": thr, **h, "met": met})
            print(f"    {k:<18} {al:>6.2f} {thr:>6.2f} {h['risk']:>7.3f}"
                  f" {h['answer_rate']:>8.0%}  {'yes' if met else 'no'}")
        results[k] = rowset
        print()

    a.out.write_text(json.dumps({"ungated": ungated, "results": results}, indent=2))
    print(f"  written -> {a.out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = np.linspace(0, 1, 101)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        colours = {"support_v1": "#b0342c", "groundedness": "#c07a1e",
                   "self_consistency": "#1a7a4a", "combined": "#24548f"}
        for k, c in colours.items():
            ts = np.array([r[k] for r in test], float)
            ax.plot(grid, [selective_risk(ts, tl, t)["risk"] for t in grid],
                    color=c, lw=2, label=k)
        ax.axhline(0.2, color="#888", lw=1, ls=":", label="α = 0.20")
        ax.axhline(ungated, color="#bbb", lw=1, ls="--", label=f"ungated {ungated:.2f}")
        ax.set_xlabel("score threshold")
        ax.set_ylabel("any-mistake risk on the held-out half")
        ax.set_ylim(-0.02, 0.72)
        ax.set_title("Which signal lets the gate bound total error?", fontsize=11)
        ax.legend(fontsize=8.4, frameon=False)
        ax.grid(alpha=0.25)
        a.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120, facecolor="white")
        print(f"  plot -> {a.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
