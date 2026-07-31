"""Why the gate cannot hit alpha on total error, and what it CAN guarantee.

Two losses are calibrated against the same support scores:

  correctness   1 if answering was a mistake for ANY reason - the question was
                unanswerable, or it was answerable and the answer came out wrong
  answerability 1 only if we answered a question the corpus cannot answer

The gap between them is the finding. The support score is a judgement about the
*excerpts*, so it predicts answerability well and correctness barely at all;
conformal risk control can only bound a risk its score can see.

    python scripts/gate_analysis.py --plot docs/figures/risk_curve.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from conformal_rag.conformal import calibrate_threshold, selective_risk

ROOT = Path(__file__).parent.parent


def split(scored, seed=7):
    idx = list(range(len(scored)))
    random.Random(seed).shuffle(idx)
    h = len(idx) // 2
    return [scored[i] for i in idx[:h]], [scored[i] for i in idx[h:]]


def arrays(rows, loss_key):
    return (np.array([r["score"] for r in rows], dtype=float),
            np.array([r[loss_key] for r in rows], dtype=float))


def sweep(cal, test, loss_key, alphas):
    out = []
    cs, cl = arrays(cal, loss_key)
    ts, tl = arrays(test, loss_key)
    for a in alphas:
        thr = calibrate_threshold(cs, cl, a)
        h = selective_risk(ts, tl, thr)
        out.append({"alpha": a, "threshold": thr, "risk": h["risk"],
                    "answer_rate": h["answer_rate"], "n": h["n_answered"],
                    "met": h["risk"] <= a})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "runs" / "gate_scores.json")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate_analysis.json")
    a = ap.parse_args(argv)

    scored = json.loads(a.scores.read_text())
    for r in scored:                       # second loss: only unanswerable-answered
        r["loss_answerability"] = 0.0 if r["answerable"] else 1.0
        r["loss_correctness"] = r["loss"]

    cal, test = split(scored)
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"items {len(scored)} | calibration {len(cal)} / test {len(test)}\n")
    for key, label in [("loss_correctness", "LOSS = any mistake (unanswerable OR wrong answer)"),
                       ("loss_answerability", "LOSS = answered an unanswerable question")]:
        base = np.mean([r[key] for r in test])
        print(f"{label}\n  ungated risk on the test half: {base:.3f}")
        print(f"  {'alpha':>6} {'thr':>6} {'risk':>7} {'answered':>10}   met?")
        for row in sweep(cal, test, key, alphas):
            print(f"  {row['alpha']:>6.2f} {row['threshold']:>6.2f} {row['risk']:>7.3f}"
                  f" {row['answer_rate']:>9.0%}   {'yes' if row['met'] else 'NO'}")
        print()

    ans = [r for r in scored if r["answerable"]]
    ok = [r["score"] for r in ans if r["loss_correctness"] == 0]
    bad = [r["score"] for r in ans if r["loss_correctness"] == 1]
    print("why: the score judges the EXCERPTS, so it sees answerability, not correctness")
    print(f"  answerable {np.mean([r['score'] for r in ans]):.3f} vs "
          f"unanswerable {np.mean([r['score'] for r in scored if not r['answerable']]):.3f}"
          f"   <- separated")
    print(f"  correct    {np.mean(ok):.3f} vs incorrect  {np.mean(bad):.3f}"
          f"   <- not separated")

    a.out.write_text(json.dumps(
        {"correctness": sweep(cal, test, "loss_correctness", alphas),
         "answerability": sweep(cal, test, "loss_answerability", alphas)}, indent=2))
    print(f"\nwrote {a.out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = np.linspace(0, 1, 101)
        ts, _ = arrays(test, "loss_correctness")
        _, tl_c = arrays(test, "loss_correctness")
        _, tl_a = arrays(test, "loss_answerability")
        r_c = [selective_risk(ts, tl_c, t)["risk"] for t in grid]
        r_a = [selective_risk(ts, tl_a, t)["risk"] for t in grid]
        rate = [selective_risk(ts, tl_c, t)["answer_rate"] for t in grid]

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.plot(grid, r_c, color="#b0342c", lw=2.2,
                label="risk: any mistake  (score cannot see this)")
        ax.plot(grid, r_a, color="#1a7a4a", lw=2.2,
                label="risk: answered an unanswerable  (score sees this)")
        ax.plot(grid, rate, color="#24548f", lw=1.5, ls="--", label="answer rate")
        ax.axhline(0.2, color="#888", lw=1, ls=":", label="α = 0.20")
        ax.set_xlabel("support-score threshold")
        ax.set_ylabel("rate on the held-out half")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("A conformal gate can only bound a risk its score can see",
                     fontsize=11)
        ax.legend(fontsize=8.2, frameon=False, loc="center left")
        ax.grid(alpha=0.25)
        a.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=120, facecolor="white")
        print(f"plot -> {a.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
