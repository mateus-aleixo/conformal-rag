import numpy as np
import pytest

from conformal_rag.conformal import (
    ConformalGate,
    calibrate_threshold,
    selective_risk,
)


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores in [0,1]; P(wrong) decreases with score — monotone, as assumed."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, n)
    losses = (rng.uniform(0, 1, n) < (1 - scores) * 0.6).astype(float)
    return scores, losses


def test_threshold_meets_risk_in_expectation():
    """The guarantee is E[risk among answered] ≤ α over the calibration draw.

    Individual splits fluctuate around the boundary by design — a per-split
    assertion would be testing noise. So: the MEAN held-out risk across many
    independent calibration/test splits must respect α (small numerical slack),
    and gross per-split violations must be rare."""
    alpha = 0.10
    trials = 200
    risks = []
    gross = 0
    for seed in range(trials):
        cal_s, cal_l = _synthetic(400, seed)
        thr = calibrate_threshold(cal_s, cal_l, alpha)
        test_s, test_l = _synthetic(2000, 10_000 + seed)
        held = selective_risk(test_s, test_l, thr)
        if held["n_answered"] >= 30:
            risks.append(held["risk"])
            if held["risk"] > alpha + 0.05:
                gross += 1
    assert len(risks) > trials // 2  # the gate actually answers things
    assert float(np.mean(risks)) <= alpha + 0.005
    assert gross / len(risks) < 0.05


def test_degenerate_all_wrong_abstains_everything():
    scores = np.linspace(0, 1, 50)
    losses = np.ones(50)
    assert calibrate_threshold(scores, losses, alpha=0.10) == 1.0


def test_all_correct_answers_everything():
    scores = np.linspace(0, 1, 200)
    losses = np.zeros(200)
    assert calibrate_threshold(scores, losses, alpha=0.10) == 0.0


def test_empty_and_invalid_inputs():
    assert calibrate_threshold(np.array([]), np.array([]), 0.1) == 1.0
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([0.5]), np.array([2.0]), 0.1)
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([[0.5]]), np.array([[0.0]]), 0.1)


def test_mondrian_fallback_below_min_group():
    scores, losses = _synthetic(300, 7)
    groups = ["big"] * 290 + ["tiny"] * 10
    gate = ConformalGate(alpha=0.1, min_group=30).fit(scores, losses, groups)
    assert "big" in gate.group_thresholds
    assert "tiny" not in gate.group_thresholds
    d = gate.decide(0.5, "tiny")
    assert d.threshold == gate.global_threshold


def test_gate_serialisation_roundtrip():
    scores, losses = _synthetic(200, 3)
    gate = ConformalGate(alpha=0.1).fit(scores, losses, ["a"] * 100 + ["b"] * 100)
    clone = ConformalGate.from_dict(gate.to_dict())
    assert clone.global_threshold == gate.global_threshold
    assert clone.group_thresholds == gate.group_thresholds


def test_gate_is_invariant_to_monotone_rescaling():
    """Only the ordering of the score matters, never its scale.

    `calibrate_threshold` draws its candidates from the observed scores and does
    no interpolation, so any strictly increasing transform maps each candidate to
    its image and selects the identical set of answered items. This is why
    rescaling the saturated logprob score into logit space (median 0.9998, p75
    upward reading 1.0000) cannot improve the gate: it is a different number for
    the same decision. Recorded in docs/results.md.
    """
    scores, losses = _synthetic(200, seed=7)

    def logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    thr_raw = calibrate_threshold(scores, losses, 0.2)
    thr_logit = calibrate_threshold(logit(scores), losses, 0.2)

    answered_raw = scores >= thr_raw
    answered_logit = logit(scores) >= thr_logit

    np.testing.assert_array_equal(answered_raw, answered_logit)
    assert selective_risk(scores, losses, thr_raw)["risk"] == pytest.approx(
        selective_risk(logit(scores), losses, thr_logit)["risk"]
    )
