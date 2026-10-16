"""Unit tests for Cost Matrix, Loss Optimization, and Frontier Lift."""

import numpy as np
import pytest
from risklens.cost.loss_matrix import RupeeLossCalculator
from risklens.cost.optimizer import ThresholdOptimizer
from risklens.cost.frontier import ApprovalFrontierEngine


def test_loss_calculator_mechanics():
    calc = RupeeLossCalculator()
    y_true = np.array([1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.2, 0.3, 0.05])
    ead = np.array([100000, 50000, 50000, 100000, 50000])
    lgd = np.array([0.75, 0.75, 0.75, 0.75, 0.75])

    # At threshold 0.25:
    # y_pred = [1, 0, 0, 1, 0] (since prob > 0.25 for index 0 and 3)
    # Decisions match y_true exactly!
    # FN = 0, FP = 0.
    res = calc.calculate_portfolio_loss(y_true, y_prob, threshold=0.25, ead=ead, lgd=lgd)
    assert res["count_fn"] == 0
    assert res["count_fp"] == 0
    assert res["approval_count"] == 3  # approved 3 non-defaults


def test_threshold_optimizer_savings():
    opt = ThresholdOptimizer()
    rng = np.random.default_rng(42)
    n = 2000
    y_true = rng.choice([0, 1], size=n, p=[0.97, 0.03])
    # Predicted probabilities correlate with true label
    y_prob = np.clip(y_true * 0.25 + rng.beta(1.5, 8.0, size=n), 0.001, 0.999)
    ead = rng.uniform(50000, 200000, size=n)
    lgd = np.full(n, 0.75)

    res = opt.optimize_threshold(y_true, y_prob, ead=ead, lgd=lgd)
    assert "optimal_threshold" in res
    assert "total_rupee_savings" in res
    # Cost at optimal threshold should be <= cost at naive 0.5
    assert res["min_expected_loss_rupees"] <= res["naive_loss_rupees"]


def _synthetic_scores(seed=42, n=4000, signal_baseline=1.0, signal_new=2.0):
    """Defaults drawn from a latent risk; two scorers see it with different noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(latent * 1.5 - 3.2)))).astype(int)
    base = 1 / (1 + np.exp(-(signal_baseline * latent + rng.normal(size=n))))
    new = 1 / (1 + np.exp(-(signal_new * latent + rng.normal(size=n) * 0.5)))
    ead = rng.uniform(50000, 150000, size=n)
    return y, base, new, ead


def test_lift_is_measured_not_fixed():
    y, base, new, ead = _synthetic_scores()
    engine = ApprovalFrontierEngine()
    r = engine.compute_approval_expansion(y, base, new, ead=ead, n_bootstrap=50)
    # Respects the matched-risk constraint it searched under.
    assert r["risklens_expansion"]["approved_default_rate_pct"] <= r["baseline"]["approved_default_rate_pct"] + 1e-9
    # Better ranking -> strictly more approvals, and the number is not a round input.
    assert r["lift_metrics"]["additional_approved_accounts"] > 0
    assert r["lift_metrics"]["approval_lift_pct_points"] != 4.0
    lo, hi = r["bootstrap"]["lift_pct_points_ci95"]
    assert lo <= hi and r["bootstrap"]["n_resamples"] == 50


def test_same_scorer_gives_no_lift():
    y, base, _, ead = _synthetic_scores()
    r = ApprovalFrontierEngine().compute_approval_expansion(y, base, base, ead=ead, n_bootstrap=20)
    assert abs(r["lift_metrics"]["approval_lift_pct_points"]) < 0.5


def test_worse_scorer_gives_negative_lift():
    y, base, new, ead = _synthetic_scores()
    r = ApprovalFrontierEngine().compute_approval_expansion(y, new, base, ead=ead, n_bootstrap=20)
    assert r["lift_metrics"]["approval_lift_pct_points"] < 0


def test_fn_loss_matching_respects_loss_budget():
    y, base, new, ead = _synthetic_scores()
    lgd = np.full(len(y), 0.75)
    r = ApprovalFrontierEngine().compute_approval_expansion(
        y, base, new, ead=ead, lgd=lgd, match="fn_loss", n_bootstrap=20)
    assert r["risklens_expansion"]["fn_loss_rupees"] <= r["baseline"]["fn_loss_rupees"] + 1e-6


def test_invalid_match_raises():
    y, base, new, _ = _synthetic_scores()
    with pytest.raises(ValueError):
        ApprovalFrontierEngine().compute_approval_expansion(y, base, new, match="vibes")
