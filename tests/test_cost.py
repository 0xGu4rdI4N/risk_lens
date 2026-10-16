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


def test_approval_expansion_lift():
    engine = ApprovalFrontierEngine()
    rng = np.random.default_rng(42)
    n = 2000
    y_true = rng.choice([0, 1], size=n, p=[0.97, 0.03])
    base_prob = rng.uniform(0.01, 0.5, size=n)
    rl_prob = np.clip(y_true * 0.3 + rng.beta(1.2, 9.0, size=n), 0.001, 0.999)
    ead = rng.uniform(50000, 150000, size=n)

    lift = engine.compute_approval_expansion(
        y_true=y_true,
        baseline_prob=base_prob,
        risklens_prob=rl_prob,
        ead=ead,
        baseline_approval_rate=0.78,
        target_lift_pct=4.0,
    )
    assert lift["lift_metrics"]["approval_lift_pct_points"] == 4.0
    assert lift["risklens_expansion"]["approval_rate_pct"] == 82.0
    assert lift["lift_metrics"]["additional_approved_accounts"] > 0
