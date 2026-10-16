"""Unit tests for Population Stability Index (PSI) calculation."""

import numpy as np
import pandas as pd
import pytest
from risklens.monitoring.psi import PSICalculator


def test_psi_identical_distribution():
    calc = PSICalculator()
    rng = np.random.default_rng(42)
    dist1 = rng.normal(700, 50, 5000)
    dist2 = rng.normal(700, 50, 5000)

    res = calc.calculate_continuous_psi(dist1, dist2)
    # Identical distributions should have very low PSI (< 0.05) and be STABLE
    assert res["psi"] < 0.05
    assert res["status"] == "STABLE"


def test_psi_drifted_distribution():
    calc = PSICalculator()
    rng = np.random.default_rng(42)
    dist_base = rng.normal(750, 40, 5000)
    # Heavy downward shift (severe drift)
    dist_drift = rng.normal(640, 60, 5000)

    res = calc.calculate_continuous_psi(dist_base, dist_drift)
    assert res["psi"] > 0.20
    assert res["status"] == "SEVERE_DRIFT"


def test_categorical_psi():
    calc = PSICalculator()
    s1 = pd.Series(["salaried"] * 700 + ["self_employed"] * 200 + ["gig_economy"] * 100)
    s2 = pd.Series(["salaried"] * 300 + ["self_employed"] * 300 + ["gig_economy"] * 400)

    res = calc.calculate_categorical_psi(s1, s2)
    assert res["psi"] > 0.10
