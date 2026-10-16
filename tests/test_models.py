"""Unit tests for models and probability calibrators."""

import numpy as np
import pytest
from risklens.data.generator import generate_credit_dataset
from risklens.data.preprocessor import CreditRiskPreprocessor
from risklens.models.baseline import LogisticScorecardBaseline
from risklens.models.tree_models import LightGBMRiskClassifier
from risklens.models.calibrator import ProbabilityCalibrator


@pytest.fixture(scope="module")
def processed_data():
    df = generate_credit_dataset(n_samples=1500, random_state=42)
    prep = CreditRiskPreprocessor()
    prep.fit(df)
    X = prep.transform(df)
    y = df["is_default"].values
    return X, y


def test_baseline_scorecard(processed_data):
    X, y = processed_data
    scorecard = LogisticScorecardBaseline()
    scorecard.fit(X[:1000], y[:1000])
    
    probs = scorecard.predict_proba(X[1000:])
    assert len(probs) == 500
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_lightgbm_and_calibration(processed_data):
    X, y = processed_data
    lgb = LightGBMRiskClassifier()
    lgb.fit(X[:1000], y[:1000])
    
    calib = ProbabilityCalibrator(method="isotonic")
    calib.fit(lgb.model, X[800:1000], y[800:1000])
    
    calib_probs = calib.predict_proba(X[1000:])
    assert len(calib_probs) == 500
    assert np.all((calib_probs >= 0.0) & (calib_probs <= 1.0))
    
    metrics = calib.evaluate_calibration(y[1000:], calib_probs)
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert metrics["brier_score"] >= 0.0
