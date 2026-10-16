"""Unit tests for SHAP explainer and Adverse Action Notice generation."""

import numpy as np
import pytest
from risklens.data.generator import generate_credit_dataset
from risklens.data.preprocessor import CreditRiskPreprocessor
from risklens.models.tree_models import LightGBMRiskClassifier
from risklens.explainability.explainer import RiskLensExplainer
from risklens.explainability.adverse_action import AdverseActionGenerator


@pytest.fixture(scope="module")
def trained_explainer():
    df = generate_credit_dataset(n_samples=600, random_state=42)
    prep = CreditRiskPreprocessor()
    prep.fit(df)
    X = prep.transform(df)
    y = df["is_default"].values

    lgb = LightGBMRiskClassifier()
    lgb.fit(X, y)

    explainer = RiskLensExplainer(lgb.model, prep.get_feature_names())
    return explainer, X


def test_shap_instance_explanation(trained_explainer):
    explainer, X = trained_explainer
    factors = explainer.explain_instance(X[0], top_k=4)
    assert len(factors) == 4
    for f in factors:
        assert "feature" in f
        assert "shap_value" in f
        assert "direction" in f


def test_adverse_action_generator():
    mock_factors = [
        {"feature": "revolving_utilization", "shap_value": 0.85, "direction": "RISK_INCREASING"},
        {"feature": "delinquencies_2yr", "shap_value": 0.55, "direction": "RISK_INCREASING"},
        {"feature": "cibil_score", "shap_value": -0.40, "direction": "RISK_DECREASING"},
    ]
    reasons = AdverseActionGenerator.generate_reasons(mock_factors, max_reasons=2)
    assert len(reasons) == 2
    assert "AAR-02" in reasons[0]  # Revolving utilization
    assert "AAR-04" in reasons[1]  # Delinquency
