"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from risklens.api.app import app, runtime_state
from risklens.data.generator import generate_credit_dataset
from risklens.data.preprocessor import CreditRiskPreprocessor
from risklens.models.tree_models import LightGBMRiskClassifier
from risklens.models.calibrator import ProbabilityCalibrator
from risklens.explainability.explainer import RiskLensExplainer

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_mock_runtime():
    if runtime_state["model"] is None:
        df = generate_credit_dataset(n_samples=500, random_state=42)
        prep = CreditRiskPreprocessor()
        prep.fit(df)
        X = prep.transform(df)
        y = df["is_default"].values
        
        lgb = LightGBMRiskClassifier()
        lgb.fit(X, y)
        
        calib = ProbabilityCalibrator()
        calib.fit(lgb.model, X[:250], y[:250])
        
        explainer = RiskLensExplainer(lgb.model, prep.get_feature_names())
        
        runtime_state["preprocessor"] = prep
        runtime_state["model"] = lgb
        runtime_state["calibrator"] = calib
        runtime_state["explainer"] = explainer
        runtime_state["optimal_threshold"] = 0.085


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "artifacts_loaded" in data


def test_cost_config_endpoint():
    response = client.get("/config/costs")
    assert response.status_code == 200
    data = response.json()
    assert "default_fn_cost_rupees" in data


def test_predict_endpoint():
    payload = {
        "features": {
            "annual_income": 950000.0,
            "cibil_score": 780,
            "revolving_utilization": 0.22,
            "debt_to_income": 0.24,
            "delinquencies_2yr": 0,
            "inquiries_last_6m": 0,
            "open_credit_lines": 5,
            "credit_history_months": 84,
            "transaction_velocity_24h": 1.1,
            "device_risk_score": 0.05,
            "merchant_category": "retail",
            "employment_type": "salaried",
            "requested_credit_limit": 300000.0,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"] in ["APPROVED", "DECLINED"]
    assert "calibrated_default_prob" in data
    assert "expected_rupee_loss" in data
    assert "top_risk_factors" in data
