"""Unit tests for data generation, schema validation, and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from risklens.data.generator import generate_credit_dataset, generate_train_oot_splits
from risklens.data.preprocessor import CreditRiskPreprocessor, ALL_INPUT_FEATURES
from risklens.data.schema import ApplicantFeatures, ScoringRequest


def test_data_generator_shapes_and_balance():
    df = generate_credit_dataset(n_samples=1000, random_state=42)
    assert len(df) == 1000
    assert "is_default" in df.columns
    assert "cibil_score" in df.columns
    assert "exposure_at_default" in df.columns
    assert "loss_given_default" in df.columns
    
    # Class balance should be realistic (~1% to 5%)
    default_rate = df["is_default"].mean()
    assert 0.005 <= default_rate <= 0.08


def test_generator_macro_drift():
    df_base = generate_credit_dataset(n_samples=2000, random_state=42, macro_drift=False)
    df_drift = generate_credit_dataset(n_samples=2000, random_state=42, macro_drift=True)
    
    # In macro drift, average cibil should be lower and utilization should be higher
    assert df_drift["cibil_score"].mean() < df_base["cibil_score"].mean()
    assert df_drift["revolving_utilization"].mean() > df_base["revolving_utilization"].mean()


def test_schema_validation():
    valid_payload = {
        "annual_income": 850000.0,
        "cibil_score": 750,
        "revolving_utilization": 0.30,
        "debt_to_income": 0.28,
        "delinquencies_2yr": 0,
        "inquiries_last_6m": 1,
        "open_credit_lines": 4,
        "credit_history_months": 60,
        "transaction_velocity_24h": 1.5,
        "device_risk_score": 0.12,
        "merchant_category": "retail",
        "employment_type": "salaried",
        "requested_credit_limit": 150000.0,
    }
    app = ApplicantFeatures(**valid_payload)
    assert app.cibil_score == 750
    assert app.annual_income == 850000.0


def test_preprocessor_fit_transform():
    df = generate_credit_dataset(n_samples=500, random_state=42)
    prep = CreditRiskPreprocessor()
    prep.fit(df)
    
    X = prep.transform(df)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 500
    assert X.shape[1] == len(prep.get_feature_names())
    assert not np.isnan(X).any()
