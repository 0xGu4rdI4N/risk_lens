"""Data preprocessing pipeline for credit risk tabular features."""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib


NUMERICAL_FEATURES = [
    "annual_income",
    "cibil_score",
    "revolving_utilization",
    "debt_to_income",
    "delinquencies_2yr",
    "inquiries_last_6m",
    "open_credit_lines",
    "credit_history_months",
    "transaction_velocity_24h",
    "device_risk_score",
    "requested_credit_limit",
]

CATEGORICAL_FEATURES = [
    "merchant_category",
    "employment_type",
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """Production transformer pipeline preserving feature names and schema integrity."""
    
    def __init__(self):
        self.num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])
        
        self.cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", self.num_pipeline, NUMERICAL_FEATURES),
                ("cat", self.cat_pipeline, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        self.feature_names_: List[str] = []
        
    def fit(self, X: pd.DataFrame, y=None):
        self.preprocessor.fit(X[ALL_INPUT_FEATURES])
        
        # Extract transformed feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names_ = NUMERICAL_FEATURES + cat_features
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # If passed a dict or missing columns, align safely
        X_df = pd.DataFrame(X).copy()
        for col in ALL_INPUT_FEATURES:
            if col not in X_df.columns:
                X_df[col] = np.nan
        return self.preprocessor.transform(X_df[ALL_INPUT_FEATURES])
        
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)
        
    def get_feature_names(self) -> List[str]:
        return self.feature_names_
        
    def save(self, filepath: str):
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> "CreditRiskPreprocessor":
        return joblib.load(filepath)
