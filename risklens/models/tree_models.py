"""Gradient Boosted Tree Models for Imbalanced Credit Default and Fraud Risk."""

from typing import Optional
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from risklens.config import DEFAULT_MODEL_CONFIG, ModelConfig


class LightGBMRiskClassifier:
    """LightGBM configured for cost-sensitive imbalanced credit modeling."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or DEFAULT_MODEL_CONFIG
        self.model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            num_leaves=31,
            scale_pos_weight=self.config.scale_pos_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            objective="binary",
            importance_type="gain",
            verbose=-1,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric="average_precision",
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) > threshold).astype(int)

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LightGBMRiskClassifier":
        inst = cls()
        inst.model = joblib.load(filepath)
        inst.is_fitted = True
        return inst


class XGBoostRiskClassifier:
    """XGBoost alternative implementation."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or DEFAULT_MODEL_CONFIG
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            scale_pos_weight=self.config.scale_pos_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            eval_metric="aucpr",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None):
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)
