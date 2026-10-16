"""Traditional Credit Scorecard Logistic Regression Baseline."""

from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


class LogisticScorecardBaseline:
    """Traditional linear credit scorecard model used as the benchmark."""

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        self.model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs"
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LogisticScorecardBaseline":
        inst = cls()
        inst.model = joblib.load(filepath)
        inst.is_fitted = True
        return inst
