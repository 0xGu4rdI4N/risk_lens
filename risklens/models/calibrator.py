"""Probability Calibration module for accurate expected loss computation."""

from typing import Dict, Optional, Union
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import joblib


class ProbabilityCalibrator:
    """Calibrates raw tree or scorecard output probabilities using Isotonic Regression or Platt Scaling.
    
    Tree-based models (LightGBM, XGBoost) trained on imbalanced data with
    scale_pos_weight distort margins and log-odds.
    This calibrator aligns predicted probabilities with true empirical default rates,
    which is essential for mathematically sound Expected Rupee Loss calculations.
    """

    def __init__(self, method: str = "isotonic"):
        """
        Parameters
        ----------
        method : str, default='isotonic'
            'isotonic' (non-parametric monotonically increasing fit) or
            'sigmoid' (Platt scaling logistic fit).
        """
        self.method = method
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        else:
            self.calibrator = LogisticRegression(solver="lbfgs")
            
        self.base_estimator = None
        self.is_fitted = False

    def fit(self, base_estimator, X: np.ndarray, y: np.ndarray):
        """Fit calibration mapping on hold-out validation set.
        
        Parameters
        ----------
        base_estimator : classifier instance with predict_proba method
        X : features matrix
        y : binary ground truth labels
        """
        self.base_estimator = base_estimator
        
        # Extract uncalibrated probability of positive class
        if hasattr(base_estimator, "predict_proba"):
            raw_p = base_estimator.predict_proba(X)
            if raw_p.ndim == 2 and raw_p.shape[1] == 2:
                raw_p = raw_p[:, 1]
        else:
            raw_p = base_estimator.predict(X)
            
        raw_p = np.asarray(raw_p, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        
        if self.method == "isotonic":
            self.calibrator.fit(raw_p, y)
        else:
            self.calibrator.fit(raw_p.reshape(-1, 1), y)
            
        self.is_fitted = True
        return self

    def predict_proba_from_raw(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration transformation directly to raw uncalibrated probabilities."""
        if not self.is_fitted:
            raise RuntimeError("ProbabilityCalibrator is not fitted yet.")
            
        raw_p = np.asarray(raw_probs, dtype=float).ravel()
        if self.method == "isotonic":
            calib = self.calibrator.predict(raw_p)
            return np.clip(calib, 0.0, 1.0)
        else:
            return self.calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Transform feature matrix X into calibrated default probabilities."""
        if not self.is_fitted:
            raise RuntimeError("ProbabilityCalibrator is not fitted yet.")
            
        if self.base_estimator is None:
            raise RuntimeError("No base estimator registered.")
            
        if hasattr(self.base_estimator, "predict_proba"):
            raw_p = self.base_estimator.predict_proba(X)
            if raw_p.ndim == 2 and raw_p.shape[1] == 2:
                raw_p = raw_p[:, 1]
        else:
            raw_p = self.base_estimator.predict(X)
            
        return self.predict_proba_from_raw(raw_p)

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict:
        """Compute Brier Score, Expected Calibration Error (ECE), and reliability curve."""
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        
        brier = float(brier_score_loss(y_true, y_prob))
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        
        # Expected Calibration Error (ECE)
        bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        # Ensure strictly increasing
        bin_edges[0] = -1e-5
        bin_edges[-1] = 1.0 + 1e-5
        
        ece = 0.0
        n_total = len(y_true)
        
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                bin_weight = np.sum(mask) / n_total
                ece += bin_weight * np.abs(bin_acc - bin_conf)
                
        return {
            "brier_score": brier,
            "ece": float(ece),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ProbabilityCalibrator":
        return joblib.load(filepath)
