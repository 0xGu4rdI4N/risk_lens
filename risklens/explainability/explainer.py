"""SHAP Explainability Engine using TreeExplainer."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib


class RiskLensExplainer:
    """Computes global feature importance and local applicant-level SHAP attributions."""

    def __init__(self, model, feature_names: List[str]):
        """
        Parameters
        ----------
        model : fitted tree estimator (e.g. LGBMClassifier.booster_ or xgb.Booster)
        feature_names : list of feature strings matching model input
        """
        # If wrapped, extract underlying booster
        if hasattr(model, "booster_"):
            self.booster = model.booster_
        elif hasattr(model, "get_booster"):
            self.booster = model.get_booster()
        else:
            self.booster = model
            
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.booster)

    def explain_instance(
        self,
        X_single: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """Compute local SHAP attribution for a single applicant record."""
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
            
        shap_values = self.explainer.shap_values(X_single)
        
        # Handle binary classification output formats in shap
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            vals = shap_values[0]
        else:
            vals = np.array(shap_values).flatten()

        feature_impacts = []
        for name, val, f_val in zip(self.feature_names, vals, X_single[0]):
            feature_impacts.append({
                "feature": name,
                "shap_value": float(val),
                "feature_value": float(f_val),
                "direction": "RISK_INCREASING" if val > 0 else "RISK_DECREASING"
            })
            
        # Sort descending by absolute SHAP impact
        feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return feature_impacts[:top_k]

    def get_global_importance(self, X_sample: np.ndarray) -> pd.DataFrame:
        """Compute mean absolute SHAP value across a representative sample."""
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = np.array(shap_values)
            
        mean_abs = np.mean(np.abs(vals), axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        return df

    def save_summary_plot(self, X_sample: np.ndarray, output_path: str):
        """Generate and save global beeswarm / bar plot."""
        shap_values = self.explainer(X_sample)
        plt.figure(figsize=(10, 6))
        # If 3D, take positive class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        shap_values.feature_names = self.feature_names
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.title("RiskLens Global SHAP Feature Attributions (Default/Fraud Risk)", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "RiskLensExplainer":
        return joblib.load(filepath)
