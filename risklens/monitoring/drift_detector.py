"""Drift detector comparing baseline training distribution against production streams."""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from risklens.config import DEFAULT_DRIFT_CONFIG, DriftConfig
from risklens.monitoring.psi import PSICalculator


class DriftDetector:
    """End-to-end drift audit harness for credit risk models."""

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DEFAULT_DRIFT_CONFIG
        self.psi_calc = PSICalculator(self.config)

    def run_drift_audit(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        baseline_scores: Optional[np.ndarray] = None,
        current_scores: Optional[np.ndarray] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Dict:
        """Execute full feature-level and score-level PSI drift audit."""
        timestamp = datetime.utcnow().isoformat()
        feature_results: List[Dict] = []
        
        # 1. Score PSI (Model Output Drift)
        score_psi_val = 0.0
        score_status = "NOT_EVALUATED"
        if baseline_scores is not None and current_scores is not None:
            s_res = self.psi_calc.calculate_continuous_psi(baseline_scores, current_scores)
            score_psi_val = s_res["psi"]
            score_status = s_res["status"]

        # 2. Continuous Features
        num_cols = numerical_features or [
            c for c in baseline_df.select_dtypes(include=[np.number]).columns
            if c in current_df.columns and c not in ["is_default", "true_risk_prob"]
        ]
        
        for col in num_cols:
            if col in baseline_df.columns and col in current_df.columns:
                res = self.psi_calc.calculate_continuous_psi(
                    baseline_df[col].dropna().values,
                    current_df[col].dropna().values,
                )
                feature_results.append({
                    "feature": col,
                    "type": "numerical",
                    "psi": float(res["psi"]),
                    "status": res["status"],
                })

        # 3. Categorical Features
        cat_cols = categorical_features or [
            c for c in baseline_df.select_dtypes(include=["object", "category"]).columns
            if c in current_df.columns and c not in ["applicant_id", "period"]
        ]
        
        for col in cat_cols:
            if col in baseline_df.columns and col in current_df.columns:
                res = self.psi_calc.calculate_categorical_psi(
                    baseline_df[col],
                    current_df[col],
                )
                feature_results.append({
                    "feature": col,
                    "type": "categorical",
                    "psi": float(res["psi"]),
                    "status": res["status"],
                })

        # Sort features descending by PSI
        feature_results.sort(key=lambda x: x["psi"], reverse=True)
        
        # Determine overall system health
        any_severe = any(r["status"] == "SEVERE_DRIFT" for r in feature_results) or score_status == "SEVERE_DRIFT"
        any_warning = any(r["status"] == "WARNING" for r in feature_results) or score_status == "WARNING"
        
        if any_severe:
            overall_status = "CRITICAL_DRIFT_ALERT"
        elif any_warning:
            overall_status = "MODERATE_DRIFT_WARNING"
        else:
            overall_status = "SYSTEM_HEALTHY_STABLE"

        return {
            "timestamp": timestamp,
            "baseline_count": len(baseline_df),
            "target_count": len(current_df),
            "score_psi": float(score_psi_val),
            "score_status": score_status,
            "overall_status": overall_status,
            "features": feature_results,
            "features_df": pd.DataFrame(feature_results),
        }

    def plot_psi_summary(self, audit_result: Dict, output_path: str):
        """Generate a bar plot of top drifting features."""
        df_feat = audit_result["features_df"].head(10).copy()
        plt.figure(figsize=(9, 5))
        
        color_map = {
            "STABLE": "#10b981",       # Green
            "WARNING": "#f59e0b",      # Amber
            "SEVERE_DRIFT": "#ef4444", # Red
        }
        colors = [color_map.get(s, "#3b82f6") for s in df_feat["status"]]
        
        bars = plt.barh(df_feat["feature"][::-1], df_feat["psi"][::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
        plt.axvline(x=self.config.stable_threshold, color="#f59e0b", linestyle="--", linewidth=1.5, label="Warning (0.10)")
        plt.axvline(x=self.config.warning_threshold, color="#ef4444", linestyle="--", linewidth=1.5, label="Severe (0.20)")
        
        plt.xlabel("Population Stability Index (PSI)", fontsize=11)
        plt.title(f"RiskLens Data Drift Audit - Top Features (Status: {audit_result['overall_status']})", fontsize=12)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
