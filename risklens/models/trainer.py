"""Model benchmarking and evaluation harness."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from risklens.cost.loss_matrix import RupeeLossCalculator
from risklens.cost.optimizer import ThresholdOptimizer


class ModelEvaluator:
    """Computes comprehensive risk metrics: PR-AUC, ROC-AUC, Brier Score, and Rupee Loss."""

    def __init__(self, loss_calc: Optional[RupeeLossCalculator] = None):
        self.loss_calc = loss_calc or RupeeLossCalculator()
        self.optimizer = ThresholdOptimizer(self.loss_calc.config)

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ead: Optional[np.ndarray] = None,
        lgd: Optional[np.ndarray] = None,
    ) -> Dict:
        """Benchmark a model across traditional statistics and commercial rupee losses."""
        # Standard Imbalance & Probability Metrics
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        
        # Loss at default 0.5 threshold
        naive_metrics = self.loss_calc.calculate_portfolio_loss(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.50,
            ead=ead,
            lgd=lgd,
        )
        
        # Cost-Optimal Threshold Search
        opt_res = self.optimizer.optimize_threshold(
            y_true=y_true,
            y_prob=y_prob,
            ead=ead,
            lgd=lgd,
        )
        
        opt_threshold = opt_res["optimal_threshold"]
        opt_metrics = opt_res["optimal_metrics"]
        
        y_pred_opt = (y_prob > opt_threshold).astype(int)
        precision_opt = float(precision_score(y_true, y_pred_opt, zero_division=0))
        recall_opt = float(recall_score(y_true, y_pred_opt, zero_division=0))
        f1_opt = float(f1_score(y_true, y_pred_opt, zero_division=0))

        return {
            "model_name": model_name,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "optimal_threshold": opt_threshold,
            "loss_at_naive_05_rupees": naive_metrics["total_loss_rupees"],
            "loss_at_optimal_threshold_rupees": opt_metrics["total_loss_rupees"],
            "rupee_savings": opt_res["total_rupee_savings"],
            "pct_savings": opt_res["pct_savings"],
            "approval_rate_optimal_pct": opt_metrics["approval_rate"] * 100.0,
            "approval_rate_naive_pct": naive_metrics["approval_rate"] * 100.0,
            "precision_at_optimal": precision_opt,
            "recall_at_optimal": recall_opt,
            "f1_at_optimal": f1_opt,
        }

    def compare_models(
        self,
        models_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        ead: Optional[np.ndarray] = None,
        lgd: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Compare multiple model probability vectors side-by-side."""
        results = []
        for name, probs in models_dict.items():
            res = self.evaluate_model(name, y_true, probs, ead=ead, lgd=lgd)
            results.append(res)
        return pd.DataFrame(results)
