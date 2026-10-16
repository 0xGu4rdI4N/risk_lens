"""Cost-Sensitive Threshold Optimization Engine."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from risklens.config import DEFAULT_COST_CONFIG, CostMatrixConfig
from risklens.cost.loss_matrix import RupeeLossCalculator


class ThresholdOptimizer:
    """Finds the decision threshold minimizing total portfolio Rupee Loss."""

    def __init__(self, config: Optional[CostMatrixConfig] = None):
        self.config = config or DEFAULT_COST_CONFIG
        self.loss_calc = RupeeLossCalculator(self.config)

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ead: Optional[np.ndarray] = None,
        lgd: Optional[np.ndarray] = None,
        n_grid_points: int = 250,
        min_threshold: float = 0.005,
        max_threshold: float = 0.70,
    ) -> Dict:
        """Exhaustive fine grid search over decision thresholds.
        
        Returns full optimization summary, optimal threshold, savings vs naive 0.5,
        and threshold curve history for plotting.
        """
        thresholds = np.linspace(min_threshold, max_threshold, n_grid_points)
        curve_records: List[Dict] = []
        
        best_loss = float("inf")
        best_threshold = 0.5
        best_metrics = {}
        
        for t in thresholds:
            res = self.loss_calc.calculate_portfolio_loss(
                y_true=y_true,
                y_prob=y_prob,
                threshold=t,
                ead=ead,
                lgd=lgd,
            )
            curve_records.append(res)
            
            if res["total_loss_rupees"] < best_loss:
                best_loss = res["total_loss_rupees"]
                best_threshold = t
                best_metrics = res

        # Evaluate at naive 0.5 threshold
        naive_metrics = self.loss_calc.calculate_portfolio_loss(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.50,
            ead=ead,
            lgd=lgd,
        )
        
        naive_loss = naive_metrics["total_loss_rupees"]
        savings_rupees = max(0.0, naive_loss - best_loss)
        pct_savings = (savings_rupees / max(naive_loss, 1e-6)) * 100.0
        
        df_curve = pd.DataFrame(curve_records)
        
        return {
            "optimal_threshold": float(best_threshold),
            "min_expected_loss_rupees": float(best_loss),
            "naive_threshold": 0.50,
            "naive_loss_rupees": float(naive_loss),
            "total_rupee_savings": float(savings_rupees),
            "pct_savings": float(pct_savings),
            "optimal_approval_rate": float(best_metrics["approval_rate"]),
            "naive_approval_rate": float(naive_metrics["approval_rate"]),
            "optimal_metrics": best_metrics,
            "naive_metrics": naive_metrics,
            "curve_dataframe": df_curve,
        }
