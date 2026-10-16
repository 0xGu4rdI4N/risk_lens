"""Approval Volume vs. Financial Risk Frontier Engine.

Quantifies the business tradeoff between approval volume and expected loss,
measuring how many more credit approvals a better risk ranking allows at the
same approved-portfolio risk, with a bootstrap interval on that lift.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from risklens.config import DEFAULT_COST_CONFIG, CostMatrixConfig
from risklens.cost.loss_matrix import RupeeLossCalculator


class ApprovalFrontierEngine:
    """Computes the Efficient Underwriting Frontier and Approval Lift Analysis."""

    def __init__(self, config: Optional[CostMatrixConfig] = None):
        self.config = config or DEFAULT_COST_CONFIG
        self.loss_calc = RupeeLossCalculator(self.config)

    def generate_frontier_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ead: Optional[np.ndarray] = None,
        lgd: Optional[np.ndarray] = None,
        n_points: int = 150,
    ) -> pd.DataFrame:
        """Compute the full trade-off curve across approval volume from 50% to 98%."""
        # Use quantiles of predicted probabilities to generate evenly spaced approval volumes
        target_approvals = np.linspace(0.50, 0.98, n_points)
        curve_data: List[Dict] = []
        
        for app_rate in target_approvals:
            # Cut-off threshold corresponding to this approval volume
            thresh = float(np.quantile(y_prob, app_rate))
            metrics = self.loss_calc.calculate_portfolio_loss(
                y_true=y_true,
                y_prob=y_prob,
                threshold=thresh,
                ead=ead,
                lgd=lgd,
            )
            
            # Loss rate = total default loss / total approved exposure
            if ead is not None:
                approved_mask = y_prob <= thresh
                approved_exposure = float(np.sum(ead[approved_mask]))
                loss_rate_pct = (metrics["fn_loss_rupees"] / max(approved_exposure, 1.0)) * 100.0
            else:
                loss_rate_pct = (metrics["fn_loss_rupees"] / max(metrics["approval_count"] * self.config.default_fn_cost, 1.0)) * 100.0

            curve_data.append({
                "approval_volume_pct": float(app_rate * 100.0),
                "threshold": thresh,
                "total_loss_rupees": metrics["total_loss_rupees"],
                "fn_loss_rupees": metrics["fn_loss_rupees"],
                "fp_loss_rupees": metrics["fp_loss_rupees"],
                "approved_default_rate_pct": metrics["approved_default_rate"] * 100.0,
                "portfolio_loss_rate_pct": float(loss_rate_pct),
                "approval_count": metrics["approval_count"],
            })
            
        return pd.DataFrame(curve_data)

    @staticmethod
    def _max_approved_at_matched_risk(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target_default_rate: float,
        min_approval_rate: float = 0.50,
    ) -> int:
        """Largest number of lowest-risk accounts that can be approved while the
        approved-portfolio default rate stays <= target_default_rate."""
        n = len(y_true)
        order = np.argsort(y_prob, kind="stable")
        cum_defaults = np.cumsum(y_true[order])
        k = np.arange(1, n + 1)
        ok = (cum_defaults / k) <= target_default_rate
        ok[: int(min_approval_rate * n)] = False  # ignore the noisy low-volume head
        idx = np.flatnonzero(ok)
        return int(idx[-1] + 1) if idx.size else int(min_approval_rate * n)

    def compute_approval_expansion(
        self,
        y_true: np.ndarray,
        baseline_prob: np.ndarray,
        risklens_prob: np.ndarray,
        ead: Optional[np.ndarray] = None,
        lgd: Optional[np.ndarray] = None,
        baseline_approval_rate: float = 0.78,
        match: str = "approved_default_rate",
        n_bootstrap: int = 500,
        seed: int = 0,
        avg_annual_margin_per_good_account: float = 6_500.0,
    ) -> Dict:
        """Measure how many more accounts RiskLens can approve at matched risk.

        1. The baseline scorecard approves its lowest-risk `baseline_approval_rate`
           share of accounts; record its approved-portfolio risk.
        2. Search for the largest RiskLens approval volume whose risk does not exceed
           the baseline's. The lift is an OUTPUT of this search, not an input.
        3. Bootstrap the accounts to put a 95% interval on the lift.

        `match` selects the risk measure held constant: "approved_default_rate"
        (share of approved accounts that default) or "fn_loss" (total rupee loss
        from approved defaults).
        """
        if match not in ("approved_default_rate", "fn_loss"):
            raise ValueError("match must be 'approved_default_rate' or 'fn_loss'")
        y_true = np.asarray(y_true).astype(int)
        baseline_prob = np.asarray(baseline_prob, dtype=float)
        risklens_prob = np.asarray(risklens_prob, dtype=float)
        n = len(y_true)
        ead_a = None if ead is None else np.asarray(ead, dtype=float)
        lgd_a = None if lgd is None else np.asarray(lgd, dtype=float)

        base_threshold = float(np.quantile(baseline_prob, baseline_approval_rate))
        base_metrics = self.loss_calc.calculate_portfolio_loss(
            y_true, baseline_prob, base_threshold, ead=ead_a, lgd=lgd_a)

        if match == "approved_default_rate":
            k_rl = self._max_approved_at_matched_risk(
                y_true, risklens_prob, base_metrics["approved_default_rate"])
        else:
            k_rl = self._max_approved_at_fn_loss(
                y_true, risklens_prob, base_metrics["fn_loss_rupees"], ead_a, lgd_a, self.config)
        order = np.argsort(risklens_prob, kind="stable")
        risklens_threshold = float(risklens_prob[order][k_rl - 1])
        risklens_metrics = self.loss_calc.calculate_portfolio_loss(
            y_true, risklens_prob, risklens_threshold, ead=ead_a, lgd=lgd_a)

        risklens_rate = risklens_metrics["approval_rate"]
        lift_pts = (risklens_rate - base_metrics["approval_rate"]) * 100.0
        additional = risklens_metrics["approval_count"] - base_metrics["approval_count"]

        newly_approved_good = risklens_metrics["count_tn"] - base_metrics["count_tn"]
        incremental_revenue = max(0, newly_approved_good) * avg_annual_margin_per_good_account
        net_benefit = incremental_revenue - (risklens_metrics["fn_loss_rupees"] - base_metrics["fn_loss_rupees"])

        boot = self._bootstrap_lift(y_true, baseline_prob, risklens_prob, baseline_approval_rate,
                                    match, ead_a, lgd_a, n_bootstrap, seed)

        def block(m):
            return {
                "approval_rate_pct": float(m["approval_rate"] * 100.0),
                "threshold": m["threshold"],
                "approval_count": m["approval_count"],
                "total_loss_rupees": m["total_loss_rupees"],
                "fn_loss_rupees": m["fn_loss_rupees"],
                "approved_default_rate_pct": float(m["approved_default_rate"] * 100.0),
            }

        return {
            "matched_on": match,
            "n_accounts": n,
            "baseline": block(base_metrics),
            "risklens_expansion": block(risklens_metrics),
            "lift_metrics": {
                "approval_lift_pct_points": float(lift_pts),
                "additional_approved_accounts": int(additional),
                "approved_default_rate_delta_pct_points": float(
                    (risklens_metrics["approved_default_rate"] - base_metrics["approved_default_rate"]) * 100.0),
                "fn_loss_delta_rupees": float(risklens_metrics["fn_loss_rupees"] - base_metrics["fn_loss_rupees"]),
                "assumed_annual_margin_per_good_account_rupees": float(avg_annual_margin_per_good_account),
                "incremental_annual_revenue_rupees": float(incremental_revenue),
                "net_financial_benefit_rupees": float(net_benefit),
            },
            "bootstrap": boot,
        }

    @staticmethod
    def _max_approved_at_fn_loss(y_true, y_prob, target_fn_loss, ead, lgd, config,
                                 min_approval_rate: float = 0.50) -> int:
        n = len(y_true)
        if ead is not None and lgd is not None:
            cost = ead * lgd
        elif ead is not None:
            cost = ead * config.default_lgd
        else:
            cost = np.full(n, config.default_fn_cost)
        order = np.argsort(y_prob, kind="stable")
        cum_loss = np.cumsum((y_true * cost)[order])
        ok = cum_loss <= target_fn_loss
        ok[: int(min_approval_rate * n)] = False
        idx = np.flatnonzero(ok)
        return int(idx[-1] + 1) if idx.size else int(min_approval_rate * n)

    def _bootstrap_lift(self, y_true, baseline_prob, risklens_prob, baseline_approval_rate,
                        match, ead, lgd, n_bootstrap, seed) -> Dict:
        rng = np.random.default_rng(seed)
        n = len(y_true)
        k_base = int(round(baseline_approval_rate * n))
        lifts = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            i = rng.integers(0, n, n)
            y, pb, pr = y_true[i], baseline_prob[i], risklens_prob[i]
            base_idx = np.argsort(pb, kind="stable")[:k_base]
            if match == "approved_default_rate":
                k_rl = self._max_approved_at_matched_risk(y, pr, y[base_idx].mean())
            else:
                e = None if ead is None else ead[i]
                l = None if lgd is None else lgd[i]
                cost = (e * l) if (e is not None and l is not None) else np.full(n, self.config.default_fn_cost)
                k_rl = self._max_approved_at_fn_loss(
                    y, pr, float((y * cost)[base_idx].sum()), e, l, self.config)
            lifts[b] = (k_rl - k_base) / n * 100.0
        lo, hi = np.percentile(lifts, [2.5, 97.5])
        return {
            "n_resamples": int(n_bootstrap),
            "lift_pct_points_mean": float(lifts.mean()),
            "lift_pct_points_ci95": [float(lo), float(hi)],
            "share_of_resamples_with_positive_lift": float((lifts > 0).mean()),
        }
