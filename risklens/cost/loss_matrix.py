"""Cost-sensitive evaluation and Expected Rupee Loss formulations."""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from risklens.config import DEFAULT_COST_CONFIG, CostMatrixConfig


class RupeeLossCalculator:
    """Calculates financial loss in INR (₹) across credit decisions.
    
    In credit card and loan underwriting:
    - False Negative (FN): Approving an applicant who defaults.
      Loss = Exposure At Default (EAD) * Loss Given Default (LGD).
    - False Positive (FP): Rejecting a creditworthy customer.
      Loss = Customer Acquisition Cost + Lifetime Interchange/Interest Foregone.
    - True Positive (TP): Flagged as high-risk/declined properly.
      Cost = Manual review / compliance handling cost (small fraction).
    - True Negative (TN): Approved good customer.
      Loss = 0.
    """
    
    def __init__(self, config: Optional[CostMatrixConfig] = None):
        self.config = config or DEFAULT_COST_CONFIG

    def calculate_portfolio_loss(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        ead: Optional[np.ndarray] = None,
        lgd: Optional[np.ndarray] = None,
        custom_fp_cost: Optional[float] = None,
    ) -> Dict[str, Union[float, int]]:
        """Calculate total and per-account financial loss at a given decision cutoff.
        
        Parameters
        ----------
        y_true : np.ndarray
            Binary true label (1 = Default / Fraud, 0 = Non-default).
        y_prob : np.ndarray
            Predicted calibrated default probability P(y=1).
        threshold : float
            Decision cut-off threshold.
            Applicants with y_prob <= threshold are APPROVED (Predicted Negative).
            Applicants with y_prob > threshold are DECLINED (Predicted Positive).
        ead : np.ndarray, optional
            Exposure At Default per applicant in INR.
        lgd : np.ndarray, optional
            Loss Given Default fraction (0.0 to 1.0) per applicant.
        custom_fp_cost : float, optional
            Opportunity cost per false rejection.
            
        Returns
        -------
        dict
            Total rupee loss, breakdown by FN and FP loss, approval rate, etc.
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        n = len(y_true)
        
        # Predicted decisions: 1 = Flagged/Declined, 0 = Approved
        y_pred = (y_prob > threshold).astype(int)
        
        # FN cost per account: EAD * LGD
        if ead is not None and lgd is not None:
            fn_costs = np.asarray(ead) * np.asarray(lgd)
        elif ead is not None:
            fn_costs = np.asarray(ead) * self.config.default_lgd
        else:
            fn_costs = np.full(n, self.config.default_fn_cost)
            
        fp_cost_val = custom_fp_cost if custom_fp_cost is not None else self.config.default_fp_cost
        tp_cost_val = self.config.manual_review_cost
        
        # Confusion matrix masks
        is_fn = (y_true == 1) & (y_pred == 0)  # Approved default
        is_fp = (y_true == 0) & (y_pred == 1)  # Declined good customer
        is_tp = (y_true == 1) & (y_pred == 1)  # Caught default/fraud
        is_tn = (y_true == 0) & (y_pred == 0)  # Approved good customer
        
        count_fn = int(np.sum(is_fn))
        count_fp = int(np.sum(is_fp))
        count_tp = int(np.sum(is_tp))
        count_tn = int(np.sum(is_tn))
        
        fn_loss_rupees = float(np.sum(fn_costs[is_fn]))
        fp_loss_rupees = float(count_fp * fp_cost_val)
        tp_friction_rupees = float(count_tp * tp_cost_val)
        
        total_loss_rupees = fn_loss_rupees + fp_loss_rupees + tp_friction_rupees
        per_account_loss = total_loss_rupees / max(n, 1)
        
        approval_count = count_tn + count_fn
        approval_rate = float(approval_count / max(n, 1))
        
        # Default rate in approved portfolio
        approved_defaults = count_fn
        approved_default_rate = float(approved_defaults / max(approval_count, 1))
        
        return {
            "threshold": float(threshold),
            "total_loss_rupees": total_loss_rupees,
            "per_account_loss_rupees": per_account_loss,
            "fn_loss_rupees": fn_loss_rupees,
            "fp_loss_rupees": fp_loss_rupees,
            "tp_friction_rupees": tp_friction_rupees,
            "count_fn": count_fn,
            "count_fp": count_fp,
            "count_tp": count_tp,
            "count_tn": count_tn,
            "approval_count": approval_count,
            "approval_rate": approval_rate,
            "approved_default_rate": approved_default_rate,
            "total_accounts": n,
        }

    def compute_individual_expected_loss(
        self,
        calibrated_prob: float,
        ead: float,
        lgd: float = 0.75,
    ) -> float:
        """Expected Rupee Loss if applicant is approved."""
        return float(calibrated_prob * ead * lgd)
