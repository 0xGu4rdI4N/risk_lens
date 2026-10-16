"""Population Stability Index (PSI) and Characteristic Stability Index (CSI).

Implements the Population Stability Index, a common credit-risk monitoring metric, to detect data distribution drift.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from risklens.config import DEFAULT_DRIFT_CONFIG, DriftConfig


class PSICalculator:
    """Calculates PSI between baseline (expected) and target (actual) distributions."""

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DEFAULT_DRIFT_CONFIG

    def calculate_continuous_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        num_bins: Optional[int] = None,
    ) -> Dict[str, Union[float, str, List]]:
        """Calculate PSI for continuous numerical series using quantile binning on expected."""
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)
        
        # Drop NaNs
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        n_bins = num_bins or self.config.num_bins
        eps = self.config.epsilon
        
        # Determine quantile breakpoints from expected distribution
        quantiles = np.linspace(0, 1, n_bins + 1)
        breakpoints = np.unique(np.quantile(expected, quantiles))
        
        # If too few unique values, fallback to uniform spacing
        if len(breakpoints) < 3:
            breakpoints = np.linspace(np.min(expected), np.max(expected), n_bins + 1)
            
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Bin both distributions
        exp_counts, _ = np.histogram(expected, bins=breakpoints)
        act_counts, _ = np.histogram(actual, bins=breakpoints)
        
        # Calculate percentage proportions
        exp_pct = exp_counts / max(len(expected), 1)
        act_pct = act_counts / max(len(actual), 1)
        
        # Epsilon smoothing
        exp_pct = np.clip(exp_pct, eps, 1.0)
        act_pct = np.clip(act_pct, eps, 1.0)
        
        # Re-normalize after clipping
        exp_pct = exp_pct / np.sum(exp_pct)
        act_pct = act_pct / np.sum(act_pct)
        
        # PSI per bin = (Actual% - Expected%) * ln(Actual% / Expected%)
        bin_psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        total_psi = float(np.sum(bin_psi))
        
        # Assign risk category
        if total_psi < self.config.stable_threshold:
            status = "STABLE"
        elif total_psi < self.config.warning_threshold:
            status = "WARNING"
        else:
            status = "SEVERE_DRIFT"
            
        return {
            "psi": total_psi,
            "status": status,
            "breakpoints": breakpoints.tolist(),
            "expected_pct": exp_pct.tolist(),
            "actual_pct": act_pct.tolist(),
            "bin_psi": bin_psi.tolist(),
        }

    def calculate_categorical_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
    ) -> Dict[str, Union[float, str, List]]:
        """Calculate PSI for discrete/categorical variables."""
        categories = list(set(expected.dropna().unique()).union(set(actual.dropna().unique())))
        eps = self.config.epsilon
        
        exp_counts = expected.value_counts(normalize=True)
        act_counts = actual.value_counts(normalize=True)
        
        exp_pct = np.array([exp_counts.get(cat, eps) for cat in categories])
        act_pct = np.array([act_counts.get(cat, eps) for cat in categories])
        
        exp_pct = np.clip(exp_pct, eps, 1.0)
        act_pct = np.clip(act_pct, eps, 1.0)
        
        exp_pct = exp_pct / np.sum(exp_pct)
        act_pct = act_pct / np.sum(act_pct)
        
        bin_psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        total_psi = float(np.sum(bin_psi))
        
        if total_psi < self.config.stable_threshold:
            status = "STABLE"
        elif total_psi < self.config.warning_threshold:
            status = "WARNING"
        else:
            status = "SEVERE_DRIFT"
            
        return {
            "psi": total_psi,
            "status": status,
            "categories": categories,
            "expected_pct": exp_pct.tolist(),
            "actual_pct": act_pct.tolist(),
            "bin_psi": bin_psi.tolist(),
        }
