"""Configuration and default hyperparameters for RiskLens Engine."""

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Ensure directories exist
for directory in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CostMatrixConfig:
    """Cost Matrix Hyperparameters for Credit Default & Fraud Underwriting.
    
    Currency: Indian Rupee (INR - ₹)
    
    - False Negative (FN): Approving an applicant who defaults / commits fraud.
      Bank loses Exposure at Default (EAD) * Loss Given Default (LGD) + Recovery Cost.
      Typical EAD: ₹50,000 to ₹500,000. LGD: 0.75.
      Default baseline FN cost if EAD is fixed: ₹75,000.
      
    - False Positive (FP): Rejecting or blocking a creditworthy customer.
      Loss equals Customer Lifetime Margin (interest + interchange) + Customer Acquisition Cost (CAC) + manual review friction.
      Typical FP cost: ₹3,500.
      
    Cost Ratio FN : FP ≈ 21.4 : 1. Standard 0.5 threshold is commercially disastrous.
    """
    default_fn_cost: float = 75_000.0   # Fixed default loss if EAD not provided
    default_fp_cost: float = 3_500.0    # Opportunity cost of declined good applicant
    manual_review_cost: float = 450.0   # Cost to manually investigate flagged application
    default_lgd: float = 0.75           # Loss Given Default percentage


@dataclass(frozen=True)
class DriftConfig:
    """PSI (Population Stability Index) rule-of-thumb thresholds used in credit-risk monitoring."""
    stable_threshold: float = 0.10      # PSI < 0.10: Stable, no distribution change
    warning_threshold: float = 0.20     # 0.10 <= PSI < 0.20: Moderate drift, monitor
    # PSI >= 0.20: Significant drift, action required (re-calibration / retraining)
    num_bins: int = 10
    epsilon: float = 1e-4               # Smoothing epsilon for zero count bins


@dataclass(frozen=True)
class ModelConfig:
    """Training, calibration, and tuning hyperparameters."""
    random_state: int = 42
    test_size: float = 0.20
    oot_size: float = 0.15              # Out-Of-Time validation split for macro drift
    n_estimators: int = 250
    learning_rate: float = 0.04
    max_depth: int = 6
    scale_pos_weight: float = 15.0      # Compensate for 2% - 3% default rate
    subsample: float = 0.85
    colsample_bytree: float = 0.85


# Instantiated singletons for convenient import
DEFAULT_COST_CONFIG = CostMatrixConfig()
DEFAULT_DRIFT_CONFIG = DriftConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
