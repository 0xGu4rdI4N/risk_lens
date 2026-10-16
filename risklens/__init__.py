"""RiskLens: Cost-Sensitive Credit Default & Fraud Detection Engine.

Optimizing expected rupee loss rather than accuracy, measuring with PR-AUC,
setting cut-off threshold on purpose, explaining decisions with SHAP,
and monitoring data distribution with PSI.
"""

__version__ = "1.0.0"
__author__ = "Raunak Kumar"

from risklens.config import DEFAULT_COST_CONFIG, DEFAULT_DRIFT_CONFIG, DEFAULT_MODEL_CONFIG

__all__ = [
    "DEFAULT_COST_CONFIG",
    "DEFAULT_DRIFT_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "__version__",
]
