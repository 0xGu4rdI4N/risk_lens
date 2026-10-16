"""FastAPI Production Inference and Underwriting Engine."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from risklens import __version__
from risklens.config import MODELS_DIR, DEFAULT_COST_CONFIG
from risklens.data.preprocessor import CreditRiskPreprocessor
from risklens.data.schema import (
    ApplicantFeatures,
    ScoringRequest,
    ScoringResult,
    BatchScoringRequest,
    ThresholdOptimizationResult,
    DriftReport,
)
from risklens.cost.loss_matrix import RupeeLossCalculator
from risklens.cost.optimizer import ThresholdOptimizer
from risklens.cost.frontier import ApprovalFrontierEngine
from risklens.models.calibrator import ProbabilityCalibrator
from risklens.models.tree_models import LightGBMRiskClassifier
from risklens.explainability.explainer import RiskLensExplainer
from risklens.explainability.adverse_action import AdverseActionGenerator
from risklens.monitoring.drift_detector import DriftDetector


# Global runtime state
runtime_state = {
    "preprocessor": None,
    "model": None,
    "calibrator": None,
    "explainer": None,
    "optimal_threshold": 0.085,  # Default cost-sensitive threshold
    "loss_calculator": RupeeLossCalculator(),
    "optimizer": ThresholdOptimizer(),
    "frontier_engine": ApprovalFrontierEngine(),
    "drift_detector": DriftDetector(),
}


def load_artifacts_if_available():
    """Load pre-trained artifacts if present in MODELS_DIR."""
    prep_path = MODELS_DIR / "preprocessor.joblib"
    model_path = MODELS_DIR / "lightgbm_model.joblib"
    calib_path = MODELS_DIR / "calibrator.joblib"
    explainer_path = MODELS_DIR / "explainer.joblib"
    thresh_path = MODELS_DIR / "optimal_threshold.txt"

    if prep_path.exists():
        runtime_state["preprocessor"] = CreditRiskPreprocessor.load(str(prep_path))
    if model_path.exists():
        runtime_state["model"] = LightGBMRiskClassifier.load(str(model_path))
    if calib_path.exists():
        runtime_state["calibrator"] = ProbabilityCalibrator.load(str(calib_path))
    if explainer_path.exists():
        runtime_state["explainer"] = RiskLensExplainer.load(str(explainer_path))
    if thresh_path.exists():
        try:
            runtime_state["optimal_threshold"] = float(thresh_path.read_text().strip())
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts_if_available()
    yield


app = FastAPI(
    title="RiskLens API",
    description="Cost-Sensitive Credit Default & Fraud Detection Underwriting Service",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "RiskLens Underwriting Engine",
        "version": __version__,
        "artifacts_loaded": {
            "preprocessor": runtime_state["preprocessor"] is not None,
            "model": runtime_state["model"] is not None,
            "calibrator": runtime_state["calibrator"] is not None,
            "explainer": runtime_state["explainer"] is not None,
        },
        "active_threshold": runtime_state["optimal_threshold"],
    }


@app.post("/predict", response_model=ScoringResult, tags=["Underwriting"])
def score_applicant(request: ScoringRequest):
    """Score an individual applicant, evaluate rupee risk, explain with SHAP, and generate adverse action notice if declined."""
    if runtime_state["model"] is None or runtime_state["preprocessor"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not trained or loaded yet. Run pipeline training first.",
        )

    features_dict = request.features.model_dump()
    app_id = features_dict.get("applicant_id", "APP-ONLINE")
    df_single = pd.DataFrame([features_dict])

    # Preprocessing
    X_trans = runtime_state["preprocessor"].transform(df_single)
    
    # Raw score & calibrated probability
    raw_score = float(runtime_state["model"].predict_proba(X_trans)[0])
    if runtime_state["calibrator"] is not None:
        calib_prob = float(runtime_state["calibrator"].predict_proba(X_trans)[0])
    else:
        calib_prob = raw_score

    threshold = request.threshold if request.threshold is not None else runtime_state["optimal_threshold"]

    # Decision logic: Approval if risk prob <= threshold
    is_declined = calib_prob > threshold
    decision = "DECLINED" if is_declined else "APPROVED"

    # Exposure & Expected Loss
    limit = features_dict.get("requested_credit_limit", 100_000.0)
    util = features_dict.get("revolving_utilization", 0.30)
    ead = limit * (0.65 + 0.35 * min(1.0, util))
    expected_loss = runtime_state["loss_calculator"].compute_individual_expected_loss(
        calibrated_prob=calib_prob,
        ead=ead,
    )

    # SHAP Explainability & Adverse Action Notices
    top_factors = []
    adverse_reasons = []
    if runtime_state["explainer"] is not None:
        top_factors = runtime_state["explainer"].explain_instance(X_trans, top_k=5)
        if is_declined:
            adverse_reasons = AdverseActionGenerator.generate_reasons(top_factors, max_reasons=4)

    return ScoringResult(
        applicant_id=app_id,
        raw_score=round(raw_score, 5),
        calibrated_default_prob=round(calib_prob, 5),
        decision=decision,
        applied_threshold=round(threshold, 4),
        expected_rupee_loss=round(expected_loss, 2),
        top_risk_factors=top_factors,
        adverse_action_reasons=adverse_reasons,
    )


@app.post("/batch-predict", tags=["Underwriting"])
def batch_score(request: BatchScoringRequest):
    """Vectorized scoring of multiple credit applications."""
    if runtime_state["model"] is None or runtime_state["preprocessor"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not trained or loaded yet.",
        )

    data = [app.model_dump() for app in request.applicants]
    df = pd.DataFrame(data)
    X_trans = runtime_state["preprocessor"].transform(df)

    raw_scores = runtime_state["model"].predict_proba(X_trans)
    if runtime_state["calibrator"] is not None:
        calib_probs = runtime_state["calibrator"].predict_proba(X_trans)
    else:
        calib_probs = raw_scores

    threshold = request.threshold if request.threshold is not None else runtime_state["optimal_threshold"]
    decisions = ["DECLINED" if p > threshold else "APPROVED" for p in calib_probs]

    approvals = sum(1 for d in decisions if d == "APPROVED")
    approval_rate = approvals / len(decisions)

    return {
        "total_applicants": len(decisions),
        "total_approved": approvals,
        "approval_rate_pct": round(approval_rate * 100.0, 2),
        "applied_threshold": threshold,
        "results": [
            {
                "applicant_id": row.get("applicant_id"),
                "calibrated_default_prob": round(float(p), 4),
                "decision": dec,
            }
            for row, p, dec in zip(data, calib_probs, decisions)
        ],
    }


@app.get("/config/costs", tags=["Configuration"])
def get_cost_config():
    """Retrieve current operational cost matrix hyperparameters."""
    return {
        "default_fn_cost_rupees": runtime_state["loss_calculator"].config.default_fn_cost,
        "default_fp_cost_rupees": runtime_state["loss_calculator"].config.default_fp_cost,
        "manual_review_cost_rupees": runtime_state["loss_calculator"].config.manual_review_cost,
        "active_cutoff_threshold": runtime_state["optimal_threshold"],
    }
