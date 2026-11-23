"""Pydantic schemas for RiskLens data contracts, APIs, and scoring payloads."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ApplicantFeatures(BaseModel):
    """Credit applicant and transaction risk attributes."""
    applicant_id: Optional[str] = Field(default=None, description="Unique loan/card applicant identifier")
    annual_income: float = Field(..., ge=10_000, description="Annual verifiable income in INR (₹)")
    cibil_score: int = Field(..., ge=300, le=900, description="Bureau credit score (CIBIL/Experian)")
    revolving_utilization: float = Field(..., ge=0.0, le=2.5, description="Credit card line utilization ratio (0.0 to 1.0+)")
    debt_to_income: float = Field(..., ge=0.0, le=1.5, description="Total monthly debt service divided by gross monthly income")
    delinquencies_2yr: int = Field(..., ge=0, description="Number of 30+ DPD delinquencies past 24 months")
    inquiries_last_6m: int = Field(..., ge=0, description="Hard credit inquiries in past 6 months")
    open_credit_lines: int = Field(..., ge=0, description="Number of currently active open credit lines")
    credit_history_months: int = Field(..., ge=1, description="Length of credit history in months")
    transaction_velocity_24h: float = Field(..., ge=0.0, description="Transaction velocity factor past 24 hours")
    device_risk_score: float = Field(..., ge=0.0, le=1.0, description="Digital device fingerprint risk score (0.0=safe, 1.0=bot/fraud)")
    merchant_category: str = Field(default="retail", description="Merchant risk category (retail, travel, luxury, crypto, gaming)")
    employment_type: str = Field(default="salaried", description="Employment category (salaried, self_employed, gig_economy)")
    requested_credit_limit: float = Field(..., ge=5_000, description="Requested credit limit / loan amount in INR (₹)")


class ScoringRequest(BaseModel):
    """Single applicant scoring request with optional custom threshold."""
    features: ApplicantFeatures
    threshold: Optional[float] = Field(default=None, description="Optional custom decision threshold override")


class ScoringResult(BaseModel):
    """Detailed risk scoring response."""
    applicant_id: Optional[str]
    raw_score: float = Field(..., description="Uncalibrated model output")
    calibrated_default_prob: float = Field(..., description="Calibrated default probability P(Default=1)")
    decision: str = Field(..., description="'APPROVED' or 'DECLINED'")
    applied_threshold: float = Field(..., description="Decision cut-off threshold applied")
    expected_rupee_loss: float = Field(..., description="Expected loss in INR if approved (EAD * LGD * Prob)")
    top_risk_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Top SHAP feature contributions")
    adverse_action_reasons: List[str] = Field(default_factory=list, description="Regulatory compliance plain-English reasons if declined")


class BatchScoringRequest(BaseModel):
    applicants: List[ApplicantFeatures]
    threshold: Optional[float] = None


class ThresholdOptimizationResult(BaseModel):
    """Result of cost-sensitive threshold search."""
    optimal_threshold: float
    min_expected_loss_rupees: float
    naive_loss_rupees_at_05: float
    total_rupee_savings: float
    pct_savings: float
    approval_rate_at_optimal: float
    approval_rate_at_naive: float
    target_lift_approval_rate: float
    target_lift_threshold: float


class PSIScoreItem(BaseModel):
    feature_name: str
    psi_value: float
    status: str  # "STABLE", "WARNING", "SEVERE_DRIFT"


class DriftReport(BaseModel):
    timestamp: str
    total_samples_baseline: int
    total_samples_target: int
    score_psi: float
    score_status: str
    feature_psi: List[PSIScoreItem]
    overall_system_status: str
