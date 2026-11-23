"""High-fidelity synthetic Credit Default & Fraud Dataset generator.

Simulates authentic financial distributions, realistic non-linear default risk,
EAD (Exposure at Default), LGD (Loss Given Default), and controlled Out-Of-Time (OOT)
macroeconomic distribution shifts to benchmark PSI drift detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_credit_dataset(
    n_samples: int = 100_000,
    random_state: int = 42,
    macro_drift: bool = False,
) -> pd.DataFrame:
    """Generate synthetic credit and fraud records with realistic financial covariance.
    
    Parameters
    ----------
    n_samples : int
        Number of synthetic records to generate.
    random_state : int
        Seed for reproducibility.
    macro_drift : bool
        If True, applies macroeconomic stress (delinquency creep, bureau score deflation,
        utilization jump) representing out-of-time population drift.
        
    Returns
    -------
    pd.DataFrame
        Synthetic credit records with feature attributes, costs, and default labels.
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Annual Income (Log-normal distribution, median ~₹750,000, mean ~₹950,000)
    mu_inc = 13.5
    sigma_inc = 0.65
    annual_income = np.exp(rng.normal(mu_inc, sigma_inc, n_samples))
    annual_income = np.clip(annual_income, 120_000, 15_000_000)
    
    # 2. Bureau CIBIL Score (300 to 900)
    # Drift shifts score distribution downwards by ~25 points
    drift_score_shift = -28.0 if macro_drift else 0.0
    cibil_base = rng.beta(6.0, 2.8, n_samples) * 550 + 320 + drift_score_shift
    cibil_score = np.clip(np.round(cibil_base), 300, 890).astype(int)
    
    # 3. Revolving Line Utilization (0.0 to 1.0+)
    # Drift pushes credit card utilization higher (liquidity squeeze)
    drift_util_boost = 0.12 if macro_drift else 0.0
    revolving_utilization = rng.beta(2.0, 4.5, n_samples) + drift_util_boost
    revolving_utilization = np.clip(revolving_utilization, 0.01, 1.45)
    
    # 4. Debt-to-Income (DTI)
    dti_base = 0.28 + 0.15 * (revolving_utilization) - 0.08 * (cibil_score - 650) / 250
    dti = np.clip(dti_base + rng.normal(0, 0.09, n_samples), 0.02, 1.20)
    
    # 5. Delinquencies in Past 2 Years (Poisson, correlated with low CIBIL)
    delinq_rate = np.exp(-0.008 * (cibil_score - 500))
    if macro_drift:
        delinq_rate *= 1.45  # 45% uptick in delinquencies under stress
    delinquencies_2yr = rng.poisson(np.clip(delinq_rate, 0.05, 3.5), n_samples)
    
    # 6. Hard Credit Inquiries past 6 months
    inq_rate = 0.8 + 1.2 * (revolving_utilization > 0.65)
    inquiries_last_6m = rng.poisson(inq_rate, n_samples)
    
    # 7. Open Credit Lines
    open_lines_base = 2 + (annual_income / 300_000) + rng.poisson(2, n_samples)
    open_credit_lines = np.clip(np.round(open_lines_base), 1, 25).astype(int)
    
    # 8. Credit History Length (Months)
    credit_history_months = np.clip(
        np.round(rng.normal(84, 42, n_samples) + (annual_income / 100_000)),
        6,
        360
    ).astype(int)
    
    # 9. Transaction Velocity past 24 hours (Velocity factor)
    transaction_velocity_24h = np.clip(rng.exponential(1.5, n_samples), 0.1, 18.0)
    
    # 10. Device Fingerprint Risk Score (0.0 to 1.0)
    device_risk_score = rng.beta(1.1, 6.0, n_samples)
    # Extreme device risk bursts for synthetic fraud attacks
    fraud_burst = rng.random(n_samples) < 0.012
    device_risk_score[fraud_burst] = rng.uniform(0.75, 0.99, np.sum(fraud_burst))
    
    # 11. Categorical Features: Merchant Category & Employment Type
    merchant_cats = ["retail", "travel", "luxury", "gaming", "crypto"]
    merchant_probs = [0.60, 0.20, 0.10, 0.06, 0.04]
    merchant_category = rng.choice(merchant_cats, size=n_samples, p=merchant_probs)
    
    employment_types = ["salaried", "self_employed", "gig_economy"]
    emp_probs = [0.65, 0.25, 0.10]
    employment_type = rng.choice(employment_types, size=n_samples, p=emp_probs)
    
    # 12. Requested Credit Limit / Loan Amount (₹)
    # Approximately 15% to 35% of annual income
    limit_factor = rng.uniform(0.15, 0.35, n_samples)
    requested_credit_limit = np.round(annual_income * limit_factor / 5000) * 5000
    requested_credit_limit = np.clip(requested_credit_limit, 25_000, 2_500_000)
    
    # 13. Financial Exposure Parameters: EAD & LGD
    # EAD (Exposure At Default): typically 70% to 100% of line at default
    ead = requested_credit_limit * np.clip(0.65 + 0.35 * revolving_utilization, 0.50, 1.0)
    
    # LGD (Loss Given Default): typical unsecured recovery loss ~ 65% - 85%
    lgd = np.clip(rng.beta(7.0, 3.0, n_samples), 0.45, 0.95)
    
    # 14. Latent Default & Fraud Probability Engine (Authentic FinTech Non-Linear Multi-Factor Model)
    util_risk = 3.6 * np.maximum(0.0, revolving_utilization - 0.40) ** 1.8
    dti_risk = 3.2 * np.maximum(0.0, dti - 0.35) ** 1.6
    cibil_gap = np.maximum(0.0, (660.0 - cibil_score) / 100.0)
    cibil_risk = -0.012 * (cibil_score - 700.0) + 2.2 * (cibil_gap ** 1.5)
    
    fraud_interaction = 5.0 * (device_risk_score > 0.60) * (transaction_velocity_24h > 4.0) * (requested_credit_limit > 300_000)
    leverage_spiral = 3.0 * (revolving_utilization > 0.60) * (inquiries_last_6m >= 2) * (dti > 0.40)
    
    z = (
        -6.40  # Base calibration intercept for realistic ~3.5% baseline default rate
        + cibil_risk
        + util_risk
        + dti_risk
        + 0.65 * delinquencies_2yr
        + 0.40 * inquiries_last_6m
        + 2.80 * device_risk_score
        + fraud_interaction
        + leverage_spiral
        + 0.85 * (merchant_category == "crypto").astype(float)
        + 0.65 * (merchant_category == "gaming").astype(float)
        + 0.40 * (employment_type == "gig_economy").astype(float)
    )
    
    true_default_prob = 1.0 / (1.0 + np.exp(-z))
    is_default = (rng.random(n_samples) < true_default_prob).astype(int)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "applicant_id": [f"APP-{100000 + i}" for i in range(n_samples)],
        "annual_income": np.round(annual_income, 2),
        "cibil_score": cibil_score,
        "revolving_utilization": np.round(revolving_utilization, 4),
        "debt_to_income": np.round(dti, 4),
        "delinquencies_2yr": delinquencies_2yr,
        "inquiries_last_6m": inquiries_last_6m,
        "open_credit_lines": open_credit_lines,
        "credit_history_months": credit_history_months,
        "transaction_velocity_24h": np.round(transaction_velocity_24h, 2),
        "device_risk_score": np.round(device_risk_score, 4),
        "merchant_category": merchant_category,
        "employment_type": employment_type,
        "requested_credit_limit": requested_credit_limit,
        "exposure_at_default": np.round(ead, 2),
        "loss_given_default": np.round(lgd, 4),
        "true_risk_prob": np.round(true_default_prob, 5),
        "is_default": is_default,
        "period": "OOT" if macro_drift else "IN_TIME",
    })
    
    return df


def generate_train_oot_splits(
    n_train: int = 80_000,
    n_oot: int = 20_000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate both in-time training/test cohort and out-of-time macroeconomic drift cohort."""
    df_intime = generate_credit_dataset(n_samples=n_train, random_state=random_state, macro_drift=False)
    df_oot = generate_credit_dataset(n_samples=n_oot, random_state=random_state + 999, macro_drift=True)
    return df_intime, df_oot
