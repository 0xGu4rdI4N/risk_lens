"""Interactive Streamlit Underwriting & Risk Monitoring Dashboard."""

import json

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from risklens.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR, DEFAULT_COST_CONFIG
from risklens.data.preprocessor import CreditRiskPreprocessor
from risklens.models.calibrator import ProbabilityCalibrator
from risklens.models.tree_models import LightGBMRiskClassifier
from risklens.cost.loss_matrix import RupeeLossCalculator
from risklens.cost.frontier import ApprovalFrontierEngine
from risklens.explainability.explainer import RiskLensExplainer
from risklens.explainability.adverse_action import AdverseActionGenerator
from risklens.monitoring.drift_detector import DriftDetector

# Page configuration
st.set_page_config(
    page_title="RiskLens | Cost-Sensitive Underwriting Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-stable {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-severe {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_runtime_artifacts():
    """Cache models, preprocessor, and explainer in memory."""
    prep_path = MODELS_DIR / "preprocessor.joblib"
    model_path = MODELS_DIR / "lightgbm_model.joblib"
    calib_path = MODELS_DIR / "calibrator.joblib"
    explainer_path = MODELS_DIR / "explainer.joblib"
    thresh_path = MODELS_DIR / "optimal_threshold.txt"

    artifacts = {}
    if prep_path.exists():
        artifacts["prep"] = CreditRiskPreprocessor.load(str(prep_path))
    if model_path.exists():
        artifacts["model"] = LightGBMRiskClassifier.load(str(model_path))
    if calib_path.exists():
        artifacts["calib"] = ProbabilityCalibrator.load(str(calib_path))
    if explainer_path.exists():
        artifacts["explainer"] = RiskLensExplainer.load(str(explainer_path))
    if thresh_path.exists():
        try:
            artifacts["optimal_threshold"] = float(thresh_path.read_text().strip())
        except Exception:
            artifacts["optimal_threshold"] = 0.085
    else:
        artifacts["optimal_threshold"] = 0.085
        
    return artifacts


def load_reports() -> dict:
    """Read the metrics written by scripts/run_pipeline.py so the UI never shows hand-typed numbers."""
    out = {}
    for key, name in [("lift", "approval_lift_metrics.json"), ("bench", "model_benchmark.json"),
                      ("drift", "drift_audit.json"), ("summary", "run_summary.json")]:
        path = REPORTS_DIR / name
        if path.exists():
            out[key] = json.loads(path.read_text())
    return out


artifacts = load_runtime_artifacts()
reports = load_reports()
loss_calc = RupeeLossCalculator()
frontier_engine = ApprovalFrontierEngine()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/security-shield-green.png", width=64)
st.sidebar.title("RiskLens Engine")
st.sidebar.caption("Cost-Sensitive Credit Risk & Fraud Detection | InfoSecIITR")

menu = st.sidebar.radio(
    "Navigation Module",
    [
        "Executive Dashboard",
        "Cost-Loss Frontier & Approval Lift",
        "Underwriting Sandbox & SHAP",
        "PSI Drift Monitor",
        "Mathematical Formulations",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Cost Matrix Parameters (INR ₹)")
st.sidebar.markdown(f"**FN Cost (Default Loss):** ₹{loss_calc.config.default_fn_cost:,.0f}")
st.sidebar.markdown(f"**FP Cost (Foregone Margin):** ₹{loss_calc.config.default_fp_cost:,.0f}")
st.sidebar.markdown(f"**Cost Ratio (FN:FP):** {loss_calc.config.default_fn_cost / loss_calc.config.default_fp_cost:.1f} : 1")
st.sidebar.markdown(f"**Default Cut-off $\\tau^*$:** `{artifacts.get('optimal_threshold', 0.085):.3f}`")


# -------------------------------------------------------------
# 1. EXECUTIVE DASHBOARD
# -------------------------------------------------------------
if menu == "Executive Dashboard":
    st.title("💳 RiskLens: Cost-Sensitive Credit Underwriting & Fraud Intelligence")
    st.markdown("""
    **Cost-sensitive credit risk engine demo, running on synthetic data.**
    Optimizes **expected rupee loss rather than accuracy**, ranks with **PR-AUC**,
    explains decisions with **SHAP**, and monitors drift with **PSI (Population Stability Index)**.
    """)
    if not reports:
        st.warning("No pipeline outputs found. Run `python scripts/run_pipeline.py` to populate the metrics below.")

    bench = {r["model_name"]: r for r in reports.get("bench", [])}
    rl_row = bench.get("RiskLens (Calibrated LightGBM)")
    sc_row = bench.get("Baseline Scorecard (Logistic)")
    lift = reports.get("lift")
    drift = reports.get("drift")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if rl_row and sc_row:
            st.metric("Model PR-AUC", f"{rl_row['pr_auc']:.4f}",
                      f"{(rl_row['pr_auc'] / sc_row['pr_auc'] - 1) * 100:+.1f}% vs Scorecard",
                      help="Precision-Recall AUC on the held-out synthetic test set")
        else:
            st.metric("Model PR-AUC", "n/a")
    with col2:
        if lift:
            lm = lift["lift_metrics"]
            lo, hi = lift["bootstrap"]["lift_pct_points_ci95"]
            st.metric("Approval Volume", f"{lift['risklens_expansion']['approval_rate_pct']:.1f}%",
                      f"{lm['approval_lift_pct_points']:+.2f} pts at matched risk",
                      help=f"95% bootstrap CI on the lift: {lo:+.1f} to {hi:+.1f} pts")
        else:
            st.metric("Approval Volume", "n/a")
    with col3:
        if rl_row and sc_row:
            gain = sc_row["loss_at_optimal_threshold_rupees"] - rl_row["loss_at_optimal_threshold_rupees"]
            st.metric("Loss vs Scorecard (best cutoff)", f"₹{gain / 1e5:,.1f} Lakhs lower",
                      f"{gain / sc_row['loss_at_optimal_threshold_rupees'] * 100:.1f}% reduction",
                      help="Both models evaluated at their own cost-optimal cutoff on the same test set")
        else:
            st.metric("Loss vs Scorecard", "n/a")
    with col4:
        if drift:
            st.metric("Model Score PSI", f"{drift['score_psi']:.3f}", drift.get("score_status", ""),
                      help="Calibrated score, in-time vs out-of-time cohort. Rule of thumb: <0.10 stable, 0.10-0.20 warning, >=0.20 significant")
        else:
            st.metric("Model Score PSI", "n/a")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("🎯 Why Optimizing Rupee Loss Beats Accuracy")
        st.markdown("""
        In traditional machine learning, models are trained to maximize **Accuracy** or Cross-Entropy loss, implicitly assuming that:
        $$\\text{Cost}(\\text{False Positive}) = \\text{Cost}(\\text{False Negative})$$
        
        In credit card underwriting and fraud prevention, this assumption is **financially catastrophic**:
        * **False Negative (Default/Fraud Approved):** Customer defaults on their line. The lender loses Exposure at Default $\\times$ Loss Given Default (approx. **₹75,000+**).
        * **False Positive (Good Applicant Rejected):** Creditworthy customer is declined. Lender loses lifetime interchange fee and annual fee margin (approx. **₹3,500**).
        
        Because **False Negatives are ~21x more expensive than False Positives**, setting the decision cut-off at standard `0.5` causes massive underwriting losses. RiskLens finds the exact threshold $\\tau^*$ that minimizes expected rupee loss.
        """)

    with col_right:
        st.subheader("🏆 Approval Lift at Matched Risk")
        if lift:
            lm, bs = lift["lift_metrics"], lift["bootstrap"]
            st.info(f"""
            **Measured on synthetic data (n = {lift['n_accounts']:,} test accounts):**

            Holding the approved-portfolio default rate at or below the baseline scorecard's
            ({lift['baseline']['approved_default_rate_pct']:.2f}%), RiskLens can approve
            **{lift['risklens_expansion']['approval_rate_pct']:.1f}%** of applicants versus
            {lift['baseline']['approval_rate_pct']:.1f}% for the scorecard
            (**{lm['approval_lift_pct_points']:+.2f} pts**, {lm['additional_approved_accounts']:+,} accounts).

            95% bootstrap interval on the lift: **{bs['lift_pct_points_ci95'][0]:+.1f} to {bs['lift_pct_points_ci95'][1]:+.1f} pts**.
            The interval is wide because the test set has few defaults.
            """)
        else:
            st.info("Run the pipeline to compute the measured approval lift.")


# -------------------------------------------------------------
# 2. COST-LOSS FRONTIER & APPROVAL LIFT
# -------------------------------------------------------------
elif menu == "Cost-Loss Frontier & Approval Lift":
    st.title("📈 Approval Volume vs. Expected Rupee Loss Frontier")
    st.markdown("""
    Explore how adjusting the decision threshold shifts the frontier between portfolio approval volume and rupee credit loss.
    """)

    frontier_path = REPORTS_DIR / "frontier_curve.parquet"
    if frontier_path.exists():
        df_frontier = pd.read_parquet(frontier_path)
    else:
        # Fallback synthetic frontier for demonstration if not yet trained
        apps = np.linspace(50, 98, 100)
        losses = 1_200_000 + 45_000 * np.exp((apps - 50) / 11)
        df_frontier = pd.DataFrame({
            "approval_volume_pct": apps,
            "threshold": np.linspace(0.02, 0.35, 100),
            "total_loss_rupees": losses,
            "approved_default_rate_pct": 0.4 + 0.04 * (apps - 50),
        })

    # Interactive Slider
    st.subheader("🎛️ Dynamic Threshold Cut-off Controller")
    selected_thresh = st.slider(
        "Decision Cut-off Threshold (τ)",
        min_value=0.01,
        max_value=0.35,
        value=float(artifacts.get("optimal_threshold", 0.085)),
        step=0.005,
        format="%.3f"
    )

    # Approximate metrics at selected threshold
    closest_idx = (df_frontier["threshold"] - selected_thresh).abs().idxmin()
    row = df_frontier.loc[closest_idx]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Approval Volume", f"{row['approval_volume_pct']:.1f}%")
    with m2:
        st.metric("Expected Portfolio Loss", f"₹{row['total_loss_rupees']:,.0f}")
    with m3:
        st.metric("Approved Default Rate", f"{row['approved_default_rate_pct']:.2f}%")
    with m4:
        is_opt = abs(selected_thresh - artifacts.get("optimal_threshold", 0.085)) < 0.01
        st.metric("Cutoff Status", "Cost Optimal ★" if is_opt else "Sub-Optimal")

    # Plot Frontier Curve
    st.markdown("### Underwriting Frontier Curve")
    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    color = "#3b82f6"
    ax1.set_xlabel("Approval Volume (%)", fontsize=11)
    ax1.set_ylabel("Expected Rupee Loss (₹ Lakhs)", color=color, fontsize=11)
    line1 = ax1.plot(
        df_frontier["approval_volume_pct"],
        df_frontier["total_loss_rupees"] / 100_000,
        color=color,
        linewidth=2.5,
        label="Expected Rupee Loss"
    )
    ax1.tick_params(axis='y', labelcolor=color)

    # Secondary axis: Default rate
    ax2 = ax1.twinx()
    color = "#ef4444"
    ax2.set_ylabel("Approved Default Rate (%)", color=color, fontsize=11)
    line2 = ax2.plot(
        df_frontier["approval_volume_pct"],
        df_frontier["approved_default_rate_pct"],
        color=color,
        linestyle="--",
        linewidth=2,
        label="Approved Default Rate"
    )
    ax2.tick_params(axis='y', labelcolor=color)

    # Mark baseline and matched-risk approval volumes
    lift_rep = reports.get("lift")
    if lift_rep:
        b_rate = lift_rep["baseline"]["approval_rate_pct"]
        r_rate = lift_rep["risklens_expansion"]["approval_rate_pct"]
        ax1.axvline(x=b_rate, color="#94a3b8", linestyle=":", label=f"Scorecard baseline ({b_rate:.1f}%)")
        ax1.axvline(x=r_rate, color="#10b981", linestyle="-.", linewidth=2,
                    label=f"RiskLens at matched risk ({r_rate:.1f}%)")

    # Current selected threshold point
    ax1.scatter([row["approval_volume_pct"]], [row["total_loss_rupees"] / 100_000], color="#f59e0b", s=120, zorder=5, label="Active Operating Cutoff")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(loc="upper left")
    plt.title("Portfolio Underwriting Frontier: Approval Volume vs. Expected Rupee Loss", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig)


# -------------------------------------------------------------
# 3. UNDERWRITING SANDBOX & SHAP EXPLAINABILITY
# -------------------------------------------------------------
elif menu == "Underwriting Sandbox & SHAP":
    st.title("🔍 Individual Applicant Risk Scoring & SHAP Explainer")
    st.markdown("Test single applicant credit profiles, examine the calibrated default probability, and inspect regulatory adverse action codes.")

    # Preset profiles
    preset = st.selectbox(
        "Choose an Archetype Profile or Enter Custom Attributes:",
        [
            "Prime Salaried Executive (CIBIL 790, Low DTI)",
            "Borderline Subprime (CIBIL 640, 68% Utilization)",
            "Distressed Credit Seeker (3 Delinquencies, 4 Inquiries)",
            "Synthetic Fraud Spike (Device Anomaly, High Velocity)",
            "Custom Profile",
        ]
    )

    defaults = {
        "annual_income": 950_000.0,
        "cibil_score": 750,
        "revolving_utilization": 0.28,
        "debt_to_income": 0.25,
        "delinquencies_2yr": 0,
        "inquiries_last_6m": 0,
        "open_credit_lines": 5,
        "credit_history_months": 72,
        "transaction_velocity_24h": 1.2,
        "device_risk_score": 0.08,
        "merchant_category": "retail",
        "employment_type": "salaried",
        "requested_credit_limit": 250_000.0,
    }

    if "Borderline" in preset:
        defaults.update({"cibil_score": 640, "revolving_utilization": 0.68, "debt_to_income": 0.48, "inquiries_last_6m": 2})
    elif "Distressed" in preset:
        defaults.update({"cibil_score": 570, "revolving_utilization": 0.89, "delinquencies_2yr": 3, "inquiries_last_6m": 4, "debt_to_income": 0.62})
    elif "Fraud" in preset:
        defaults.update({"device_risk_score": 0.92, "transaction_velocity_24h": 12.5, "merchant_category": "crypto", "requested_credit_limit": 500_000.0})

    colA, colB, colC = st.columns(3)
    with colA:
        income = st.number_input("Annual Verifiable Income (₹)", value=defaults["annual_income"], step=50_000.0)
        cibil = st.slider("CIBIL Bureau Score", 300, 900, defaults["cibil_score"])
        util = st.slider("Revolving Line Utilization", 0.0, 1.5, defaults["revolving_utilization"], step=0.05)
        dti = st.slider("Debt-to-Income (DTI)", 0.0, 1.2, defaults["debt_to_income"], step=0.02)

    with colB:
        delinq = st.number_input("Delinquencies (30+ DPD in 2yr)", value=defaults["delinquencies_2yr"], min_value=0, max_value=15)
        inquiries = st.number_input("Hard Inquiries (Last 6 Months)", value=defaults["inquiries_last_6m"], min_value=0, max_value=20)
        open_lines = st.number_input("Active Open Credit Lines", value=defaults["open_credit_lines"], min_value=1, max_value=30)
        history_months = st.number_input("Credit History (Months)", value=defaults["credit_history_months"], min_value=6, max_value=400)

    with colC:
        velocity = st.number_input("24h Transaction Velocity Factor", value=defaults["transaction_velocity_24h"], min_value=0.1, max_value=25.0)
        device_score = st.slider("Device Fingerprint Risk (0=Clean, 1=Bot)", 0.0, 1.0, defaults["device_risk_score"], step=0.02)
        merchant = st.selectbox("Merchant Category", ["retail", "travel", "luxury", "gaming", "crypto"], index=["retail", "travel", "luxury", "gaming", "crypto"].index(defaults["merchant_category"]))
        employment = st.selectbox("Employment Type", ["salaried", "self_employed", "gig_economy"], index=["salaried", "self_employed", "gig_economy"].index(defaults["employment_type"]))
        credit_limit = st.number_input("Requested Credit Limit (₹)", value=defaults["requested_credit_limit"], step=25_000.0)

    applicant_data = {
        "annual_income": income,
        "cibil_score": cibil,
        "revolving_utilization": util,
        "debt_to_income": dti,
        "delinquencies_2yr": int(delinq),
        "inquiries_last_6m": int(inquiries),
        "open_credit_lines": int(open_lines),
        "credit_history_months": int(history_months),
        "transaction_velocity_24h": velocity,
        "device_risk_score": device_score,
        "merchant_category": merchant,
        "employment_type": employment,
        "requested_credit_limit": credit_limit,
    }

    if st.button("Evaluate Credit Risk & Run SHAP Attribution", type="primary"):
        prep = artifacts.get("prep")
        model = artifacts.get("model")
        calib = artifacts.get("calib")
        explainer = artifacts.get("explainer")
        cutoff = artifacts.get("optimal_threshold", 0.085)

        if prep and model and calib:
            df_app = pd.DataFrame([applicant_data])
            X_trans = prep.transform(df_app)
            raw_prob = float(model.predict_proba(X_trans)[0])
            prob = float(calib.predict_proba(X_trans)[0])
            
            # Exposure At Default calculation
            ead = credit_limit * (0.65 + 0.35 * min(1.0, util))
            exp_loss = prob * ead * 0.75

            st.markdown("---")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Calibrated Default Probability P(Default)", f"{prob:.2%}", help="Platt-scaled (sigmoid) calibrated risk probability")
            with res_col2:
                st.metric("Applied Decision Cutoff", f"{cutoff:.3f}", help="Cost-sensitive optimal threshold")
            with res_col3:
                st.metric("Expected Rupee Loss (EAD * LGD * P)", f"₹{exp_loss:,.0f}")

            # Decision banner
            is_approved = prob <= cutoff
            if is_approved:
                st.success(f"✅ **DECISION: APPROVED** — Calibrated risk ({prob:.2%}) is within the portfolio cost-optimal limit ({cutoff:.2%}).")
            else:
                st.error(f"❌ **DECISION: DECLINED** — Calibrated risk ({prob:.2%}) exceeds the cost-optimal cutoff ({cutoff:.2%}).")

            # SHAP Breakdown
            st.markdown("### 🔍 Local SHAP Feature Attribution Breakdown")
            if explainer:
                factors = explainer.explain_instance(X_trans, top_k=6)
                df_factors = pd.DataFrame(factors)
                
                fig_shap, ax = plt.subplots(figsize=(9, 3.8))
                colors = ["#ef4444" if d == "RISK_INCREASING" else "#10b981" for d in df_factors["direction"]]
                ax.barh(df_factors["feature"][::-1], df_factors["shap_value"][::-1], color=colors[::-1])
                ax.axvline(0, color="gray", linewidth=0.8)
                ax.set_xlabel("SHAP Impact on Default Log-Odds")
                ax.set_title("Key Drivers for This Applicant Decision", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_shap)

                # Regulatory Adverse Action Notice if declined
                if not is_approved:
                    st.markdown("#### 📜 Adverse-Action Style Decline Reasons (illustrative, not a compliance artifact)")
                    reasons = AdverseActionGenerator.generate_reasons(factors, max_reasons=4)
                    for r in reasons:
                        st.markdown(f"* {r}")
        else:
            st.warning("Model artifacts not yet generated. Run `risklens train` or pipeline first.")


# -------------------------------------------------------------
# 4. PSI DRIFT MONITOR
# -------------------------------------------------------------
elif menu == "PSI Drift Monitor":
    st.title("🛡️ Population Stability Index (PSI) & Macro Drift Audit")
    st.markdown("""
    Continuous data drift surveillance benchmarking **Out-Of-Time (OOT)** production applicant distributions against baseline model validation data.
    """)

    psi_plot_path = PLOTS_DIR / "psi_drift_report.png"
    if psi_plot_path.exists():
        st.image(str(psi_plot_path), caption="Live PSI Drift Audit Scorecard")
    else:
        st.info("Run `risklens drift-check` in the terminal to generate the live PSI audit chart.")

    st.markdown("""
    ### 🚦 PSI rule-of-thumb thresholds (industry convention):
    * <span class="badge-stable">PSI < 0.10</span> **STABLE:** No significant shift in credit population. Model is healthy.
    * <span class="badge-warning">0.10 ≤ PSI < 0.20</span> **WARNING:** Moderate drift detected. Increase monitoring frequency.
    * <span class="badge-severe">PSI ≥ 0.20</span> **CRITICAL DRIFT:** Severe population distribution shift. Retraining / recalibration triggered.
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 5. MATHEMATICAL FORMULATIONS
# -------------------------------------------------------------
elif menu == "Mathematical Formulations":
    st.title("📐 Mathematical & Commercial Foundations")
    st.markdown(r"""
    ### 1. Expected Rupee Loss Formulation
    Let $y_i \in \{0, 1\}$ denote the true default state, and $p_i = \mathbb{P}(y_i = 1 | \mathbf{x}_i)$ be the calibrated default probability.
    Given decision threshold $\tau$, an applicant is approved if $p_i \le \tau$ and declined if $p_i > \tau$.
    
    The portfolio expected rupee loss is:
    $$\mathcal{L}(\tau) = \sum_{i=1}^N \left[ y_i \cdot \mathbb{I}(p_i \le \tau) \cdot \text{Cost}_{FN}(i) + (1 - y_i) \cdot \mathbb{I}(p_i > \tau) \cdot \text{Cost}_{FP} + y_i \cdot \mathbb{I}(p_i > \tau) \cdot C_{\text{review}} \right]$$
    
    Where:
    $$\text{Cost}_{FN}(i) = \text{EAD}_i \times \text{LGD}_i$$
    
    The optimal cut-off $\tau^*$ satisfies:
    $$\tau^* = \arg\min_{\tau \in [0, 1]} \mathcal{L}(\tau)$$

    ---

    ### 2. Population Stability Index (PSI)
    $$\text{PSI} = \sum_{b=1}^B \left( \% \text{Actual}_b - \% \text{Expected}_b \right) \times \ln\left( \frac{\% \text{Actual}_b}{\% \text{Expected}_b} \right)$$
    
    ---

    ### 3. Probability Calibration (Platt scaling)
    With raw log-odds $s_i$, fit $a, b$ on a held-out calibration set by maximum likelihood:
    $$\hat{p}_i = \sigma(a\,s_i + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$
    Because $\sigma$ is strictly increasing, calibration changes probabilities but not the ranking, so PR-AUC and ROC-AUC are unchanged.
    """)
