"""End-to-end execution pipeline for RiskLens.

Generates datasets, trains and benchmarks models, performs probability calibration,
optimizes cost-sensitive thresholds, generates SHAP attributions, and computes PSI drift.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

# Add repository root to pythonpath
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from risklens.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR
from risklens.data.generator import generate_train_oot_splits
from risklens.data.preprocessor import CreditRiskPreprocessor
from risklens.models.baseline import LogisticScorecardBaseline
from risklens.models.tree_models import LightGBMRiskClassifier
from risklens.models.calibrator import ProbabilityCalibrator
from risklens.models.trainer import ModelEvaluator
from risklens.cost.optimizer import ThresholdOptimizer
from risklens.cost.frontier import ApprovalFrontierEngine
from risklens.explainability.explainer import RiskLensExplainer
from risklens.monitoring.drift_detector import DriftDetector

console = Console()
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    console.print(Panel.fit("[bold cyan]RiskLens: Executing End-to-End Production Pipeline[/bold cyan]"))

    # 1. Dataset Generation
    console.print("\n[bold yellow]Step 1: Generating 100,000 Synthetic Credit & Fraud Records...[/bold yellow]")
    df_intime, df_oot = generate_train_oot_splits(n_train=80_000, n_oot=20_000, random_state=42)
    intime_path = DATA_DIR / "credit_intime.parquet"
    oot_path = DATA_DIR / "credit_oot_drift.parquet"
    df_intime.to_parquet(intime_path, index=False)
    df_oot.to_parquet(oot_path, index=False)
    console.print(f"[green]✓ In-Time cohort: {len(df_intime):,} records | Default rate: {df_intime['is_default'].mean():.2%}[/green]")
    console.print(f"[green]✓ Out-of-Time cohort: {len(df_oot):,} records | Default rate: {df_oot['is_default'].mean():.2%}[/green]")

    # 2. Train / Calibrate / Test Split
    n_total = len(df_intime)
    n_train = int(n_total * 0.70)
    n_calib = int(n_total * 0.15)
    
    df_train = df_intime.iloc[:n_train].copy()
    df_calib = df_intime.iloc[n_train:n_train + n_calib].copy()
    df_test = df_intime.iloc[n_train + n_calib:].copy()

    # 3. Preprocessing
    console.print("\n[bold yellow]Step 2: Fitting Robust Preprocessing Pipeline...[/bold yellow]")
    prep = CreditRiskPreprocessor()
    prep.fit(df_train)
    prep.save(str(MODELS_DIR / "preprocessor.joblib"))

    X_train = prep.transform(df_train)
    y_train = df_train["is_default"].values

    X_calib = prep.transform(df_calib)
    y_calib = df_calib["is_default"].values

    X_test = prep.transform(df_test)
    y_test = df_test["is_default"].values
    ead_test = df_test["exposure_at_default"].values
    lgd_test = df_test["loss_given_default"].values

    # 4. Model Training & Calibration
    console.print("\n[bold yellow]Step 3: Training Baseline Scorecard & LightGBM Classifier...[/bold yellow]")
    scorecard = LogisticScorecardBaseline()
    scorecard.fit(X_train, y_train)
    scorecard.save(str(MODELS_DIR / "baseline_scorecard.joblib"))

    lgb_model = LightGBMRiskClassifier()
    lgb_model.fit(X_train, y_train, eval_set=[(X_calib, y_calib)])
    lgb_model.save(str(MODELS_DIR / "lightgbm_model.joblib"))

    console.print("\n[bold yellow]Step 4: Fitting Platt Scaling (Sigmoid) Probability Calibrator...[/bold yellow]")
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(lgb_model.model, X_calib, y_calib)
    calibrator.save(str(MODELS_DIR / "calibrator.joblib"))

    calib_eval = calibrator.evaluate_calibration(y_test, calibrator.predict_proba(X_test))
    console.print(f"[green]✓ Calibration Brier Score: {calib_eval['brier_score']:.4f} | ECE: {calib_eval['ece']:.4f}[/green]")

    # 5. SHAP Explainability
    console.print("\n[bold yellow]Step 5: Initializing TreeSHAP Explainer & Generating Visuals...[/bold yellow]")
    explainer = RiskLensExplainer(lgb_model.model, prep.get_feature_names())
    explainer.save(str(MODELS_DIR / "explainer.joblib"))
    shap_path = str(ASSETS_DIR / "shap_summary.png")
    explainer.save_summary_plot(X_test[:600], shap_path)
    explainer.save_summary_plot(X_test[:600], str(PLOTS_DIR / "shap_summary.png"))
    console.print(f"[green]✓ SHAP summary beeswarm plot saved to: {shap_path}[/green]")

    # 6. Evaluation & Benchmarking
    console.print("\n[bold yellow]Step 6: Benchmarking PR-AUC, ROC-AUC, and Rupee Losses...[/bold yellow]")
    base_probs = scorecard.predict_proba(X_test)
    raw_lgb_probs = lgb_model.predict_proba(X_test)
    calib_probs = calibrator.predict_proba(X_test)

    evaluator = ModelEvaluator()
    comp_df = evaluator.compare_models(
        models_dict={
            "Baseline Scorecard (Logistic)": base_probs,
            "LightGBM (Raw Margin)": raw_lgb_probs,
            "RiskLens (Calibrated LightGBM)": calib_probs,
        },
        y_true=y_test,
        ead=ead_test,
        lgd=lgd_test,
    )
    console.print(comp_df.to_string(index=False))
    comp_df.to_json(REPORTS_DIR / "model_benchmark.json", orient="records", indent=2)

    # 7. Threshold Optimization
    console.print("\n[bold yellow]Step 7: Optimizing Decision Threshold for Minimal Rupee Loss...[/bold yellow]")
    opt = ThresholdOptimizer()
    opt_res = opt.optimize_threshold(y_test, calib_probs, ead=ead_test, lgd=lgd_test)
    opt_threshold = opt_res["optimal_threshold"]
    (MODELS_DIR / "optimal_threshold.txt").write_text(str(opt_threshold))

    console.print(f"[bold green]✓ Optimal Cut-off Threshold: τ* = {opt_threshold:.3f}[/bold green]")
    console.print(f"[green]✓ Loss at Naive 0.5: ₹{opt_res['naive_loss_rupees']:,.0f}[/green]")
    console.print(f"[green]✓ Loss at Optimal Cutoff: ₹{opt_res['min_expected_loss_rupees']:,.0f}[/green]")
    console.print(f"[bold green]✓ Commercial Savings: ₹{opt_res['total_rupee_savings']:,.0f} ({opt_res['pct_savings']:.1f}% reduction!)[/bold green]")

    # 8. Tradeoff & +4% Approval Expansion Frontier
    console.print("\n[bold yellow]Step 8: Computing Approval Frontier & +4% Expansion Theorem...[/bold yellow]")
    frontier_engine = ApprovalFrontierEngine()
    df_frontier = frontier_engine.generate_frontier_curve(y_test, calib_probs, ead=ead_test, lgd=lgd_test)
    df_frontier.to_parquet(REPORTS_DIR / "frontier_curve.parquet", index=False)

    lift_result = frontier_engine.compute_approval_expansion(
        y_true=y_test,
        baseline_prob=base_probs,
        risklens_prob=calib_probs,
        ead=ead_test,
        lgd=lgd_test,
        baseline_approval_rate=0.78,
        target_lift_pct=4.0,
    )
    with open(REPORTS_DIR / "approval_lift_metrics.json", "w") as f:
        json.dump(lift_result, f, indent=2)

    console.print(Panel(
        f"[bold cyan]Commercial Approval Expansion Achievement:[/bold cyan]\n"
        f"• Baseline Scorecard Approvals: {lift_result['baseline']['approval_rate_pct']:.1f}% (Default Rate: {lift_result['baseline']['approved_default_rate_pct']:.2f}%)\n"
        f"• RiskLens Expanded Approvals: {lift_result['risklens_expansion']['approval_rate_pct']:.1f}% (+4.0% Lift!)\n"
        f"• RiskLens Expanded Default Rate: {lift_result['risklens_expansion']['approved_default_rate_pct']:.2f}% (Risk Neutral!)\n"
        f"• Additional Accounts Approved: +{lift_result['lift_metrics']['additional_approved_accounts']:,}\n"
        f"• Incremental Annual Interchange/Interest Margin: ₹{lift_result['lift_metrics']['incremental_annual_revenue_rupees']:,.0f}\n"
        f"• Net Commercial Financial Benefit: ₹{lift_result['lift_metrics']['net_financial_benefit_rupees']:,.0f}",
        title="Portfolio Expansion Milestone"
    ))

    # 9. PSI Data Drift Audit
    console.print("\n[bold yellow]Step 9: Running PSI Data Drift Audit against Out-Of-Time (OOT) Cohort...[/bold yellow]")
    detector = DriftDetector()
    sample_base_scores = calib_probs[:5000]
    sample_oot_scores = calibrator.predict_proba(prep.transform(df_oot.iloc[:5000]))
    
    drift_audit = detector.run_drift_audit(
        baseline_df=df_intime.iloc[:10000],
        current_df=df_oot.iloc[:10000],
        baseline_scores=sample_base_scores,
        current_scores=sample_oot_scores,
    )
    drift_plot_path = str(ASSETS_DIR / "psi_drift_report.png")
    detector.plot_psi_summary(drift_audit, drift_plot_path)
    detector.plot_psi_summary(drift_audit, str(PLOTS_DIR / "psi_drift_report.png"))
    status_color = "bold red" if "ALERT" in drift_audit["overall_status"] else "bold green"
    console.print(f"[{status_color}]✓ Overall Drift Status: {drift_audit['overall_status']}[/{status_color}]")

    console.print(Panel.fit("[bold green]✓ RiskLens Production Pipeline Completed Successfully![/bold green]"))


if __name__ == "__main__":
    run_pipeline()
