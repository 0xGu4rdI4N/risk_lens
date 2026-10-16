"""RiskLens Command Line Interface (CLI)."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from risklens import __version__
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


def cmd_generate_data(args):
    """Generate high-fidelity training and out-of-time drift datasets."""
    console.print(Panel.fit("[bold cyan]RiskLens Data Generation[/bold cyan]"))
    with console.status("Generating 80k in-time and 20k out-of-time records..."):
        df_intime, df_oot = generate_train_oot_splits(
            n_train=args.train_size,
            n_oot=args.oot_size,
            random_state=args.seed,
        )
        intime_path = DATA_DIR / "credit_intime.parquet"
        oot_path = DATA_DIR / "credit_oot_drift.parquet"
        df_intime.to_parquet(intime_path, index=False)
        df_oot.to_parquet(oot_path, index=False)

    console.print(f"[green]✓ In-Time Data saved to: {intime_path} ({len(df_intime):,} rows, default rate: {df_intime['is_default'].mean():.2%})[/green]")
    console.print(f"[green]✓ Out-Of-Time Drift Data saved to: {oot_path} ({len(df_oot):,} rows, default rate: {df_oot['is_default'].mean():.2%})[/green]")


def cmd_train(args):
    """Execute end-to-end model training, calibration, and benchmark."""
    console.print(Panel.fit("[bold magenta]RiskLens Model Training & Calibration Engine[/bold magenta]"))
    
    intime_path = DATA_DIR / "credit_intime.parquet"
    if not intime_path.exists():
        console.print("[yellow]Data not found. Auto-generating synthetic cohort...[/yellow]")
        df_intime, df_oot = generate_train_oot_splits()
        df_intime.to_parquet(intime_path, index=False)
        df_oot.to_parquet(DATA_DIR / "credit_oot_drift.parquet", index=False)
    else:
        df_intime = pd.read_parquet(intime_path)

    n_total = len(df_intime)
    n_train = int(n_total * 0.70)
    n_calib = int(n_total * 0.15)
    
    df_train = df_intime.iloc[:n_train].copy()
    df_calib = df_intime.iloc[n_train:n_train + n_calib].copy()
    df_test = df_intime.iloc[n_train + n_calib:].copy()

    console.print(f"Split sizes: Train={len(df_train):,}, Calib={len(df_calib):,}, Test={len(df_test):,}")

    with console.status("Fitting preprocessor and models..."):
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

        # 1. Baseline Logistic Scorecard
        scorecard = LogisticScorecardBaseline()
        scorecard.fit(X_train, y_train)
        scorecard.save(str(MODELS_DIR / "baseline_scorecard.joblib"))

        # 2. LightGBM Classifier
        lgb_model = LightGBMRiskClassifier()
        lgb_model.fit(X_train, y_train, eval_set=[(X_calib, y_calib)])
        lgb_model.save(str(MODELS_DIR / "lightgbm_model.joblib"))

        # 3. Probability Calibrator (Isotonic)
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(lgb_model.model, X_calib, y_calib)
        calibrator.save(str(MODELS_DIR / "calibrator.joblib"))

        # 4. SHAP TreeExplainer
        explainer = RiskLensExplainer(lgb_model.model, prep.get_feature_names())
        explainer.save(str(MODELS_DIR / "explainer.joblib"))
        explainer.save_summary_plot(X_test[:500], str(PLOTS_DIR / "shap_summary.png"))

    console.print("[green]✓ Models, Calibrator, and SHAP Explainer successfully trained and persisted![/green]")

    # Predictions on test cohort
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

    # Print Comparison Table
    table = Table(title="Model Evaluation Benchmark: Rupee Loss & Accuracy Tradeoff")
    table.add_column("Model Name", style="bold cyan")
    table.add_column("PR-AUC", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("Brier Score", justify="right")
    table.add_column("Optimal Cutoff", justify="right")
    table.add_column("Loss @ Naive 0.5 (₹)", justify="right")
    table.add_column("Loss @ Optimal (₹)", justify="right", style="green")
    table.add_column("Rupee Savings (₹)", justify="right", style="bold green")
    table.add_column("Approval Rate (%)", justify="right")

    for _, row in comp_df.iterrows():
        table.add_row(
            str(row["model_name"]),
            f"{row['pr_auc']:.4f}",
            f"{row['roc_auc']:.4f}",
            f"{row['brier_score']:.4f}",
            f"{row['optimal_threshold']:.3f}",
            f"₹{row['loss_at_naive_05_rupees']:,.0f}",
            f"₹{row['loss_at_optimal_threshold_rupees']:,.0f}",
            f"₹{row['rupee_savings']:,.0f}",
            f"{row['approval_rate_optimal_pct']:.1f}%",
        )
    console.print(table)

    # Optimize and save the operational threshold
    opt = ThresholdOptimizer()
    opt_res = opt.optimize_threshold(y_test, calib_probs, ead=ead_test, lgd=lgd_test)
    (MODELS_DIR / "optimal_threshold.txt").write_text(str(opt_res["optimal_threshold"]))

    # Save Frontier Data
    frontier_engine = ApprovalFrontierEngine()
    df_frontier = frontier_engine.generate_frontier_curve(y_test, calib_probs, ead=ead_test, lgd=lgd_test)
    df_frontier.to_parquet(REPORTS_DIR / "frontier_curve.parquet", index=False)

    # Compute +4% approval lift theorem demonstration
    lift_demo = frontier_engine.compute_approval_expansion(
        y_true=y_test,
        baseline_prob=base_probs,
        risklens_prob=calib_probs,
        ead=ead_test,
        lgd=lgd_test,
        baseline_approval_rate=0.78,
        target_lift_pct=4.0,
    )

    console.print(Panel(
        f"[bold gold1]Approval Expansion Frontier Result (+4% Target):[/bold gold1]\n"
        f"• Baseline Approval Volume: [cyan]{lift_demo['baseline']['approval_rate_pct']:.1f}%[/cyan] "
        f"(Approved Default Rate: {lift_demo['baseline']['approved_default_rate_pct']:.2f}%)\n"
        f"• RiskLens Expanded Approval Volume: [bold green]{lift_demo['risklens_expansion']['approval_rate_pct']:.1f}% (+4.0% Lift)[/bold green] "
        f"(Approved Default Rate: {lift_demo['risklens_expansion']['approved_default_rate_pct']:.2f}%)\n"
        f"• Additional Accounts Approved: [bold green]+{lift_demo['lift_metrics']['additional_approved_accounts']:,}[/bold green]\n"
        f"• Incremental Annual Revenue: [bold green]₹{lift_demo['lift_metrics']['incremental_annual_revenue_rupees']:,.0f}[/bold green]\n"
        f"• Net Commercial Financial Benefit: [bold green]₹{lift_demo['lift_metrics']['net_financial_benefit_rupees']:,.0f}[/bold green]\n"
        f"• Risk Neutrality Maintained: [bold green]{lift_demo['lift_metrics']['risk_neutrality_achieved']}[/bold green]",
        title="Commercial Underwriting Frontier"
    ))


def cmd_drift_check(args):
    """Run Population Stability Index (PSI) drift audit between baseline and production cohort."""
    console.print(Panel.fit("[bold blue]RiskLens PSI Data Drift Audit[/bold blue]"))
    
    intime_path = DATA_DIR / "credit_intime.parquet"
    oot_path = DATA_DIR / "credit_oot_drift.parquet"
    
    if not intime_path.exists() or not oot_path.exists():
        console.print("[red]Datasets not found. Run 'risklens generate-data' first.[/red]")
        sys.exit(1)

    df_base = pd.read_parquet(intime_path)
    df_oot = pd.read_parquet(oot_path)

    prep_path = MODELS_DIR / "preprocessor.joblib"
    calib_path = MODELS_DIR / "calibrator.joblib"
    
    base_scores, oot_scores = None, None
    if prep_path.exists() and calib_path.exists():
        prep = CreditRiskPreprocessor.load(str(prep_path))
        calib = ProbabilityCalibrator.load(str(calib_path))
        base_scores = calib.predict_proba(prep.transform(df_base.sample(min(10000, len(df_base)), random_state=42)))
        oot_scores = calib.predict_proba(prep.transform(df_oot.sample(min(10000, len(df_oot)), random_state=42)))

    detector = DriftDetector()
    audit = detector.run_drift_audit(
        baseline_df=df_base,
        current_df=df_oot,
        baseline_scores=base_scores,
        current_scores=oot_scores,
    )

    plot_path = str(PLOTS_DIR / "psi_drift_report.png")
    detector.plot_psi_summary(audit, plot_path)

    table = Table(title=f"Population Stability Index (Overall: {audit['overall_status']})")
    table.add_column("Risk Dimension / Feature", style="bold")
    table.add_column("PSI Value", justify="right")
    table.add_column("Regulatory Status", justify="center")

    if audit["score_status"] != "NOT_EVALUATED":
        table.add_row(
            "[bold cyan]Model Score (Calibrated P(Default))[/bold cyan]",
            f"{audit['score_psi']:.4f}",
            f"[bold]{audit['score_status']}[/bold]"
        )

    for item in audit["features"]:
        color = "green" if item["status"] == "STABLE" else ("yellow" if item["status"] == "WARNING" else "red")
        table.add_row(
            item["feature"],
            f"{item['psi']:.4f}",
            f"[{color}]{item['status']}[/{color}]"
        )
    console.print(table)
    console.print(f"[green]✓ Visual PSI audit chart saved to: {plot_path}[/green]")


def cmd_serve(args):
    """Launch FastAPI underwriting backend."""
    import uvicorn
    console.print(f"[bold green]Starting RiskLens API on port {args.port}...[/bold green]")
    uvicorn.run("risklens.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_dashboard(args):
    """Launch Streamlit web application."""
    import subprocess
    dash_path = Path(__file__).parent / "dashboard" / "app.py"
    console.print(f"[bold green]Starting Streamlit Dashboard on port {args.port}...[/bold green]")
    subprocess.run(["streamlit", "run", str(dash_path), "--server.port", str(args.port)])


def main():
    parser = argparse.ArgumentParser(
        prog="risklens",
        description="RiskLens: Cost-Sensitive Credit Default & Fraud Detection Engine"
    )
    parser.add_argument("--version", action="version", version=f"RiskLens v{__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-data
    p_gen = subparsers.add_parser("generate-data", help="Generate synthetic credit default & fraud datasets")
    p_gen.add_argument("--train-size", type=int, default=80_000)
    p_gen.add_argument("--oot-size", type=int, default=20_000)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=cmd_generate_data)

    # train
    p_train = subparsers.add_parser("train", help="Train LightGBM, calibrate probabilities, and optimize threshold")
    p_train.set_defaults(func=cmd_train)

    # drift-check
    p_drift = subparsers.add_parser("drift-check", help="Compute PSI metrics across features and model score")
    p_drift.set_defaults(func=cmd_drift_check)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit interactive web dashboard")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
