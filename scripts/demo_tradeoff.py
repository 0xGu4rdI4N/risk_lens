"""Print the measured approval lift at matched risk from the saved pipeline outputs."""

import sys
from pathlib import Path
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from risklens.config import REPORTS_DIR

console = Console()


def show_approval_tradeoff():
    lift_path = REPORTS_DIR / "approval_lift_metrics.json"
    benchmark_path = REPORTS_DIR / "model_benchmark.json"

    if not lift_path.exists() or not benchmark_path.exists():
        console.print("[yellow]Running full pipeline first to generate metrics...[/yellow]")
        from scripts.run_pipeline import run_pipeline
        run_pipeline()

    with open(lift_path) as f:
        lift_data = json.load(f)
    with open(benchmark_path) as f:
        bench_data = json.load(f)

    console.print(Panel.fit("[bold green]RiskLens: Approval Lift at Matched Risk (synthetic data)[/bold green]"))

    t1 = Table(title="Model Accuracy vs Commercial Rupee Loss Comparison")
    t1.add_column("Model Architecture", style="cyan")
    t1.add_column("PR-AUC", justify="right")
    t1.add_column("ROC-AUC", justify="right")
    t1.add_column("Brier Score", justify="right")
    t1.add_column("Loss @ Naive 0.5", justify="right")
    t1.add_column("Loss @ Optimal τ*", justify="right", style="green")
    t1.add_column("Rupee Savings", justify="right", style="bold green")
    for row in bench_data:
        t1.add_row(
            row["model_name"], f"{row['pr_auc']:.4f}", f"{row['roc_auc']:.4f}", f"{row['brier_score']:.4f}",
            f"₹{row['loss_at_naive_05_rupees']:,.0f}", f"₹{row['loss_at_optimal_threshold_rupees']:,.0f}",
            f"₹{row['rupee_savings']:,.0f} ({row['pct_savings']:.1f}%)",
        )
    console.print(t1)

    base, rl, meta, bs = lift_data["baseline"], lift_data["risklens_expansion"], lift_data["lift_metrics"], lift_data["bootstrap"]
    t2 = Table(title="Baseline Scorecard vs RiskLens at matched approved-default rate")
    t2.add_column("Portfolio Dimension", style="bold")
    t2.add_column("Baseline Scorecard", justify="right", style="yellow")
    t2.add_column("RiskLens", justify="right", style="bold green")
    t2.add_column("Delta", justify="right", style="cyan")
    t2.add_row("Approval Volume", f"{base['approval_rate_pct']:.2f}%", f"{rl['approval_rate_pct']:.2f}%",
               f"{meta['approval_lift_pct_points']:+.2f} pts")
    t2.add_row("Approved Accounts", f"{base['approval_count']:,}", f"{rl['approval_count']:,}",
               f"{meta['additional_approved_accounts']:+,}")
    t2.add_row("Approved Default Rate", f"{base['approved_default_rate_pct']:.3f}%", f"{rl['approved_default_rate_pct']:.3f}%",
               f"{meta['approved_default_rate_delta_pct_points']:+.3f} pts")
    t2.add_row("95% bootstrap CI on lift", "", "",
               f"[{bs['lift_pct_points_ci95'][0]:+.2f}, {bs['lift_pct_points_ci95'][1]:+.2f}] pts")
    t2.add_row("Net benefit (assumed margin)", "", f"₹{meta['net_financial_benefit_rupees']:,.0f}",
               f"₹{meta['assumed_annual_margin_per_good_account_rupees']:,.0f}/good acct/yr")
    console.print(t2)


if __name__ == "__main__":
    show_approval_tradeoff()
