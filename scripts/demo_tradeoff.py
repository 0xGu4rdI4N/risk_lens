"""Demonstration script for the +4% Credit Approval Lift Theorem."""

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

    console.print(Panel.fit("[bold green]RiskLens: The +4% Approval Expansion Proof[/bold green]"))

    # Table 1: Model Comparison
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
            row["model_name"],
            f"{row['pr_auc']:.4f}",
            f"{row['roc_auc']:.4f}",
            f"{row['brier_score']:.4f}",
            f"₹{row['loss_at_naive_05_rupees']:,.0f}",
            f"₹{row['loss_at_optimal_threshold_rupees']:,.0f}",
            f"₹{row['rupee_savings']:,.0f} ({row['pct_savings']:.1f}%)",
        )
    console.print(t1)

    # Table 2: The Approval Expansion Frontier
    t2 = Table(title="Quantified Tradeoff: Baseline Scorecard vs. RiskLens +4% Expansion")
    t2.add_column("Portfolio Dimension", style="bold")
    t2.add_column("Baseline Scorecard", justify="right", style="yellow")
    t2.add_column("RiskLens +4% Lift", justify="right", style="bold green")
    t2.add_column("Variance / Delta", justify="right", style="cyan")

    base = lift_data["baseline"]
    rl = lift_data["risklens_expansion"]
    meta = lift_data["lift_metrics"]

    t2.add_row("Approval Volume", f"{base['approval_rate_pct']:.1f}%", f"{rl['approval_rate_pct']:.1f}%", f"+{meta['approval_lift_pct_points']:.1f}% Points")
    t2.add_row("Approved Accounts", f"{base['approval_count']:,}", f"{rl['approval_count']:,}", f"+{meta['additional_approved_accounts']:,}")
    t2.add_row("Approved Default Rate", f"{base['approved_default_rate_pct']:.2f}%", f"{rl['approved_default_rate_pct']:.2f}%", f"{rl['approved_default_rate_pct'] - base['approved_default_rate_pct']:+.2f}% (Risk Neutral)")
    t2.add_row("Incremental Margin Earned", "—", f"₹{meta['incremental_annual_revenue_rupees']:,.0f}", f"+₹{meta['incremental_annual_revenue_rupees']:,.0f}")
    t2.add_row("Net Financial Benefit", "—", f"₹{meta['net_financial_benefit_rupees']:,.0f}", f"+₹{meta['net_financial_benefit_rupees']:,.0f}")

    console.print(t2)


if __name__ == "__main__":
    show_approval_tradeoff()
