"""Rupee-loss evaluation as Spark aggregations (same definition as risklens.cost.loss_matrix)."""
from typing import Dict, Optional, Sequence

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from risklens.config import DEFAULT_COST_CONFIG, CostMatrixConfig


def _loss_aggs(prob: str, threshold, config: CostMatrixConfig):
    approved = F.col(prob) <= threshold
    y = F.col("is_default") == 1
    fn_cost = F.col("exposure_at_default") * F.col("loss_given_default")
    return [
        F.count(F.lit(1)).alias("total_accounts"),
        F.sum(F.when(y & approved, fn_cost).otherwise(0.0)).alias("fn_loss_rupees"),
        F.sum(F.when(~y & ~approved, 1).otherwise(0)).alias("count_fp"),
        F.sum(F.when(y & ~approved, 1).otherwise(0)).alias("count_tp"),
        F.sum(F.when(y & approved, 1).otherwise(0)).alias("count_fn"),
        F.sum(F.when(approved, 1).otherwise(0)).alias("approval_count"),
    ]


def _finish(row: Dict, config: CostMatrixConfig) -> Dict:
    fp_loss = row["count_fp"] * config.default_fp_cost
    tp_friction = row["count_tp"] * config.manual_review_cost
    n, approved = row["total_accounts"], row["approval_count"]
    row.update({
        "fp_loss_rupees": float(fp_loss), "tp_friction_rupees": float(tp_friction),
        "total_loss_rupees": float(row["fn_loss_rupees"] + fp_loss + tp_friction),
        "approval_rate": approved / max(n, 1),
        "approved_default_rate": row["count_fn"] / max(approved, 1),
    })
    return row


def portfolio_loss(scored: DataFrame, threshold: float, config: Optional[CostMatrixConfig] = None) -> Dict:
    config = config or DEFAULT_COST_CONFIG
    row = scored.agg(*_loss_aggs("calibrated_default_prob", F.lit(float(threshold)), config)).first().asDict()
    row["fn_loss_rupees"] = float(row["fn_loss_rupees"] or 0.0)
    for k in ("count_fp", "count_tp", "count_fn", "approval_count"):
        row[k] = int(row[k] or 0)
    row["threshold"] = float(threshold)
    return _finish(row, config)


def loss_curve(scored: DataFrame, thresholds: Optional[Sequence[float]] = None,
               config: Optional[CostMatrixConfig] = None) -> DataFrame:
    """One distributed pass: cross-join with the threshold grid, aggregate per threshold."""
    config = config or DEFAULT_COST_CONFIG
    spark = SparkSession.getActiveSession()
    thresholds = list(np.linspace(0.005, 0.70, 250)) if thresholds is None else list(thresholds)
    grid = spark.createDataFrame([(float(t),) for t in thresholds], ["threshold"])
    agg = (scored.crossJoin(F.broadcast(grid))
           .groupBy("threshold")
           .agg(*_loss_aggs("calibrated_default_prob", F.col("threshold"), config)))
    return (agg.withColumn("total_loss_rupees",
                           F.col("fn_loss_rupees") + F.col("count_fp") * F.lit(config.default_fp_cost)
                           + F.col("count_tp") * F.lit(config.manual_review_cost))
            .withColumn("approval_rate", F.col("approval_count") / F.col("total_accounts"))
            .orderBy("threshold"))


def optimal_threshold(curve: DataFrame) -> Dict:
    best = curve.orderBy("total_loss_rupees", "threshold").first()
    return {"optimal_threshold": float(best["threshold"]), "min_loss_rupees": float(best["total_loss_rupees"]),
            "approval_rate": float(best["approval_rate"])}
