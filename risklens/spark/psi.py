"""Population Stability Index computed with Spark (quantile bins on the baseline, grouped counts)."""
from typing import Dict, List, Optional

import numpy as np
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from risklens.config import DEFAULT_DRIFT_CONFIG, DriftConfig


def _status(psi: float, cfg: DriftConfig) -> str:
    return "STABLE" if psi < cfg.stable_threshold else "WARNING" if psi < cfg.warning_threshold else "SEVERE_DRIFT"


def _psi_from_counts(exp_counts: np.ndarray, act_counts: np.ndarray, cfg: DriftConfig) -> float:
    exp_pct = np.clip(exp_counts / max(exp_counts.sum(), 1), cfg.epsilon, 1.0)
    act_pct = np.clip(act_counts / max(act_counts.sum(), 1), cfg.epsilon, 1.0)
    exp_pct, act_pct = exp_pct / exp_pct.sum(), act_pct / act_pct.sum()
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def bin_counts(df: DataFrame, col: str, splits: List[float]) -> np.ndarray:
    """Count rows per bucket with one groupBy; empty buckets are filled with zero."""
    b = Bucketizer(splits=splits, inputCol=col, outputCol="_bin", handleInvalid="skip")
    rows = b.transform(df.select(col).dropna()).groupBy("_bin").count().collect()
    counts = np.zeros(len(splits) - 1)
    for r in rows:
        counts[int(r["_bin"])] = r["count"]
    return counts


def continuous_psi(baseline: DataFrame, current: DataFrame, col: str, num_bins: Optional[int] = None,
                   config: Optional[DriftConfig] = None) -> Dict:
    cfg = config or DEFAULT_DRIFT_CONFIG
    n_bins = num_bins or cfg.num_bins
    quantiles = [i / n_bins for i in range(n_bins + 1)]
    edges = np.unique(baseline.select(col).dropna().stat.approxQuantile(col, quantiles, 0.0))
    if len(edges) < 3:
        lo, hi = baseline.agg(F.min(col), F.max(col)).first()
        edges = np.linspace(lo, hi, n_bins + 1)
    splits = [-float("inf")] + [float(e) for e in edges[1:-1]] + [float("inf")]
    exp_c, act_c = bin_counts(baseline, col, splits), bin_counts(current, col, splits)
    psi = _psi_from_counts(exp_c, act_c, cfg)
    return {"feature": col, "type": "numerical", "psi": psi, "status": _status(psi, cfg),
            "breakpoints": splits, "expected_counts": exp_c.tolist(), "actual_counts": act_c.tolist()}


def categorical_psi(baseline: DataFrame, current: DataFrame, col: str,
                    config: Optional[DriftConfig] = None) -> Dict:
    cfg = config or DEFAULT_DRIFT_CONFIG
    e = {r[col]: r["count"] for r in baseline.select(col).dropna().groupBy(col).count().collect()}
    a = {r[col]: r["count"] for r in current.select(col).dropna().groupBy(col).count().collect()}
    cats = sorted(set(e) | set(a))
    exp_c = np.array([e.get(c, 0) for c in cats], dtype=float)
    act_c = np.array([a.get(c, 0) for c in cats], dtype=float)
    psi = _psi_from_counts(exp_c, act_c, cfg)
    return {"feature": col, "type": "categorical", "psi": psi, "status": _status(psi, cfg), "categories": cats}
