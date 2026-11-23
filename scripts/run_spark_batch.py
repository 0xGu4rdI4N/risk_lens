"""Spark batch job: score applicants, evaluate rupee loss, and run PSI drift checks.

    python scripts/run_spark_batch.py                 # score the in-time and out-of-time cohorts
    python scripts/run_spark_batch.py --replicate 20  # replicate rows 20x to exercise partitioning
    python scripts/run_spark_batch.py --compare-pandas

Requires the artifacts written by scripts/run_pipeline.py and a JDK (see README).
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pyspark.sql import functions as F

from risklens.config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from risklens.data.preprocessor import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from risklens.spark.loss import loss_curve, optimal_threshold, portfolio_loss
from risklens.spark.psi import categorical_psi, continuous_psi
from risklens.spark.scoring import score_dataframe
from risklens.spark.session import get_spark


def read_cohort(spark, name: str, replicate: int):
    df = spark.read.parquet(str(DATA_DIR / name))
    if replicate > 1:
        df = (df.crossJoin(spark.range(replicate).withColumnRenamed("id", "_rep"))
              .withColumn("applicant_id", F.concat_ws("-", "applicant_id", F.col("_rep").cast("string")))
              .drop("_rep"))
    return df


def run(replicate: int, out_dir: Path, shuffle_partitions: int) -> dict:
    spark = get_spark("risklens-batch", shuffle_partitions=shuffle_partitions)
    threshold = float((MODELS_DIR / "optimal_threshold.txt").read_text())
    base_df = read_cohort(spark, "credit_intime.parquet", replicate).repartition(shuffle_partitions).cache()
    oot_df = read_cohort(spark, "credit_oot_drift.parquet", replicate).repartition(shuffle_partitions).cache()
    n_base, n_oot = base_df.count(), oot_df.count()

    report = {"rows": {"in_time": n_base, "oot": n_oot}, "replicate": replicate, "spark_version": spark.version,
              "threshold_from_pipeline": threshold, "timings_s": {}}

    t0 = time.perf_counter()
    scored = {}
    for name, df in (("in_time", base_df), ("oot", oot_df)):
        s = score_dataframe(df, str(MODELS_DIR), threshold).cache()
        out = DATA_DIR / "spark_scored" / name   # parquet output stays in the git-ignored data/ folder
        s.write.mode("overwrite").partitionBy("decision").parquet(str(out))
        scored[name] = s
        report[f"loss_{name}"] = portfolio_loss(s, threshold)
    report["timings_s"]["score_and_write"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    curve = loss_curve(scored["in_time"]).cache()
    report["optimal_threshold_in_time"] = optimal_threshold(curve)
    report["timings_s"]["loss_curve_250_thresholds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feats = [continuous_psi(base_df, oot_df, c) for c in NUMERICAL_FEATURES]
    feats += [categorical_psi(base_df, oot_df, c) for c in CATEGORICAL_FEATURES]
    s_psi = continuous_psi(scored["in_time"], scored["oot"], "calibrated_default_prob")
    report["psi"] = {"score": {k: s_psi[k] for k in ("psi", "status")},
                     "features": sorted(({k: f[k] for k in ("feature", "psi", "status")} for f in feats),
                                        key=lambda x: -x["psi"])}
    report["timings_s"]["psi_all_features"] = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spark_batch_report.json").write_text(json.dumps(report, indent=2))
    spark.stop()
    return report


def compare_pandas(replicate: int) -> dict:
    """Time the same scoring + loss work in single-process pandas for an honest comparison."""
    import pandas as pd

    from risklens.cost.loss_matrix import RupeeLossCalculator
    from risklens.data.preprocessor import CreditRiskPreprocessor
    from risklens.models.calibrator import ProbabilityCalibrator

    t0 = time.perf_counter()
    prep = CreditRiskPreprocessor.load(str(MODELS_DIR / "preprocessor.joblib"))
    calib = ProbabilityCalibrator.load(str(MODELS_DIR / "calibrator.joblib"))
    threshold = float((MODELS_DIR / "optimal_threshold.txt").read_text())
    df = pd.concat([pd.read_parquet(DATA_DIR / "credit_intime.parquet")] * replicate, ignore_index=True)
    prob = calib.predict_proba(prep.transform(df))
    RupeeLossCalculator().calculate_portfolio_loss(
        df["is_default"].values, prob, threshold,
        ead=df["exposure_at_default"].values, lgd=df["loss_given_default"].values)
    return {"rows": len(df), "seconds": time.perf_counter() - t0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--replicate", type=int, default=1, help="repeat every row N times (scale test)")
    p.add_argument("--partitions", type=int, default=8)
    p.add_argument("--out", type=Path, default=REPORTS_DIR)
    p.add_argument("--compare-pandas", action="store_true")
    a = p.parse_args()
    rep = run(a.replicate, a.out, a.partitions)
    if a.compare_pandas:
        rep["pandas_in_time_score_and_loss"] = compare_pandas(a.replicate)
        (a.out / "spark_batch_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in rep if k != "psi"}, indent=2))
    print("\nTop PSI:", json.dumps(rep["psi"]["features"][:4], indent=1), "\nScore PSI:", rep["psi"]["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
