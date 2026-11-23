"""Distributed batch scoring: apply the saved preprocessor + calibrated LightGBM to a Spark DataFrame."""
from pathlib import Path
from typing import Iterator

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

SCORED_SCHEMA = T.StructType([
    T.StructField("applicant_id", T.StringType()),
    T.StructField("is_default", T.IntegerType()),
    T.StructField("exposure_at_default", T.DoubleType()),
    T.StructField("loss_given_default", T.DoubleType()),
    T.StructField("calibrated_default_prob", T.DoubleType()),
])

_CACHE: dict = {}


def _load(models_dir: str):
    """Load artifacts once per Python worker process, not once per batch."""
    if models_dir not in _CACHE:
        from risklens.data.preprocessor import CreditRiskPreprocessor
        from risklens.models.calibrator import ProbabilityCalibrator
        d = Path(models_dir)
        _CACHE[models_dir] = (CreditRiskPreprocessor.load(str(d / "preprocessor.joblib")),
                              ProbabilityCalibrator.load(str(d / "calibrator.joblib")))
    return _CACHE[models_dir]


def make_scorer(models_dir: str):
    """Return a mapInPandas function that scores each Arrow batch with the saved models."""
    models_dir = str(models_dir)

    def score_batches(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        prep, calib = _load(models_dir)
        for pdf in batches:
            prob = calib.predict_proba(prep.transform(pdf))
            yield pd.DataFrame({
                "applicant_id": pdf["applicant_id"].astype(str).values,
                "is_default": pdf["is_default"].astype("int32").values,
                "exposure_at_default": pdf["exposure_at_default"].astype("float64").values,
                "loss_given_default": pdf["loss_given_default"].astype("float64").values,
                "calibrated_default_prob": prob.astype("float64"),
            })

    return score_batches


def score_dataframe(df: DataFrame, models_dir: str, threshold: float) -> DataFrame:
    """Score every applicant and attach the APPROVED/DECLINED decision at `threshold`."""
    scored = df.mapInPandas(make_scorer(models_dir), schema=SCORED_SCHEMA)
    return scored.withColumn(
        "decision", F.when(F.col("calibrated_default_prob") <= F.lit(threshold), "APPROVED").otherwise("DECLINED"))
