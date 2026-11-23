"""Parity tests: the Spark implementations must agree with the pandas/numpy ones."""
import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from risklens.spark.session import get_spark, java_available

pytestmark = pytest.mark.skipif(not java_available(), reason="no JDK available for Spark")

from risklens.cost.loss_matrix import RupeeLossCalculator
from risklens.cost.optimizer import ThresholdOptimizer
from risklens.data.generator import generate_credit_dataset
from risklens.data.preprocessor import CreditRiskPreprocessor
from risklens.models.calibrator import ProbabilityCalibrator
from risklens.models.tree_models import LightGBMRiskClassifier
from risklens.monitoring.psi import PSICalculator
from risklens.spark.loss import loss_curve, optimal_threshold, portfolio_loss
from risklens.spark.psi import bin_counts, categorical_psi, continuous_psi
from risklens.spark.scoring import score_dataframe


@pytest.fixture(scope="module")
def spark():
    s = get_spark("risklens-tests", master="local[2]", shuffle_partitions=2)
    yield s
    s.stop()


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df = generate_credit_dataset(n_samples=4000, random_state=11)
    prep = CreditRiskPreprocessor().fit(df.iloc[:2500])
    X = prep.transform(df)
    lgb = LightGBMRiskClassifier().fit(X[:2500], df["is_default"].values[:2500])
    calib = ProbabilityCalibrator().fit(lgb.model, X[2500:3200], df["is_default"].values[2500:3200])
    d = tmp_path_factory.mktemp("models")
    prep.save(str(d / "preprocessor.joblib"))
    calib.save(str(d / "calibrator.joblib"))
    test = df.iloc[3200:].reset_index(drop=True)
    return str(d), prep, calib, test


def test_spark_scores_match_pandas(spark, trained):
    models_dir, prep, calib, test = trained
    sdf = spark.createDataFrame(test).repartition(3)
    out = score_dataframe(sdf, models_dir, threshold=0.05).toPandas().set_index("applicant_id")
    expected = pd.Series(calib.predict_proba(prep.transform(test)), index=test["applicant_id"])
    np.testing.assert_allclose(out.loc[expected.index, "calibrated_default_prob"].values, expected.values, atol=1e-9)
    assert set(out["decision"]) <= {"APPROVED", "DECLINED"}
    assert (out["decision"] == np.where(out["calibrated_default_prob"] <= 0.05, "APPROVED", "DECLINED")).all()


def test_spark_loss_matches_numpy(spark, trained):
    models_dir, prep, calib, test = trained
    scored = score_dataframe(spark.createDataFrame(test), models_dir, 0.05)
    prob = calib.predict_proba(prep.transform(test))
    for t in (0.02, 0.05, 0.3):
        want = RupeeLossCalculator().calculate_portfolio_loss(
            test["is_default"].values, prob, t, ead=test["exposure_at_default"].values,
            lgd=test["loss_given_default"].values)
        got = portfolio_loss(scored, t)
        for k in ("total_loss_rupees", "fn_loss_rupees", "approval_count", "count_fn", "count_fp", "count_tp"):
            assert got[k] == pytest.approx(want[k], rel=1e-9), (t, k)
        assert got["approved_default_rate"] == pytest.approx(want["approved_default_rate"])


def test_spark_threshold_search_matches_optimizer(spark, trained):
    models_dir, prep, calib, test = trained
    scored = score_dataframe(spark.createDataFrame(test), models_dir, 0.05)
    prob = calib.predict_proba(prep.transform(test))
    want = ThresholdOptimizer().optimize_threshold(
        test["is_default"].values, prob, ead=test["exposure_at_default"].values, lgd=test["loss_given_default"].values)
    got = optimal_threshold(loss_curve(scored))
    assert got["optimal_threshold"] == pytest.approx(want["optimal_threshold"])
    assert got["min_loss_rupees"] == pytest.approx(want["min_expected_loss_rupees"], rel=1e-9)


def test_spark_psi_matches_numpy(spark):
    base = generate_credit_dataset(n_samples=6000, random_state=1)
    shifted = generate_credit_dataset(n_samples=6000, random_state=2, macro_drift=True)
    sb, ss = spark.createDataFrame(base), spark.createDataFrame(shifted)
    for col in ("revolving_utilization", "cibil_score", "annual_income"):
        res = continuous_psi(sb, ss, col)
        # Identical breakpoints -> identical bin counts (validates the distributed binning).
        edges = np.array(res["breakpoints"])
        assert np.array_equal(bin_counts(sb, col, res["breakpoints"]), np.histogram(base[col], bins=edges)[0])
        assert np.array_equal(bin_counts(ss, col, res["breakpoints"]), np.histogram(shifted[col], bins=edges)[0])
        # numpy interpolates quantiles, Spark picks data points, so PSI agrees closely but not exactly.
        want = PSICalculator().calculate_continuous_psi(base[col].values, shifted[col].values)["psi"]
        assert res["psi"] == pytest.approx(want, abs=0.02 + 0.05 * want), col
    cat = categorical_psi(sb, ss, "merchant_category")
    want = PSICalculator().calculate_categorical_psi(base["merchant_category"], shifted["merchant_category"])["psi"]
    assert cat["psi"] == pytest.approx(want, rel=1e-6)
