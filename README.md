# RiskLens: Cost-Sensitive Credit Default Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Evaluation Metric: PR-AUC](https://img.shields.io/badge/Metric-PR--AUC-orange.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

> **InfoSecIITR | Credit underwriting case study**
> A cost-sensitive credit default classifier that chooses its approval cutoff by **expected rupee loss rather than accuracy**, ranks risk with **PR-AUC**, calibrates probabilities with **Platt scaling**, explains decisions with **SHAP**, and monitors drift with **PSI**.

> [!IMPORTANT]
> **All data in this repository is synthetic.** Applicants, defaults, EAD and LGD are produced by `risklens/data/generator.py` from a hand-written risk model. Every number below describes how the method behaves on that generator, not on real borrowers, and none of it should be read as evidence about real-world portfolio performance. Section [Limitations](#-limitations-read-before-quoting-any-number) lists what this does and does not show.

Every figure in this README is copied from the files that `scripts/run_pipeline.py` writes to `artifacts/reports/` (`model_benchmark.json`, `approval_lift_metrics.json`, `drift_audit.json`, `run_summary.json`). Re-run the pipeline and they will match.

---

## 📌 Problem

In credit underwriting the two error types have very different costs, so accuracy and the default `0.5` cutoff are the wrong targets:

| Error Type | Underwriting Event | Cost modelled here |
| :--- | :--- | :--- |
| **False Negative (FN)** | Approve an applicant who defaults | Exposure at default × loss given default (about ₹75,000 in the default config; per-account EAD × LGD in the pipeline) |
| **False Positive (FP)** | Decline a creditworthy applicant | ₹3,500 foregone margin |
| **True Positive (TP)** | Decline an applicant who would default | ₹450 manual review |

The cost parameters are assumptions set in `risklens/config.py`, not measured values. The FN : FP ratio is about 21 : 1.

**What RiskLens does:**
1. Frames underwriting as **expected rupee loss minimisation** over the approval cutoff.
2. Ranks default risk with LightGBM and compares against a logistic scorecard using **PR-AUC**.
3. Calibrates probabilities with **Platt scaling** so that predicted risk matches observed default rates.
4. Grid-searches the cutoff that minimises portfolio rupee loss.
5. Measures how many more applicants can be approved at the **same approved-portfolio default rate** as the scorecard, with a bootstrap interval.
6. Produces **SHAP-based decline reasons** in an adverse-action style (illustrative, not reviewed for regulatory compliance).
7. Monitors input and score drift with **Population Stability Index (PSI)** against an out-of-time cohort.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Synthetic data (100k records)"]
        A[In-Time Applications: 80k] --> C[Feature Pipeline]
        B[Out-Of-Time Drift Cohort: 20k] --> C
    end

    subgraph Modeling["2. Modelling and calibration"]
        C --> D[LightGBM / Logistic Baseline]
        D --> E[Platt Scaling Calibrator]
        E --> F[Calibrated P(Default)]
    end

    subgraph Optimization["3. Cost-sensitive optimisation"]
        F --> G[Rupee Loss: EAD x LGD vs FP cost]
        G --> H[Threshold search: argmin L(tau)]
        H --> I[Approval lift at matched risk + bootstrap CI]
    end

    subgraph Monitoring["4. Explainability and drift"]
        D --> J[TreeSHAP Explainer]
        J --> K[Adverse-action style reasons]
        F --> L[PSI: score and feature drift]
    end

    subgraph Serving["5. Serving and UI"]
        H --> N[FastAPI service]
        I --> O[Streamlit dashboard]
        K --> N
        L --> O
    end
```

---

## 📐 Mathematical Formulation

### 1. Portfolio expected rupee loss
Let $y_i \in \{0, 1\}$ be the default indicator, $p_i$ the calibrated default probability and $\tau$ the cutoff. An applicant is approved if $p_i \le \tau$ and declined if $p_i > \tau$.

$$\mathcal{L}(\tau) = \sum_{i=1}^N \Big[ y_i \cdot \mathbb{I}(p_i \le \tau) \cdot (\text{EAD}_i \times \text{LGD}_i) + (1 - y_i) \cdot \mathbb{I}(p_i > \tau) \cdot \text{Cost}_{FP} + y_i \cdot \mathbb{I}(p_i > \tau) \cdot C_{\text{review}} \Big]$$

In the synthetic generator:
* $\text{EAD}_i = \text{Credit Limit}_i \times \text{clip}(0.65 + 0.35 \times \text{Utilization}_i,\ 0.5,\ 1.0)$
* $\text{LGD}_i \sim \text{Beta}(7, 3)$ clipped to $[0.45, 0.95]$
* $\text{Cost}_{FP} = ₹3{,}500$, $C_{\text{review}} = ₹450$

$$\tau^* = \arg\min_{\tau \in [0.005,\ 0.70]} \mathcal{L}(\tau)$$

### 2. Platt scaling
With raw log-odds $s_i$, fit $a, b$ on a held-out calibration split: $\hat{p}_i = \sigma(a\,s_i + b)$. Because $\sigma$ is strictly increasing this changes probabilities but **not the ranking**, so PR-AUC and ROC-AUC are identical before and after calibration (see the table below).

### 3. Population Stability Index
Features and scores are binned into $B=10$ buckets:

$$\text{PSI} = \sum_{b=1}^B \left( \% \text{Actual}_b - \% \text{Expected}_b \right) \times \ln\left( \frac{\% \text{Actual}_b}{\% \text{Expected}_b} \right)$$

Rule-of-thumb thresholds used in credit-risk monitoring: **< 0.10** stable, **0.10 to 0.20** warning, **≥ 0.20** significant drift.

---

## 📊 Results (synthetic data)

Setup: 80,000 in-time records split 56,000 train / 12,000 calibration / 12,000 test (706 defaults in the test set). In-time default rate 5.96%; the out-of-time cohort (20,000 records, drift injected on purpose) has 11.94%.

### 1. Model comparison on the 12,000-account test set

| Model | PR-AUC | ROC-AUC | Brier | Cutoff τ* | Loss @ 0.5 (₹) | Loss @ τ* (₹) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic scorecard | 0.7248 | 0.9520 | 0.0845 | 0.491 | 1,34,88,613 | 1,34,38,410 |
| LightGBM (raw) | 0.7485 | 0.9560 | 0.0586 | 0.304 | 1,57,26,440 | 1,15,32,068 |
| LightGBM + Platt (RiskLens) | 0.7485 | 0.9560 | **0.0262** | 0.044 | 3,68,99,988 | 1,16,15,478 |

How to read this:
* **The gain over the scorecard comes from the model, not from calibration.** At each model's own best cutoff, RiskLens loses ₹1,16,15,478 versus ₹1,34,38,410 for the scorecard: **₹18.2 lakh, or 13.6%, lower**.
* **Calibration does not improve decisions here.** It leaves PR-AUC and ROC-AUC unchanged, and raw LightGBM at its best cutoff is slightly *cheaper* (₹1,15,32,068) than the calibrated model. What calibration buys is probability quality: Brier 0.0586 → 0.0262 and expected calibration error 0.0023, which matters when probabilities feed expected-loss calculations.
* **The "loss at 0.5" column is a weak baseline.** A calibrated model's probabilities sit near 0.03, so a 0.5 cutoff approves almost everyone (95.7%) and the loss jumps to ₹3.69 crore. Quoting "68.5% lower than 0.5" would mostly measure how bad 0.5 is, so treat the best-cutoff comparison above as the meaningful one.
* The optimal cutoff is chosen on the same test cohort it is scored on, which is slightly optimistic. See Limitations.

### 2. Approval lift at matched risk

Question: if the scorecard approves its lowest-risk 78% of applicants, how many more can RiskLens approve while the **approved-portfolio default rate stays at or below the scorecard's**? The code searches for the largest such approval volume. The lift is an output of that search, not an input.

| Metric | Scorecard (78% approved) | RiskLens at matched risk | Delta |
| :--- | :---: | :---: | :---: |
| Approval rate | 78.00% | 81.91% | **+3.91 pts** |
| Accounts approved (of 12,000) | 9,360 | 9,829 | +469 |
| Approved default rate | 0.556% | 0.549% | -0.006 pts |
| Default loss on approved accounts | ₹66,89,707 | ₹61,34,697 | -₹5,55,010 |

**Uncertainty.** 500 bootstrap resamples of the test accounts give a **95% interval of -0.77 to +6.69 points** for the lift (positive in 95.2% of resamples). With only 706 defaults in the test set, the interval is wide and includes zero, so this is evidence of a likely gain, not proof of one. Holding the default *rupee loss* constant instead gives +5.18 points (95% interval +0.07 to +6.97).

**Commercial estimate (assumption-driven).** Assuming ₹6,500 of annual margin per newly approved good account (an assumed figure, set in `compute_approval_expansion`), the 469 extra approvals are worth about ₹30,35,500 of margin; net of the lower default loss that is about **₹35,90,510**. Change the margin assumption and this moves proportionally.

---

## 🔍 Explainability

TreeSHAP attributions explain individual decisions, and top drivers of a decline are mapped to reason codes in an adverse-action style:

![SHAP Summary Plot](assets/shap_summary.png)

Example decline output from the API for a synthetic distressed profile (CIBIL 450, utilisation 0.95, 3 delinquencies, 5 inquiries, DTI 0.70):
```text
[AAR-01] Credit bureau score does not meet minimum underwriting criteria
[AAR-04] Presence of recent delinquent credit payments or past-due accounts in bureau history
[AAR-02] Proportion of revolving credit balances to total credit limits is excessively elevated
[AAR-05] Too many recent hard inquiries for new credit lines indicating credit hunger
```
These reason texts are illustrative. They have not been reviewed against FCRA, ECOA or RBI requirements, and the model uses features such as employment type that a real lender would need to assess for fair-lending risk.

---

## 🛡️ PSI drift monitoring

The audit compares the first 10,000 in-time records with the first 10,000 out-of-time records, and score drift on 5,000 scores each. The out-of-time cohort is generated with a deliberate shift (lower bureau scores, higher utilisation, more delinquencies), so detecting drift here confirms the monitor works, not that a real portfolio drifted.

![PSI Drift Report](assets/psi_drift_report.png)

| Dimension | PSI | Status |
| :--- | :---: | :--- |
| Model score (calibrated P(default)) | 0.187 | WARNING |
| `revolving_utilization` | 1.085 | SEVERE_DRIFT |
| `cibil_score` | 0.143 | WARNING |
| `delinquencies_2yr` | 0.082 | STABLE |
| `debt_to_income` | 0.072 | STABLE |
| `annual_income` | 0.003 | STABLE |

Overall status: **CRITICAL_DRIFT_ALERT** (driven by `revolving_utilization`). Full per-feature results are in `artifacts/reports/drift_audit.json`.

---

## ⚠️ Limitations (read before quoting any number)

* **Synthetic data.** Labels are drawn from `true_risk_prob`, a logistic function of the features written in the generator. A model scoring PR-AUC 0.7485 is close to the **0.7611** that the *true generating probability* itself achieves on this test set, so the headline metrics mostly reflect the generator's signal-to-noise ratio. The generator also includes interaction terms, which favour tree models over a linear scorecard by construction.
* **One dataset, one seed.** No repeated runs across seeds, and a single train/calibration/test split.
* **Threshold tuned on the test set.** τ* is searched on the same cohort used to report loss. A cleaner protocol tunes on the calibration split and scores on test.
* **Assumed economics.** The ₹3,500 FP cost, ₹450 review cost, EAD/LGD distributions and ₹6,500 margin are assumptions, and results depend on them.
* **Single label.** Default and fraud are folded into one `is_default` label. This is a default model with a few fraud-style features, not a separate fraud detector.
* **Fixed baseline operating point.** The lift is measured against a scorecard approving 78% of applicants, a choice made for the comparison.
* **Not production reviewed.** Nothing here has been validated for regulatory use, model-risk governance or fair lending.

---

## 🚀 Quickstart

### 1. Set up
```bash
git clone https://github.com/0xGu4rdI4N/risk_lens.git
cd risk_lens
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run the end-to-end pipeline
Generates 100k synthetic records, trains the scorecard and LightGBM, calibrates, optimises the cutoff, measures approval lift with a bootstrap interval, runs SHAP and the PSI audit, and writes everything to `artifacts/reports/`:
```bash
python scripts/run_pipeline.py
```

### 3. Print the measured approval lift
```bash
python scripts/demo_tradeoff.py
```

### 4. Run the tests
```bash
pytest -v tests/
```

### 5. Launch the dashboard
All dashboard metrics are read from `artifacts/reports/`; run the pipeline first.
```bash
streamlit run risklens/dashboard/app.py
```

### 6. Start the FastAPI service
```bash
uvicorn risklens.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Swagger docs at `http://localhost:8000/docs`.

---

## 🔌 API

### Score a single applicant
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "annual_income": 950000.0,
      "cibil_score": 780,
      "revolving_utilization": 0.22,
      "debt_to_income": 0.24,
      "delinquencies_2yr": 0,
      "inquiries_last_6m": 0,
      "open_credit_lines": 5,
      "credit_history_months": 84,
      "transaction_velocity_24h": 1.1,
      "device_risk_score": 0.05,
      "merchant_category": "retail",
      "employment_type": "salaried",
      "requested_credit_limit": 300000.0
    }
  }'
```

Response produced by the committed artifacts (SHAP list shortened to two entries):
```json
{
  "applicant_id": "APP-ONLINE",
  "raw_score": 0.01691,
  "calibrated_default_prob": 0.00224,
  "decision": "APPROVED",
  "applied_threshold": 0.0441,
  "expected_rupee_loss": 366.55,
  "top_risk_factors": [
    {"feature": "cibil_score", "shap_value": -1.263, "feature_value": 780, "direction": "RISK_DECREASING"},
    {"feature": "device_risk_score", "shap_value": -0.263, "feature_value": 0.05, "direction": "RISK_DECREASING"}
  ],
  "adverse_action_reasons": []
}
```

---

## 📁 Repository structure

```
risk_lens/
├── .github/workflows/ci.yml     # GitHub Actions CI (Python 3.10, 3.11, 3.12)
├── risklens/
│   ├── config.py                # Hyperparameters, cost matrix, PSI thresholds
│   ├── cli.py                   # Typer/Rich CLI
│   ├── data/                    # Synthetic generator, preprocessor, Pydantic schemas
│   ├── cost/
│   │   ├── loss_matrix.py       # Rupee loss (EAD x LGD vs FP cost)
│   │   ├── optimizer.py         # Grid search minimising portfolio rupee loss
│   │   └── frontier.py          # Approval-vs-risk frontier, matched-risk lift, bootstrap CI
│   ├── models/                  # Scorecard, LightGBM/XGBoost, Platt/isotonic calibrator, evaluator
│   ├── explainability/          # TreeSHAP explainer, adverse-action style reasons
│   ├── monitoring/              # PSI and drift audit
│   ├── api/app.py               # FastAPI service (/predict, /drift, /health)
│   └── dashboard/app.py         # Streamlit dashboard (reads artifacts/reports)
├── scripts/
│   ├── run_pipeline.py          # Full pipeline; writes artifacts/reports/*.json
│   └── demo_tradeoff.py         # Prints the saved benchmark and approval-lift tables
├── artifacts/reports/           # model_benchmark, approval_lift_metrics, drift_audit, run_summary
├── notebooks/RiskLens_Deep_Dive.ipynb
├── assets/                      # SHAP and PSI figures
├── tests/                       # 21 unit and integration tests
├── pyproject.toml
└── requirements.txt
```

---

## 📜 License & Citation

Open source under the [Apache 2.0 License](LICENSE).

```bibtex
@software{RiskLens2026,
  author = {Kumar, Raunak},
  title = {RiskLens: Cost-Sensitive Credit Default Engine},
  year = {2026},
  url = {https://github.com/0xGu4rdI4N/risk_lens}
}
```
