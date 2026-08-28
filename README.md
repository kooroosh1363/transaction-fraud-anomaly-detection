# DS-07 — Transaction Fraud & Anomaly Detection

Portfolio-grade fraud detection project built around severe class imbalance, temporal evaluation, validation-only threshold tuning, supervised fraud models, and unsupervised anomaly detection.

## What this project demonstrates

- public benchmark acquisition through OpenML
- exact row/fraud-count validation
- chronological train/validation/test split
- class imbalance audit
- class-weighted Logistic Regression
- class-weighted Random Forest
- class-weighted HistGradientBoosting
- Isolation Forest anomaly detection trained on legitimate history
- PR-AUC-first model selection
- validation-only threshold tuning
- Precision, Recall, F1, PR-AUC, ROC-AUC
- confusion matrix, FPR, FNR
- transparent cost-sensitive diagnostic
- final untouched test evaluation
- artifacts, tests, and GitHub Actions CI

## Data

The project uses the well-known anonymized credit-card fraud benchmark available through OpenML. It contains **284,807 transactions**, of which only **492** are labeled as fraud (~0.173%). Features include `Time`, `Amount`, anonymized `V1`–`V28`, and binary target `Class`.

See `DATA_SOURCE.md`, `DATA_DICTIONARY.md`, and `METHOD_CARD.md`.

## Architecture

```text
OpenML creditcard dataset
    -> validate row count + fraud count + schema
    -> sort by Time
    -> chronological split
         train 60%
         validation 20%
         test 20%
    -> candidate supervised models
         Logistic Regression
         Random Forest
         HistGradientBoosting
    -> unsupervised Isolation Forest
    -> validation PR-AUC comparison
    -> validation threshold tuning
         illustrative cost = 5*FN + FP
    -> lock model + threshold
    -> refit selected model on train + validation
    -> final untouched test
    -> ranking/classification diagnostics
    -> artifacts + pytest + GitHub Actions
```

## Why accuracy is not the main metric

Fraud prevalence is below 0.2%. A classifier that predicts every transaction as legitimate would appear more than 99.8% accurate while detecting no fraud. For that reason this project prioritizes **PR-AUC, Recall, Precision, F1, and confusion-matrix costs**.

## Threshold policy

The probability/anomaly threshold is not fixed at 0.5. It is tuned on validation only using a transparent illustrative objective:

```text
cost = 5 * false_negatives + false_positives
```

The 5:1 ratio is explicitly a modeling assumption for demonstrating cost-sensitive thresholding, not a claim about real banking economics.

## Supervised vs anomaly detection

Supervised models learn from fraud labels. Isolation Forest is trained only on legitimate training transactions and asks which future transactions look anomalous. An anomaly is not automatically fraud, so the two paradigms are evaluated under the same offline labels but interpreted differently.

## Claim boundary

This project demonstrates offline historical fraud classification and anomaly detection methodology. It does not prove production fraud-loss reduction, investigator efficiency, customer impact, or causal business lift.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline
```
