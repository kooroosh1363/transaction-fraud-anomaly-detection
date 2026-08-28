# DS-07 — Transaction Fraud & Anomaly Detection

Portfolio-grade fraud detection project built around severe class imbalance, forward-only evaluation, validation-only threshold tuning, supervised fraud models, and unsupervised anomaly detection.

## What this project demonstrates

- public benchmark acquisition through OpenML
- exact row/fraud-count validation
- forward-only 60/20/20 train/validation/test split
- class imbalance audit
- class-weighted Logistic Regression
- class-weighted Random Forest
- class-weighted HistGradientBoosting
- Isolation Forest anomaly detection trained only on legitimate training history
- PR-AUC-first model selection
- exact validation-only threshold tuning
- Precision, Recall, F1, PR-AUC, ROC-AUC
- confusion matrix, FPR, FNR
- transparent cost-sensitive diagnostic
- locked model+threshold final test evaluation
- artifacts, tests, and GitHub Actions CI

## Data

The project uses the anonymized credit-card fraud benchmark distributed through OpenML. It contains **284,807 transactions**, of which **492** are labeled as fraud (~0.173%).

The OpenML variant currently returned through `sklearn.datasets.fetch_openml` contains `V1`–`V28`, `Amount`, and `Class`, but **does not include the original `Time` column**. The project therefore preserves the source row order as `SequenceIndex` and uses it **only for forward-only splitting**. `SequenceIndex` is not a model feature and is not presented as a real timestamp.

Model inputs are therefore **29 features: `V1`–`V28` + `Amount`**.

See `DATA_SOURCE.md`, `DATA_DICTIONARY.md`, and `METHOD_CARD.md`.

## Architecture

```text
OpenML creditcard dataset
    -> validate row count + fraud count + schema
    -> preserve source row order as SequenceIndex
    -> forward-only split
         train 60%
         validation 20%
         test 20%
    -> candidate supervised models
         Logistic Regression
         Random Forest
         HistGradientBoosting
    -> unsupervised Isolation Forest
    -> validation PR-AUC comparison
    -> exact validation threshold search
         illustrative cost = 5*FN + FP
    -> lock selected train-fitted model + threshold together
    -> final untouched test
    -> classification + cost diagnostics
    -> artifacts + pytest + GitHub Actions
```

## Why accuracy is not the main metric

Fraud prevalence is below 0.2%. A classifier that predicts every transaction as legitimate would appear more than 99.8% accurate while detecting no fraud. For that reason this project prioritizes **PR-AUC, Recall, Precision, F1, and confusion-matrix costs**.

## Model selection policy

Model selection uses validation **PR-AUC** with **Recall** as a tie-breaker. PR-AUC is threshold-independent and is more informative than plain accuracy for this extremely imbalanced problem.

The best validation PR-AUC determines the model family. Threshold selection is a separate decision.

## Threshold policy

The probability/anomaly threshold is not fixed at 0.5. The code evaluates **every distinct validation score boundary** and selects the exact threshold minimizing:

```text
cost = 5 * false_negatives + false_positives
```

The 5:1 ratio is explicitly an illustrative modeling assumption, not a claim about real banking economics.

The selected **train-fitted model and its validation-tuned threshold are locked together** before final test evaluation. The project intentionally does not refit the model on train+validation while reusing the old threshold, because refitting can change score calibration and invalidate that threshold.

## Supervised vs anomaly detection

Supervised models learn from fraud labels. Isolation Forest is trained only on legitimate training transactions and asks which future transactions look anomalous. An anomaly is not automatically fraud, so the two paradigms are evaluated under the same offline labels but interpreted differently.

## Validated observations

The current CI-validated run shows an important lesson about imbalanced fraud detection:

- Logistic Regression has the strongest validation PR-AUC among the tested models.
- Random Forest and HistGradientBoosting achieve strong precision/recall at their tuned thresholds but slightly lower PR-AUC.
- Isolation Forest has a high ROC-AUC but very weak PR-AUC and can fail badly at a low-alert threshold.

This is exactly why ROC-AUC alone is not enough for a rare-event fraud problem.

## Claim boundary

This project demonstrates offline historical fraud classification and anomaly-detection methodology. It does not prove production fraud-loss reduction, investigator efficiency, customer impact, or causal business lift.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline
```
