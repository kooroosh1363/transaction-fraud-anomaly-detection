# Fraud Detection Method Card

## Intended use
Educational/portfolio demonstration of leakage-aware fraud classification and anomaly detection on the public credit-card fraud benchmark distributed through OpenML.

## Problem framing
The target is highly imbalanced binary fraud detection. Accuracy is not used as the primary quality measure. Candidate supervised models are compared with an unsupervised Isolation Forest anomaly detector.

## Temporal evaluation
Transactions are sorted by `Time` and split 60% train / 20% validation / 20% test in chronological order. Model choice and threshold tuning use validation only. The selected model is refit on train+validation and evaluated once on the final test partition.

## Candidate methods
- class-weighted Logistic Regression
- class-weighted Random Forest
- class-weighted HistGradientBoosting
- Isolation Forest trained only on legitimate training transactions

## Selection and threshold policy
Model selection uses validation PR-AUC with Recall as a tie-break. Decision threshold is tuned only on validation with a transparent illustrative cost function: `5 * false_negatives + false_positives`. The 5:1 cost ratio is a modeling assumption, not an observed business cost.

## Metrics
Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrix, false-positive rate, and false-negative rate are reported.

## Limitations
- PCA-like anonymized V1–V28 variables are not business-interpretable raw features;
- `Time` is relative seconds rather than a real calendar timestamp;
- the public benchmark is historical and not representative of all payment systems;
- fraud prevalence and adversarial behavior drift in production;
- the illustrative cost ratio is not a real institution's loss model;
- offline metrics do not prove fraud-loss reduction, customer impact, or causal value.

## Production extensions
A production system would add calibrated business costs, probability calibration, velocity/device/network features, investigator feedback, delayed-label handling, drift monitoring, champion/challenger deployment, rule+ML orchestration, and threshold policies tied to review capacity.
