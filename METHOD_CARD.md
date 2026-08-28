# Fraud Detection Method Card

## Intended use
Educational/portfolio demonstration of leakage-aware fraud classification and anomaly detection on the public credit-card fraud benchmark distributed through OpenML.

## Problem framing
The target is highly imbalanced binary fraud detection. Accuracy is not used as the primary quality measure. Candidate supervised models are compared with an unsupervised Isolation Forest anomaly detector.

## Forward-only evaluation
The OpenML variant currently returned through `sklearn.datasets.fetch_openml` omits the original `Time` column. The project therefore preserves source row order as `SequenceIndex` and performs a forward-only 60% train / 20% validation / 20% test split. `SequenceIndex` is used only for ordering and is never a model feature or a claimed timestamp.

If a compatible source variant containing `Time` is supplied, the same code uses `Time` for ordering instead.

## Candidate methods
- class-weighted Logistic Regression
- class-weighted Random Forest
- class-weighted HistGradientBoosting
- Isolation Forest trained only on legitimate training transactions

## Selection policy
Model selection uses validation PR-AUC with Recall as a tie-break. PR-AUC is primary because fraud prevalence is extremely low and ROC-AUC can look strong even when the positive-class ranking is operationally weak.

## Threshold policy
For each model, every distinct validation score boundary is considered as a candidate threshold. The chosen threshold minimizes the transparent illustrative cost function:

`5 * false_negatives + false_positives`

The 5:1 cost ratio is a modeling assumption, not an observed business cost.

The selected train-fitted model and its validation-tuned threshold are locked together before the final test is evaluated. The model is deliberately not refit on train+validation while reusing the prior threshold, because refitting can shift score calibration and make the old threshold inconsistent with the new model.

## Metrics
Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrix, false-positive rate, false-negative rate, and the illustrative FP/FN cost are reported.

## Limitations
- V1–V28 are anonymized transformed variables and are not directly business-interpretable;
- the currently retrieved OpenML variant does not expose the original `Time` field;
- preserved source row order is useful for forward-only evaluation but is not equivalent to a verified wall-clock timeline;
- the public benchmark is historical and not representative of all payment systems;
- fraud prevalence and adversarial behavior drift in production;
- the illustrative cost ratio is not a real institution's loss model;
- threshold choice in production should account for investigator capacity and actual loss economics;
- offline metrics do not prove fraud-loss reduction, customer impact, or causal value.

## Production extensions
A production system would add calibrated business costs, probability calibration, velocity/device/network features, investigator feedback, delayed-label handling, drift monitoring, champion/challenger deployment, rule+ML orchestration, review-capacity-aware thresholds, and online operational monitoring.
