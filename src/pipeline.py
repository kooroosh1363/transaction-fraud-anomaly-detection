from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset, temporal_split

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FEATURES = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def choose_threshold(y, score):
    candidates = np.unique(np.quantile(score, np.linspace(0.90, 0.9999, 250)))
    best = None
    for t in candidates:
        m = metrics(y, score, t)
        # Fraud misses are treated as five times as costly as false alerts.
        cost = 5*m["fn"] + m["fp"]
        key = (cost, -m["recall"], -m["precision"])
        if best is None or key < best[0]:
            best = (key, m)
    return best[1]


def build_models():
    return {
        "logistic": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))]),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=220, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42, class_weight="balanced"),
    }


def score_model(model, X):
    return model.predict_proba(X)[:,1]


def main():
    ART.mkdir(exist_ok=True)
    df, audit = load_dataset()
    train, val, test, split = temporal_split(df)
    Xtr, ytr = train[FEATURES], train["Class"].to_numpy()
    Xv, yv = val[FEATURES], val["Class"].to_numpy()
    Xt, yt = test[FEATURES], test["Class"].to_numpy()

    val_rows = []
    fitted = {}
    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        s = score_model(model, Xv)
        m = choose_threshold(yv, s)
        m["model"] = name
        val_rows.append(m)

    # Unsupervised anomaly detector trained only on legitimate training transactions.
    iso = IsolationForest(n_estimators=250, contamination=float(ytr.mean()), random_state=42, n_jobs=-1)
    iso.fit(Xtr[ytr == 0])
    iso_score = -iso.decision_function(Xv)
    iso_m = choose_threshold(yv, iso_score)
    iso_m["model"] = "isolation_forest"
    val_rows.append(iso_m)

    val_df = pd.DataFrame(val_rows)
    selected = str(val_df.sort_values(["pr_auc", "recall"], ascending=False).iloc[0]["model"])
    selected_threshold = float(val_df.loc[val_df["model"] == selected, "threshold"].iloc[0])

    if selected == "isolation_forest":
        train_val = pd.concat([train, val], ignore_index=True)
        final_model = IsolationForest(n_estimators=250, contamination=float(train_val["Class"].mean()), random_state=42, n_jobs=-1)
        final_model.fit(train_val.loc[train_val["Class"] == 0, FEATURES])
        test_score = -final_model.decision_function(Xt)
    else:
        train_val = pd.concat([train, val], ignore_index=True)
        final_model = build_models()[selected]
        final_model.fit(train_val[FEATURES], train_val["Class"])
        test_score = score_model(final_model, Xt)

    test_result = metrics(yt, test_score, selected_threshold)
    test_result["model"] = selected

    joblib.dump({"model": final_model, "threshold": selected_threshold, "features": FEATURES}, ART / "model.joblib")
    val_df.to_csv(ART / "validation_metrics.csv", index=False)
    pd.DataFrame([test_result]).to_csv(ART / "test_metrics.csv", index=False)
    report = {
        "data_audit": audit,
        "split": split,
        "selection_policy": "highest validation PR-AUC; recall tie-break; threshold tuned on validation cost = 5*FN + FP",
        "candidate_models": ["logistic", "random_forest", "hist_gradient_boosting", "isolation_forest"],
        "validation_results": val_df.to_dict(orient="records"),
        "selected_model": selected,
        "selected_threshold": selected_threshold,
        "test_result": test_result,
        "claim_boundary": "offline historical fraud classification/anomaly detection; no guarantee of production fraud loss reduction or causal business impact"
    }
    (ART / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
