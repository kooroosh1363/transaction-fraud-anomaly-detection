from pathlib import Path
import json
import pandas as pd
from src.pipeline import main


def test_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "artifacts" / "metrics.json").read_text())
    assert metrics["data_audit"]["rows"] == 284807
    assert metrics["data_audit"]["fraud_rows"] == 492
    assert metrics["selected_model"] in {"logistic","random_forest","hist_gradient_boosting","isolation_forest"}
    test = metrics["test_result"]
    for key in ["precision","recall","f1","pr_auc","roc_auc","false_positive_rate","false_negative_rate"]:
        assert 0 <= test[key] <= 1
    assert test["tp"] + test["fn"] == metrics["split"]["test_fraud"]
    val = pd.read_csv(root / "artifacts" / "validation_metrics.csv")
    assert set(val["model"]) == {"logistic","random_forest","hist_gradient_boosting","isolation_forest"}
    assert (root / "artifacts" / "model.joblib").exists()
