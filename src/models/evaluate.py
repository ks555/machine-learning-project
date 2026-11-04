from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            metrics["brier"] = float(brier_score_loss(y_test, y_proba))
        else:
            metrics["roc_auc"] = None
            metrics["brier"] = None
    except Exception:
        metrics["roc_auc"] = None
        metrics["brier"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0)
    return metrics
