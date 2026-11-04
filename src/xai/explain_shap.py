import shap
import numpy as np
import pandas as pd

def get_shap_explainer(model, X_background):
    try:
        # Try TreeExplainer first
        clf = model
        # if pipeline
        if hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", model)
            pre = model.named_steps.get("preprocessor")
            if pre is not None:
                Xb = pre.transform(X_background)
            else:
                Xb = X_background
        else:
            Xb = X_background
        try:
            explainer = shap.TreeExplainer(clf)
            return explainer, Xb
        except Exception:
            explainer = shap.Explainer(clf, Xb)
            return explainer, Xb
    except Exception as e:
        raise e

def explain_with_shap(model, X_instance, X_background=None):
    if X_background is None and hasattr(model, "named_steps"):
        pre = model.named_steps.get("preprocessor")
        if pre is not None:
            # can't get background easily; use instance
            X_background = X_instance
    explainer, Xb = get_shap_explainer(model, X_background if X_background is not None else X_instance)
    try:
        shap_values = explainer(Xb)
    except Exception:
        shap_values = explainer.shap_values(Xb)
    return explainer, shap_values, Xb
