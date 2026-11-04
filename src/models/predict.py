import json
import joblib
from pathlib import Path
import pandas as pd
from src.features.url_features import extract_features_from_url
from src.utils.io import load_yaml, load_json

def load_artifacts(config_path="config/config.yaml", paths_path="config/paths.yaml"):
    pths = load_yaml(paths_path)
    preprocessor = joblib.load(pths["models"]["preprocessor"])
    feature_list = load_json(pths["models"]["feature_list"])
    registry = load_json(pths["models"]["registry"])
    return preprocessor, feature_list, registry, pths

def predict_url(url, model_name="random_forest", config_path="config/config.yaml", paths_path="config/paths.yaml"):
    preprocessor, feature_list, registry, pths = load_artifacts(config_path, paths_path)
    feats = extract_features_from_url(url)
    df = pd.DataFrame([feats])
    # align to feature_list
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_list]
    model_path = registry.get(model_name)
    if not model_path or not Path(model_path).exists():
        # try to find any
        for k, v in registry.items():
            if Path(v).exists():
                model_path = v
                model_name = k
                break
    pipe = joblib.load(model_path)
    pred = pipe.predict(df)[0]
    proba = None
    if hasattr(pipe, "predict_proba"):
        try:
            proba = float(pipe.predict_proba(df)[0][1])
        except Exception:
            proba = None
    return {
        "url": url,
        "model": model_name,
        "prediction": int(pred),  # 1 = phishing
        "label": "Phishing" if int(pred) == 1 else "Legitimate",
        "probability_phishing": proba,
        "features": feats,
    }
