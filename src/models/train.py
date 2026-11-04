import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.data.load_data import load_dataset
from src.data.preprocess import split_data, build_preprocessor
from src.features.build_features import build_feature_dataframe
from src.models.registry import get_models
from src.models.evaluate import evaluate_model
from src.utils.io import load_yaml, save_json
from src.utils.logging import get_logger

logger = get_logger("train")

def train_all_models(raw_data_path=None, features_path=None, config_path="config/config.yaml", paths_path="config/paths.yaml"):
    cfg = load_yaml(config_path)
    pths = load_yaml(paths_path)
    seed = cfg.get("project", {}).get("seed", 42)

    # Load features
    if features_path and Path(features_path).exists():
        df_feat = pd.read_csv(features_path)
    elif raw_data_path and Path(raw_data_path).exists():
        df_raw = load_dataset(raw_data_path, config_path=config_path)
        df_feat = build_feature_dataframe(df_raw)
    else:
        raise FileNotFoundError("Provide raw_data_path or features_path")

    if "label" not in df_feat.columns:
        raise ValueError("Features dataframe must contain 'label' column")

    # Keep url if present
    url_col_df = "url" if "url" in df_feat.columns else None
    X_full = df_feat.drop(columns=["label"])
    if url_col_df and url_col_df in X_full.columns:
        X_full = X_full.drop(columns=[url_col_df])
    y_full = df_feat["label"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(pd.concat([X_full, y_full], axis=1).rename(columns={y_full.name: "label"}), config_path=config_path)
    # easier: rebuild from split by reindexing not needed; just split on original
    # redo split properly
    from sklearn.model_selection import train_test_split
    data_cfg = cfg.get("data", {})
    strat = y_full if data_cfg.get("stratify", True) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=data_cfg.get("test_size", 0.2), random_state=seed, stratify=strat)
    strat_tr = y_tr if data_cfg.get("stratify", True) else None
    val_ratio = data_cfg.get("val_size", 0.1) / (1 - data_cfg.get("test_size", 0.2))
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val_ratio, random_state=seed, stratify=strat_tr)
    X_train, X_val, X_test, y_train, y_val, y_test = X_tr, X_va, X_te, y_tr, y_va, y_te

    preprocessor, num_cols = build_preprocessor(X_train, config_path=config_path)
    models = get_models(cfg)

    Path(pths["models"]["dir"]).mkdir(parents=True, exist_ok=True)

    # Fit preprocessor
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, pths["models"]["preprocessor"])
    # Save feature list
    feature_list = X_train.columns.tolist()
    save_json(feature_list, pths["models"]["feature_list"])

    results = {}
    prep_cfg = cfg.get("preprocessing", {})
    imb = prep_cfg.get("handle_imbalance", "class_weight")

    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            if imb == "smote" and name not in ["voting_ensemble", "stacking_ensemble"]:
                pipe = ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=seed)), ("clf", model)])
            else:
                pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
            pipe.fit(X_train, y_train)
            metrics = evaluate_model(pipe, X_test, y_test)
            model_path = Path(pths["models"]["dir"]) / f"{name}.joblib"
            joblib.dump(pipe, model_path)
            results[name] = {
                "metrics": metrics,
                "model_path": str(model_path),
            }
            logger.info(f"{name} done - ROC-AUC: {metrics.get('roc_auc')}")
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            results[name] = {"error": str(e)}

    save_json(results, pths["models"]["metrics"])
    # simple registry
    registry = {name: res.get("model_path") for name, res in results.items() if "model_path" in res}
    save_json(registry, pths["models"]["registry"])
    logger.info("Training complete. Results saved.")
    return results
