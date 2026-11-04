import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.utils.io import load_yaml

def split_data(df, config_path="config/config.yaml"):
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    test_size = data_cfg.get("test_size", 0.2)
    val_size = data_cfg.get("val_size", 0.1)
    stratify = data_cfg.get("stratify", True)
    seed = cfg.get("project", {}).get("seed", 42)

    y = df["label"]
    X = df.drop(columns=["label"])
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    # val from train
    strat_t = y_train if stratify else None
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=seed, stratify=strat_t)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_preprocessor(X, config_path="config/config.yaml"):
    cfg = load_yaml(config_path)
    prep_cfg = cfg.get("preprocessing", {})
    scale = prep_cfg.get("scale", "standard")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=prep_cfg.get("impute", "median"))),
    ])
    if scale == "standard":
        numeric_transformer.steps.append(("scaler", StandardScaler()))
    elif scale == "minmax":
        numeric_transformer.steps.append(("scaler", MinMaxScaler()))
    elif scale == "robust":
        numeric_transformer.steps.append(("scaler", RobustScaler()))
    # if none, skip scaler

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", "drop", cat_cols),  # drop non-numeric by default for URL lexical numeric features
        ],
        remainder="drop",
    )
    return preprocessor, num_cols
