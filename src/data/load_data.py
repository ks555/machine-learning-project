import pandas as pd
from pathlib import Path
from src.utils.io import load_yaml

def load_dataset(path, url_col="url", label_col="label", config_path="config/config.yaml"):
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    url_col = data_cfg.get("url_col", url_col)
    label_col = data_cfg.get("label_col", label_col)
    label_map = data_cfg.get("label_map", {})

    df = pd.read_csv(path)
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    url_col_l = url_col.lower()
    label_col_l = label_col.lower()
    if url_col_l not in df.columns:
        # try common alternatives
        for alt in ["url", "link", "website", "domain"]:
            if alt in df.columns:
                url_col_l = alt
                break
    if label_col_l not in df.columns:
        for alt in ["label", "class", "target", "status", "phishing"]:
            if alt in df.columns:
                label_col_l = alt
                break

    df = df.rename(columns={url_col_l: "url", label_col_l: "label"})
    # Map labels
    if df["label"].dtype == object:
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(lambda x: label_map.get(x, x))
    # Try to coerce
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    return df
