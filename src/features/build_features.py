import pandas as pd
from .url_features import extract_features_batch

def build_feature_dataframe(df, url_col="url"):
    urls = df[url_col].fillna("").astype(str).tolist()
    feat_df = extract_features_batch(urls)
    # keep url for reference
    feat_df = feat_df.copy()
    feat_df.insert(0, "url", df[url_col].values)
    if "label" in df.columns:
        feat_df["label"] = df["label"].values
    return feat_df
