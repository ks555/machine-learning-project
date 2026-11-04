import streamlit as st
import pandas as pd
from src.monitoring.drift import detect_drift
from src.utils.io import load_yaml
from pathlib import Path

st.set_page_config(page_title="Monitoring & Drift - SmartPhish", page_icon="📈", layout="wide")

st.title("📈 Monitoring & Model/Data Drift Detection")
st.markdown("Compare current batch of features against a reference distribution.")

cfg = load_yaml("config/config.yaml")
ref_path = cfg.get("monitoring", {}).get("reference_path", "models/reference_distribution.parquet")

ref_df = None
if Path(ref_path).exists():
    try:
        ref_df = pd.read_parquet(ref_path)
        st.success(f"Loaded reference: {ref_path} ({ref_df.shape})")
    except Exception as e:
        st.warning(f"Could not load reference: {e}")
else:
    st.info("No reference distribution found. Save training features as reference (e.g., models/reference_distribution.parquet) to enable drift detection.")

cur_file = st.file_uploader("Upload current batch (CSV with features)", type=["csv"])
if cur_file and ref_df is not None:
    cur_df = pd.read_csv(cur_file)
    if "label" in cur_df.columns:
        cur_df = cur_df.drop(columns=["label"])
    if "url" in cur_df.columns:
        cur_df = cur_df.drop(columns=["url"])
    with st.spinner("Computing drift metrics (PSI, KS)..."):
        drift_res = detect_drift(ref_df.drop(columns=["label","url"], errors="ignore"), cur_df, psi_threshold=cfg["monitoring"]["psi_threshold"], ks_threshold=cfg["monitoring"]["ks_threshold"])
    st.dataframe(drift_res, use_container_width=True)
    drifted = drift_res[drift_res["drifted"] == True] if "drifted" in drift_res.columns else pd.DataFrame()
    if len(drifted) > 0:
        st.warning(f"⚠️ Drift detected in {len(drifted)} feature(s)")
    else:
        st.success("No significant drift detected")
elif cur_file and ref_df is None:
    st.warning("Upload reference first or save reference_distribution.parquet")
