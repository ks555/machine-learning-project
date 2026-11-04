import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Model Comparison - SmartPhish", page_icon="📊", layout="wide")

st.title("📊 Model Performance Comparison")

metrics_path = Path("models/metrics.json")
if not metrics_path.exists():
    st.warning("No trained models found. Run `python scripts/train_all.py` first.")
else:
    with open(metrics_path) as f:
        results = json.load(f)
    rows = []
    for name, res in results.items():
        if "metrics" in res:
            m = res["metrics"]
            rows.append({
                "Model": name,
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1-Score": m.get("f1"),
                "ROC-AUC": m.get("roc_auc"),
                "Brier Score": m.get("brier"),
            })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], axis=0), use_container_width=True)
        st.subheader("Visual Comparison")
        try:
            import plotly.express as px
            fig = px.bar(df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]), x="Model", y="value", color="variable", barmode="group", title="Model Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Plotly not available or error: {e}")
    else:
        st.info("No valid metrics to display.")
