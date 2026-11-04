import streamlit as st
import pandas as pd
from src.models.predict import predict_url
from src.utils.validation import validate_url
import json
from pathlib import Path

st.set_page_config(page_title="URL Scan - SmartPhish", page_icon="🔍", layout="wide")

st.title("🔍 URL Phishing Detection")
st.markdown("Enter a URL to analyze for potential phishing indicators.")

url = st.text_input("Enter URL", placeholder="https://example.com/login...")

col1, col2 = st.columns([1, 4])
with col1:
    model_name = st.selectbox("Model", ["random_forest", "gradient_boosting", "logistic_regression", "svm", "decision_tree", "voting_ensemble", "stacking_ensemble"])
with col2:
    scan = st.button("Scan URL", type="primary")

if scan and url:
    if not validate_url(url):
        st.error("Invalid URL format. Please enter a valid URL.")
    else:
        try:
            with st.spinner("Analyzing URL features and making prediction..."):
                result = predict_url(url, model_name=model_name)
            st.success("Analysis Complete")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediction", result["label"])
            c2.metric("Phishing Probability", f"{result['probability_phishing']:.3f}" if result["probability_phishing"] is not None else "N/A")
            c3.metric("Model Used", result["model"])
            st.subheader("Extracted Features")
            feat_df = pd.DataFrame([result["features"]])
            st.dataframe(feat_df.T.rename(columns={0: "Value"}), use_container_width=True)
            st.warning("⚠️ This is a research/educational prediction indicating potential risk. Always verify through official channels.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you have trained models first (run scripts/train_all.py)")
elif scan:
    st.warning("Please enter a URL to scan.")
