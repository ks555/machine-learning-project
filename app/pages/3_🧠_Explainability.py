import streamlit as st
import pandas as pd
from src.models.predict import load_artifacts
from src.features.url_features import extract_features_from_url
from src.xai.explain_shap import explain_with_shap
from src.xai.visualize import plot_shap_summary, plot_shap_waterfall
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Explainability - SmartPhish", page_icon="🧠", layout="wide")

st.title("🧠 Explainable AI (XAI) with SHAP")
st.markdown("Understand which URL features influenced the prediction.")

url = st.text_input("Enter URL to explain", placeholder="https://example.com/suspicious-page")
model_name = st.selectbox("Model", ["random_forest", "gradient_boosting", "xgboost" if False else "random_forest"])

if st.button("Explain Prediction"):
    if not url:
        st.warning("Enter a URL")
    else:
        try:
            preprocessor, feature_list, registry, pths = load_artifacts()
            model_path = registry.get(model_name)
            pipe = joblib.load(model_path)
            feats = extract_features_from_url(url)
            df = pd.DataFrame([feats])
            for c in feature_list:
                if c not in df.columns:
                    df[c] = 0
            df_inst = df[feature_list]
            with st.spinner("Computing SHAP values..."):
                try:
                    explainer, shap_values, Xb = explain_with_shap(pipe, df_inst, df_inst)
                    st.subheader("Feature Contributions (Waterfall)")
                    fig = plot_shap_waterfall(explainer, shap_values, index=0)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.subheader("Global Feature Impact (Summary)")
                    fig2 = plot_shap_summary(shap_values, Xb)
                    st.pyplot(fig2)
                    plt.close(fig2)
                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")
                    st.info("SHAP works best with tree-based models (Random Forest, Gradient Boosting).")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Train models first.")
