import streamlit as st
from src.models.predict import predict_url
from src.utils.validation import validate_url
from src.utils.io import load_yaml

st.set_page_config(page_title="Risk Assessment - SmartPhish", page_icon="🛡️", layout="wide")

st.title("🛡️ Risk Assessment & Safety Recommendations")
st.markdown("Get a calibrated risk score based on model predictions and heuristics.")

url = st.text_input("Enter URL", placeholder="https://example.com/...")
model_name = st.selectbox("Model", ["random_forest", "gradient_boosting", "logistic_regression"])

if st.button("Assess Risk"):
    if not url or not validate_url(url):
        st.error("Invalid URL")
    else:
        try:
            cfg = load_yaml("config/config.yaml")
            res = predict_url(url, model_name=model_name)
            proba = res["probability_phishing"] if res["probability_phishing"] is not None else 0.0
            # Simple risk scoring
            score = int(proba * 100)
            bands = cfg.get("risk_assessment", {}).get("bands", [])
            band = None
            for b in bands:
                if score >= b["min"] and score < b["max"]:
                    band = b
                    break
            if band is None and bands:
                band = bands[-1]
            c1, c2 = st.columns(2)
            c1.metric("Risk Score", f"{score}/100")
            if band:
                c2.metric("Risk Level", band["name"])
            if band:
                st.markdown(f"**Color:** <span style='color:{band['color']}'>{band['name']}</span>", unsafe_allow_html=True)
                st.info(f"**Recommendation:** {band['action']}")
            st.subheader("Assessment Details")
            st.json({
                "url": url,
                "prediction": res["label"],
                "phishing_probability": proba,
                "risk_score": score,
                "model": res["model"],
            })
            st.warning("⚠️ For educational/research use only. Do not rely solely on this tool for critical security decisions.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Train models first.")
