import streamlit as st

st.set_page_config(
    page_title="SmartPhish - Home",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ SmartPhish: Advanced AI-Based Phishing Detection & Risk Assessment")
st.markdown("""
**SmartPhish** uses Machine Learning, AI, and Explainable AI (XAI) to identify potentially phishing websites and assess their security risks.

> *Predictions indicate potential risk rather than guaranteeing that a website is safe or malicious.*

### Features
- **🔍 URL Scan**: Analyze any URL for phishing indicators in real-time
- **📊 Model Comparison**: Compare performance of multiple ML models
- **🧠 Explainability (XAI)**: Understand which features drive predictions with SHAP
- **🛡️ Risk Assessment**: Get calibrated risk scores with actionable recommendations
- **📈 Monitoring & Drift**: Track data/prediction drift over time
- **📚 About & Recommendations**: Learn about phishing and best practices

### Getting Started
1. Train models using `scripts/train_all.py` (place data in `data/raw/`)
2. Use the **URL Scan** page to test URLs
3. Explore model insights and explanations

**Navigate using the sidebar to access different modules.**
""")

st.info("Tip: For best results, train models on a representative dataset before using URL scanning.")
