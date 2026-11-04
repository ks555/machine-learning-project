import streamlit as st

st.set_page_config(page_title="About & Recommendations - SmartPhish", page_icon="📚", layout="wide")

st.title("📚 About SmartPhish & Safety Recommendations")

st.header("About the Project")
st.markdown("""
**SmartPhish** is an advanced cybersecurity research tool that combines Machine Learning, AI, and Explainable AI (XAI) for phishing detection and risk assessment.

**Objectives:**
- Detect potentially phishing URLs using lexical/structural features
- Compare multiple ML algorithms and ensembles
- Provide interpretable explanations via SHAP
- Assess risk with actionable recommendations
- Enable monitoring for data/model drift
- Demonstrate end-to-end ML workflow

**Disclaimer:** This tool is for research and educational purposes only. Predictions indicate *potential risk* and do not guarantee safety or maliciousness.
""")

st.header("Phishing Safety Recommendations")
st.markdown("""
### General Best Practices
- **Verify URLs**: Check domain spelling carefully (lookalikes like `arnazon.com`)
- **Use HTTPS cautiously**: HTTPS doesn't guarantee legitimacy
- **Don't rush**: Urgent messages are common phishing tactics
- **Check sender**: Verify email/SMS senders through official channels
- **Use MFA**: Enable multi-factor authentication on important accounts
- **Keep updated**: Update browser, OS, and security software
- **Use password managers**: They help detect fake login pages

### If You Suspect Phishing
1. **Don't click** suspicious links or download attachments
2. **Don't enter** credentials, banking, or personal info
3. **Verify directly** via official website/app (not the message link)
4. **Report** to your organization/anti-phishing service (e.g., reportphishing@apwg.org, or your email provider)
5. **Change passwords** if you may have entered them
6. **Monitor accounts** for unauthorized activity

### Red Flags
- Misspelled domains or unusual TLDs
- IP addresses instead of domain names
- URL shorteners in unexpected contexts
- Requests for sensitive info via email/SMS
- Poor grammar/spelling in official-looking messages
- Mismatched URLs on hover
""")

st.header("Technical Notes")
st.markdown("""
- **Features**: Primarily URL-lexical for speed and reproducibility (no forced live crawling)
- **Models**: Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, Voting/Stacking
- **XAI**: SHAP for feature attributions
- **Drift**: PSI and KS tests for distribution shifts
- **Risk**: Probability-based scoring with configurable bands
""")
