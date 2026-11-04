# SmartPhish: Advanced AI-Based Phishing Detection and Risk Assessment System

SmartPhish is an advanced cybersecurity project that uses Machine Learning (ML), Artificial Intelligence (AI), and Explainable AI (XAI) to identify potentially phishing websites and assess their security risks. It analyzes website URLs and extracted features to classify them as potentially phishing or legitimate, providing interpretable risk assessments, visual analytics, and safety recommendations via an interactive Streamlit web app.

> **Disclaimer:** SmartPhish is intended as a research and educational tool. Predictions indicate *potential risk* and do not guarantee a website is safe or malicious.

## Key Features

- **URL Feature Engineering**: Extracts lexical, structural, and host-based features (length, special characters, subdomains, entropy, presence of IPs, suspicious TLDs, redirection hints, etc.).
- **Multi-Model Comparison**: Implements and compares Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting, and ensemble techniques (Voting/Stacking).
- **Advanced ML**: Supports feature selection (correlation/Mutual Information), hyperparameter tuning (Grid/Random/SearchCV), class imbalance handling (class_weight/SMOTE), and optional deep learning experiments.
- **Explainable AI (XAI)**: Integrates SHAP (and hooks for LIME) to show per-prediction feature contributions and global feature importance.
- **Risk Assessment Engine**: Translates model probabilities into a calibrated risk score (0-100) with severity bands (Low/Medium/High/Critical) and actionable safety recommendations.
- **Interactive Streamlit App**: Multi-page UI for URL scanning, model comparison, explainability, risk assessment, monitoring/drift, and guidance.
- **Continuous Improvement Ready**: Designed to ingest new labeled data, evaluate on unseen examples, retrain when appropriate, and monitor model/data drift.
- **Reproducible Pipeline**: Modular `src/` code, config-driven paths/params, saved artifacts (models, scalers, feature lists, metrics), and training scripts.

## Architecture

```text
Data (raw/interim/processed) -> Feature Extraction (src/features) 
  -> Preprocessing (scaling/imputation/selection) 
    -> Model Training/Eval (src/models) 
      -> Artifacts (models/) 
        -> Inference + XAI (src/xai, src/models/predict.py) 
          -> Streamlit App (app/) + Monitoring (src/monitoring)
```

## Quickstart

### 1) Environment setup
```bash
# Create and activate venv (example)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2) Prepare data
Place labeled URL datasets in `data/raw/`. Common public sources (for research/education): [PhishTank](https://www.phishtank.com/), [OpenPhish](https://openphish.com/), [Kaggle phishing URL datasets](https://www.kaggle.com/search?q=phishing+url+dataset). 
Expected schema (flexible): a `url` column and a `label` column (`0` = legitimate, `1` = phishing) or `phishing`/`legitimate`. The loaders in `src/data/` are written to be robust to common variants.

### 3) Extract features & preprocess
```bash
python scripts/extract_features.py --input data/raw/your_dataset.csv --output data/processed/features.csv
```
Or use the training script which can orchestrate end-to-end.

### 4) Train models
```bash
# Train all configured models with default settings
python scripts/train_all.py

# Or use src.models.train programmatically
```
Trained artifacts will be saved to `models/` (e.g., `rf_model.joblib`, `preprocessor.joblib`, `feature_list.json`, `metrics.json`).

### 5) Run the app
```bash
streamlit run app/Home.py
```
Navigate to the multi-page app to scan URLs, compare models, explain predictions, assess risk, and view monitoring tools.

## Configuration

- `config/config.yaml`: model lists, hyperparams, CV, class balancing, XAI toggles, risk scoring weights/bands.
- `config/paths.yaml`: all filesystem paths (data/models/reports).

## Feature extraction notes

Extracted features are primarily **URL-lexical** (fast, offline, no live crawling required by default). You can extend with:
- Host-based (DNS/WHOIS) - use cautiously with timeouts/caching
- Content-based (HTML/JS) - requires fetching; may be blocked/rate-limited
- Certificate/redirect chains - heavier

The implementation favors safe, reproducible features suitable for batch training and real-time inference.

## Explainability & Risk

- **SHAP**: TreeExplainer for tree models, Kernel/Explainer fallbacks where applicable. Values are aggregated for global/local views.
- **Risk score**: Combines calibrated probability with feature-risk heuristics and severity mapping. Configurable in `config/config.yaml`.
- **Recommendations**: Rule-based safety guidance per risk band.

## Monitoring & Drift

`src/monitoring/drift.py` includes utilities for:
- **Data drift**: Population Stability Index (PSI), feature distributions (KS/Wasserstein) on reference vs current batches
- **Prediction drift**: Shift in predicted probabilities/class proportions
- **Concept drift**: Performance tracking over time (requires new labels)

Use the "Monitoring & Drift" page to upload a current batch and compare against a saved reference (e.g., training distribution).

## Extending

- Add models: register in `src/models/registry.py` and `config/config.yaml`
- Add features: extend `src/features/url_features.py` and update `feature_list.json`
- Add XAI methods: add modules under `src/xai/`
- Deep learning: plug a PyTorch/Keras trainer in `src/models/` (optional deps noted)

## Security & Ethics

- Only analyze URLs you are authorized to test.
- Avoid storing sensitive URLs unnecessarily.
- Respect rate limits and Terms of Service for external lookups.
- This tool is for defensive research/education.

## License

See LICENSE (add your preferred license, e.g., MIT).

## Acknowledgments

Inspired by phishing detection literature (e.g., lexical URL analysis, ensemble learning, XAI for security). Compatible with common academic datasets.