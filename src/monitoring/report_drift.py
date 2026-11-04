import pandas as pd
from pathlib import Path
from src.utils.io import save_json

def generate_drift_report(drift_df, out_path="reports/metrics/drift_report.json"):
    report = {
        "total_features": len(drift_df),
        "drifted_features": int(drift_df["drifted"].sum()) if "drifted" in drift_df.columns else 0,
        "details": drift_df.to_dict(orient="records"),
    }
    save_json(report, out_path)
    return report
