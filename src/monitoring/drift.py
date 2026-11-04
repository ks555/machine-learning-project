import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def compute_psi(expected, actual, buckets=10, eps=1e-6):
    expected = np.array(expected)
    actual = np.array(actual)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] -= eps
    breakpoints[-1] += eps
    psi = 0.0
    for i in range(len(breakpoints) - 1):
        e_perc = np.mean((expected >= breakpoints[i]) & (expected < breakpoints[i+1]))
        a_perc = np.mean((actual >= breakpoints[i]) & (actual < breakpoints[i+1]))
        e_perc = max(e_perc, eps)
        a_perc = max(a_perc, eps)
        psi += (a_perc - e_perc) * np.log(a_perc / e_perc)
    return float(psi)

def compute_ks(expected, actual):
    try:
        stat, p = ks_2samp(expected, actual)
        return float(stat), float(p)
    except Exception:
        return None, None

def detect_drift(ref_df, cur_df, features=None, psi_threshold=0.1, ks_threshold=0.05):
    if features is None:
        features = ref_df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f in cur_df.columns]
    results = []
    for f in features:
        try:
            psi = compute_psi(ref_df[f].dropna(), cur_df[f].dropna())
            ks_stat, ks_p = compute_ks(ref_df[f].dropna(), cur_df[f].dropna())
            drifted = False
            if psi is not None and psi > psi_threshold:
                drifted = True
            if ks_stat is not None and ks_p is not None and ks_p < ks_threshold:
                drifted = True
            results.append({
                "feature": f,
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_p,
                "drifted": drifted,
            })
        except Exception as e:
            results.append({"feature": f, "error": str(e)})
    return pd.DataFrame(results)
