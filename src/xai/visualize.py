import matplotlib.pyplot as plt
import shap

def plot_shap_summary(shap_values, X, max_display=20):
    plt.figure()
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    return fig

def plot_shap_waterfall(explainer, shap_values, index=0, max_display=15):
    plt.figure()
    try:
        shap.plots.waterfall(shap_values[index], max_display=max_display, show=False)
    except Exception:
        shap.summary_plot(shap_values, show=False)
    fig = plt.gcf()
    return fig
