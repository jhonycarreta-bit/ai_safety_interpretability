import shap
import pandas as pd

def explain_model(model, X_test: pd.DataFrame):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    shap.save_html("shap_summary.html")
    print("SHAP summary saved to shap_summary.html")
