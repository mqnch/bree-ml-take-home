import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# --- Premium Aesthetic Settings ---
sns.set_theme(
    style="whitegrid",
    palette="crest",
    rc={
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
})
# ----------------------------------

def main():
    # 1. Data Setup
    print("Loading data...")
    df = pd.read_csv('survival_features.csv')

    # Separate targets
    y_duration = df['duration'].values
    y_event = df['event'].values

    # XGBoost Cox expects y to be duration, with negative durations for censored data
    y_xgb = np.where(y_event == 1, y_duration, -y_duration)

    # Features
    X = df.drop(columns=['duration', 'event'])

    # 2. Model Training
    print("Training XGBoost Survival (Cox) model...")
    # Using 'survival:cox' objective. 
    # For Cox models in XGBoost, the output is the log hazard ratio.
    # Higher value indicates higher risk of event.
    xgb_model = xgb.XGBRegressor(
        objective='survival:cox',
        tree_method='hist',
        eval_metric='cox-nloglik',
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=150,
        random_state=42
    )

    xgb_model.fit(X, y_xgb)

    # 3. Evaluation
    print("Evaluating model...")
    preds = xgb_model.predict(X)

    # Calculate Concordance Index (C-index)
    # concordance_index assumes higher predictions mean longer survival.
    # Since XGBoost survival:cox predicts risk (higher = shorter survival / higher hazard),
    # we pass -preds to the concordance_index function.
    c_index = concordance_index(y_duration, -preds, y_event)
    print(f"Concordance Index (C-index): {c_index:.4f}")

    # 4. Interpretability (SHAP)
    print("Generating SHAP Explanations...")
    os.makedirs('graphs', exist_ok=True)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)

    # A. SHAP Summary Plot (Beeswarm)
    plt.figure()
    # Passing show=False so we can save and add titles
    shap.summary_plot(shap_values, X, show=False)
    plt.suptitle("SHAP Summary Plot: Global Feature Importance & Impact", y=0.98)
    sns.despine()
    plt.savefig('graphs/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved SHAP Summary Plot to 'graphs/shap_summary_plot.png'")

    # B. SHAP Waterfall plot for a specific high-risk applicant
    # Let's find an applicant that is high risk and actually defaulted
    default_indices = np.where(y_event == 1)[0]
    # preds[default_indices] gives risks for these individuals. Argmax gives highest risk.
    high_risk_index = default_indices[np.argmax(preds[default_indices])]

    plt.figure()
    # shap.waterfall_plot takes an Explanation object
    shap.waterfall_plot(shap_values[high_risk_index], show=False)
    plt.suptitle(f"SHAP Waterfall Plot for High-Risk Applicant (Index {high_risk_index})", y=0.98)
    sns.despine()
    plt.savefig('graphs/shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP Waterfall Plot to 'graphs/shap_waterfall_plot.png' for applicant {high_risk_index}")

    # 5. Save the model
    model_path = 'xgboost_survival_model.json'
    xgb_model.save_model(model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()
