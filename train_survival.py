import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

    # 2. Train/Test Split (Stratified by event, 80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, dur_train, dur_test, evt_train, evt_test = (
        train_test_split(
            X, y_xgb, y_duration, y_event,
            test_size=0.2, stratify=y_event, random_state=42,
        )
    )

    # 3. Model Training (on train set only)
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

    xgb_model.fit(X_train, y_train)

    # 4. Evaluation (on held-out test set only)
    print("Evaluating model on held-out test set...")
    preds_test = xgb_model.predict(X_test)

    # Calculate Concordance Index (C-index)
    # concordance_index assumes higher predictions mean longer survival.
    # Since XGBoost survival:cox predicts risk (higher = shorter survival / higher hazard),
    # we pass -preds to the concordance_index function.
    c_index = concordance_index(dur_test, -preds_test, evt_test)
    print(f"Concordance Index (C-index) on test set: {c_index:.4f}")

    # 5. Interpretability (SHAP — explanatory, evaluated on test set)
    print("Generating SHAP Explanations on unseen test set...")
    os.makedirs('graphs', exist_ok=True)

    explainer = shap.TreeExplainer(xgb_model)
    # Generate explanations using the hold-out set to represent behavior in production
    shap_values = explainer(X_test)

    # A. SHAP Summary Plot (Beeswarm)
    plt.figure()
    # Passing show=False so we can save and add titles
    shap.summary_plot(shap_values, X_test, show=False)
    plt.suptitle("SHAP Summary Plot: Global Feature Importance & Impact (Test Set)", y=0.98)
    sns.despine()
    plt.savefig('graphs/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved SHAP Summary Plot to 'graphs/shap_summary_plot.png'")

    # B. SHAP Waterfall plot for a specific high-risk applicant
    # Evaluate risk on the held-out test set
    default_indices_test = np.where(evt_test == 1)[0]
    # Identify the highest risk applicant *in the test set*
    highest_risk_idx_test = default_indices_test[np.argmax(preds_test[default_indices_test])]

    plt.figure()
    # shap.waterfall_plot takes an Explanation object
    shap.waterfall_plot(shap_values[highest_risk_idx_test], show=False)
    plt.suptitle(f"SHAP Waterfall Plot for High-Risk Applicant (Test Index {highest_risk_idx_test})", y=0.98)
    sns.despine()
    plt.savefig('graphs/shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP Waterfall Plot to 'graphs/shap_waterfall_plot.png' for test applicant {highest_risk_idx_test}")

    # 5. Save the model
    model_path = 'xgboost_survival_model.json'
    xgb_model.save_model(model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()
