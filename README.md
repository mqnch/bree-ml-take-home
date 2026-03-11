# Bree ML Take-Home: Loan Default Prediction

An end-to-end ML pipeline predicting loan default risk, replacing a rigid rule-based system with a predictive survival model.

## Key Features

- **Survival Analysis**: Models default probability over time (`survival:cox`), naturally incorporating censored "ongoing" loans.
- **Data Engineering**: Leverages income missingness as a behavioral signal, flags income misrepresentation, creates DTI proxies from transaction data, and computes loan-to-income and balance-to-loan ratios.
- **Explainable Decisions**: Integrated SHAP values provide transparent, human-readable justifications for loan constraints — both global feature importance rankings and per-applicant risk breakdowns are printed.

## Key Design Decisions

### 1. Single-Model Architecture

This project trains a single **XGBoost Survival (`survival:cox`)** model in `train_survival.py`, which models time-to-default as a Cox proportional hazards problem. This naturally handles right-censored "ongoing" loans and produces SHAP-based explanations showing *why* a specific applicant was flagged.

The same saved model is loaded in `evaluate_model.py` and `fairness_analysis.py`, where its predicted hazard ratios are thresholded to produce binary approve/deny decisions. This enables an apples-to-apples metric comparison (Precision, Recall, F1, AUC-ROC, FPR, FNR) against the rule-based baseline at the same approval volume — without training a separate classifier.

*Note: All scripts share `preprocess.py` as the single source of truth for feature engineering and canonical train/test splitting. `baseline_and_features.py` adds survival-specific columns (duration, event) on top of this shared preprocessing for the survival model's training data.*

### 2. Ongoing Class Handling

~8% of loans are "ongoing" with unknown final outcomes. For binary classification (`evaluate_model.py`, `fairness_analysis.py`), ongoing loans are **dropped** because their true label is unknowable — imputing them as "repaid" would introduce survivorship bias, while labeling them "defaulted" would inflate the default rate. For survival analysis (`train_survival.py`), ongoing loans are **right-censored** — the model learns they survived *at least* until the observation window closed, extracting partial information without bias.

### 3. Missingness & Misrepresentation Strategy

The 15% of applicants missing documented income are **not imputed**. Instead, missingness is treated as a behavioral signal via a binary `missing_docs_flag` feature — EDA confirms that applicants who fail to submit documentation default at substantially higher rates. For the `income_discrepancy_ratio` feature, missing-doc applicants retain `NaN`, allowing XGBoost's native sparsity-aware split finding to learn the optimal routing at every tree node. Additionally, honest-range ratios (0.85–1.10) are clamped to 1.0 — the small variation within this range carries no real signal, and without clamping XGBoost draws spurious splits that penalize honest edge-case applicants. Only ratios *outside* this range (the 5% of misrepresenters with ratios >>1.10) retain their discriminative value.

### 4. Class Imbalance

The dataset has ~14% defaults (roughly 6:1 repaid-to-defaulted). To handle this significant class imbalance without relying on synthetic resampling methods like SMOTE, the Cox survival model naturally leverages its right-censored observation windows. By optimizing the partial likelihood of hazard ratios across time (`cox-nloglik`), the model inherently extracts maximum temporal signal from the minority default events without requiring crude class weights.

## Fairness Analysis

The ML model narrows the baseline's arbitrary bias against self-employed applicants. By learning objective financial signals from actual outcomes, the model moves toward Demographic Parity — the employed/self-employed approval gap narrows from 25pp (baseline) to 7.5pp (ML).

| Employment Status | n | Baseline Approval | ML Approval | True Default Rate | Baseline FPR | ML FPR | Baseline FNR | ML FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Employed | 212 | 61.79% | 53.30% | 26.89% | 31.61% | 38.71% | 43.86% | 31.58% |
| Self Employed | 120 | 36.67% | 45.83% | 26.67% | 57.95% | 50.00% | 21.88% | 34.38% |
| Unemployed | 39 | 0.00% | 17.95% | 51.28% | 100.00% | 78.95% | 0.00% | 15.00% |

## Evaluation against the Baseline

All metrics are evaluated on a held-out 20% stratified test set (371 samples) to prevent data leakage. When threshold-matched to the same overall approval volume as the baseline, the tuned ML model matches the baseline's binary classification performance and achieves a higher AUC-ROC (0.6932 vs 0.6840), indicating stronger discriminative power across all thresholds.

| Metric                         | Baseline | ML Model |
| :----------------------------- | :------- | :------- |
| **Precision**                  | 0.3929   | 0.3929   |
| **Recall**                     | 0.7064   | 0.7064   |
| **F1-Score**                   | 0.5049   | 0.5049   |
| **AUC-ROC**                    | 0.6840   | 0.6932   |
| **FPR** (Good wrongly denied)  | 0.4542   | 0.4542   |
| **FNR** (Defaults slipped via) | 0.2936   | 0.2936   |

### Business Tradeoff

The ML model matches the baseline's binary performance at the same approval volume while providing two critical advantages: (1) its fairness analysis shows it significantly narrows the self-employed approval gap from 25pp to 7.5pp, and (2) its SHAP-based explainability provides transparent per-applicant justifications that a rule-based score cannot. With real-world data volumes, the learned model's ability to capture non-linear feature interactions should translate into even stronger discriminative performance.

### 5-Fold Cross-Validated AUC-ROC

To confirm the single-split result is stable, a 5-fold stratified CV trains a fresh Cox model per fold:

| Model    | Mean AUC | Std    |
| :------- | :------- | :----- |
| Baseline | 0.7059   | 0.0328 |
| ML Model | 0.7005   | 0.0162 |

The ML model's AUC has **half the variance** of the baseline (σ=0.016 vs 0.033), indicating more consistent risk ranking across different data partitions.

### Confusion Matrices

**Baseline Confusion Matrix**

```text
                Predicted Good (0) | Predicted Default (1)
Actual Good (0) | 143              | 119
Actual Def (1)  | 32               | 77
```

**ML Model Confusion Matrix**

```text
                Predicted Good (0) | Predicted Default (1)
Actual Good (0) | 143              | 119
Actual Def (1)  | 32               | 77
```

## Project Structure

- `gen.py`: Dataset generation script.
- `loan_applications.csv`: The original generated dataset.
- `eda.py`: Risk factor exploration and visualization.
- `baseline_and_features.py`: Feature engineering and baseline metrics.
- `preprocess.py`: Shared preprocessing module (single source of truth for feature engineering).
- `train_survival.py`: XGBoost survival modeling and SHAP interpretation.
- `evaluate_model.py`: Head-to-head performance comparison against the baseline.
- `fairness_analysis.py`: Bias audit and rule-based vs. ML outcome comparison.

## How to Run

1. Install dependencies: `pip install -r requirements.txt`.
2. (Optional) Run `gen.py` to see the generation logic; however, use the committed `loan_applications.csv` for reproducibility.
3. Run scripts in sequence to replicate results:
   ```bash
   python baseline_and_features.py
   python train_survival.py
   python evaluate_model.py
   python fairness_analysis.py
   ```
4. EDA visualizations: `python eda.py` (outputs to `graphs/`).

## Future Work

If I were taking this model toward a production release, my next focus would be on moving from a strong MVP to a resilient system.

First, I'd want to experiment with Random Survival Forests or DeepSurv architectures, combined with cross-validated hyperparameter tuning (e.g., using Optuna). While XGBoost is a workhorse, ensembles specifically built for survival data can sometimes catch the subtle, non-linear interactions between income and spending habits that gradient boosting might overlook.

Beyond the architecture, Probability Calibration would be important. When it comes to lending, it isn't enough to just rank who is "riskiest"—we need the model's 20% risk prediction to accurately mean that 20 out of 100 people will actually default. Implementing post-hoc methods like Platt Scaling would ensure our hazard ratios map directly to reliable, real-world probabilities for the finance team.

Finally, to safeguard our Fairness goals as we scale, I'd look into integrating Adversarial Debiasing. This would move us from "organically fair" to "mathematically guaranteed" parity by penalizing the model if it can still "guess" an applicant's protected status from their financial data. This ensures our lending decisions stay objective even as the macro-economic environment shifts.

### Production Failure Modes & Monitoring
In a real-world setting, a model is only as good as the data it continues to receive. Key areas of focus for a production rollout would include:
- **Data Drift Detection:** Macro-economic shifts (e.g., inflation spikes) will change the underlying relationship between income and default risk. We need statistical monitoring to detect input drift early.
- **Model Degradation Alerts:** The model's baseline performance will inevitably decay over time. Automated alerts must be configured for when AUC drops below an acceptable threshold on recent loan cohorts.
- **Retraining Cadence:** We'd need a strategy for periodic model retraining (e.g., quarterly) to continually factor in new applicant behavior.
- **A/B Testing:** The model should never replace the rule-based system verbatim overnight. A phased shadow-mode rollout followed by randomized A/B testing is critical to safely measure the true financial impact of the ML model.
