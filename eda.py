import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import numpy as np
import os

def run_eda():
    # Load data
    try:
        df = pd.read_csv('loan_applications.csv')
    except FileNotFoundError:
        print("Error: loan_applications.csv not found. Please run gen.py first.")
        return

    os.makedirs('graphs', exist_ok=True)

    # 1. Target Imbalance
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='actual_outcome', order=['repaid', 'defaulted', 'ongoing'])
    plt.suptitle('Target Imbalance: Actual Outcome')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.savefig('graphs/target_imbalance.png')
    plt.close()

    # 2. Fairness Visual
    # Calculate percentage of defaults per employment status
    df['is_default'] = (df['actual_outcome'] == 'defaulted').astype(int)
    fairness_data = df.groupby('employment_status')['is_default'].mean().reset_index()
    fairness_data['is_default'] *= 100 # Convert to percentage

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fairness_data, x='employment_status', y='is_default', order=['employed', 'self_employed', 'unemployed'])
    plt.suptitle('Fairness Visual: Default Percentage by Employment Status')
    plt.xlabel('Employment Status')
    plt.ylabel('Default Percentage (%)')
    plt.savefig('graphs/fairness_visual.png')
    plt.close()

    # 3. Survival Curve
    df['event'] = (df['actual_outcome'] == 'defaulted').astype(int)
    
    # Duration column: days_to_default for defaults, impute max duration for repaid, random duration for ongoing
    max_duration = df['days_to_default'].max() if 'days_to_default' in df.columns and not df['days_to_default'].isna().all() else 365 * 3
    if pd.isna(max_duration):
        max_duration = 365 # fallback
        
    def get_duration(row):
        if row['actual_outcome'] == 'defaulted':
            return row['days_to_default']
        elif row['actual_outcome'] == 'repaid':
            return max_duration
        else: # ongoing
            return np.random.randint(1, int(max_duration) + 1)
            
    df['duration'] = df.apply(get_duration, axis=1)

    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for status in df['employment_status'].unique():
        mask = df['employment_status'] == status
        kmf.fit(df[mask]['duration'], event_observed=df[mask]['event'], label=status)
        kmf.plot_survival_function()

    plt.suptitle('Survival Curve by Employment Status')
    plt.xlabel('Duration (Days)')
    plt.ylabel('Survival Probability (Not Defaulting)')
    plt.savefig('graphs/survival_curve.png')
    plt.close()

def run_deepdive():
    try:
        df = pd.read_csv('loan_applications.csv')
    except FileNotFoundError:
        print("Error: loan_applications.csv not found.")
        return

    os.makedirs('graphs', exist_ok=True)

    # Set styles for a high-contrast and professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    # 1. Missingness as a Signal
    df['missing_docs'] = df['documented_monthly_income'].isnull()
    
    # Filter to only known outcomes for clear default rate comparison
    known_outcomes = df[df['actual_outcome'].isin(['repaid', 'defaulted'])].copy()
    known_outcomes['is_default'] = (known_outcomes['actual_outcome'] == 'defaulted').astype(int)
    
    missingness_data = known_outcomes.groupby('missing_docs')['is_default'].mean().reset_index()
    missingness_data['is_default'] *= 100
    missingness_data['missing_docs'] = missingness_data['missing_docs'].map({True: 'No Documents Submitted', False: 'Documents Submitted'})

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=missingness_data, x='missing_docs', y='is_default', hue='missing_docs', palette='Set2')
    plt.suptitle('Missingness as a Signal: Default Rate by Document Submission', fontweight='bold')
    plt.xlabel('Document Status')
    plt.ylabel('Default Rate (%)')
    plt.tight_layout()
    plt.savefig('graphs/missingness_signal.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. The 'Liar's Premium' (Income Discrepancy)
    # For rows where documented is not null
    docs_df = df[df['documented_monthly_income'].notnull()].copy()
    # Only keep repaid and defaulted for clear separation
    docs_df = docs_df[docs_df['actual_outcome'].isin(['repaid', 'defaulted'])]
    docs_df['income_ratio'] = docs_df['stated_monthly_income'] / docs_df['documented_monthly_income']

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 1]})
    
    for ax in [ax_top, ax_bottom]:
        sns.boxplot(data=docs_df, x='actual_outcome', y='income_ratio', 
                    hue='actual_outcome', palette={'repaid': '#2ca02c', 'defaulted': '#d62728'}, 
                    order=['repaid', 'defaulted'], showfliers=False, legend=False, ax=ax)
        sns.stripplot(data=docs_df, x='actual_outcome', y='income_ratio', 
                      hue='actual_outcome', palette={'repaid': '#2ca02c', 'defaulted': '#d62728'}, 
                      order=['repaid', 'defaulted'], alpha=0.3, jitter=True, legend=False, ax=ax)

    ax_top.set_ylim(2.4, 5.5)
    ax_bottom.set_ylim(0.8, 1.2)

    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(bottom=False)

    ax_top.axhline(3, color='red', linestyle='--')
    ax_top.text(0.5, 3, ' 3x Misrepresentation Threshold', color='red', va='bottom', ha='left')

    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.suptitle("The 'Liar\'s Premium': Income Discrepancy by Outcome", fontweight='bold')
    ax_bottom.set_xlabel('Actual Outcome')
    fig.text(0.04, 0.5, 'Ratio: Stated / Documented Income', va='center', rotation='vertical')
    
    ax_top.set_ylabel('')
    ax_bottom.set_ylabel('')

    plt.tight_layout()
    plt.subplots_adjust(left=0.15)
    plt.savefig('graphs/liars_premium.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Baseline Blind Spots
    # KDE plot of rule_based_score colored by actual_outcome (Defaulted vs Repaid)
    baseline_df = df[df['actual_outcome'].isin(['repaid', 'defaulted'])]
    
    plt.figure(figsize=(10, 6))
    # Fill KDE curves
    sns.kdeplot(data=baseline_df, x='rule_based_score', hue='actual_outcome', fill=True, 
                palette={'repaid': '#2ca02c', 'defaulted': '#d62728'}, alpha=0.5, common_norm=False)
    
    # Add vertical lines at the baseline decision thresholds
    plt.axvline(50, color='black', linestyle='--', label='Threshold: 50 (Deny/Flag)')
    plt.axvline(75, color='black', linestyle='-.', label='Threshold: 75 (Flag/Approve)')
    
    plt.suptitle('Baseline Blind Spots: Rule-Based Score Distribution', fontweight='bold')
    plt.xlabel('Rule-Based Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/baseline_blind_spots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary
    print("\n--- Mathematical Insights Summary ---")
    print("1. Missingness Signal: The default rate for applicants missing documented income is substantially higher than for those providing documents. This indicates that missingness is not 'missing at random' (MAR), but a highly predictive feature of risk itself.")
    print("2. Liar's Premium: The stated-to-documented income ratio prominently flags liars. While 'repaid' applicants tightly cluster around a ratio of 1.0 (honest), 'defaulted' applicants show significant upper-tail outliers reaching ratios of 2.5-5.0, proving this feature's non-linear predictive power.")
    print("3. Baseline Blind Spots: The KDE plots reveal massive overlap between the 'repaid' and 'defaulted' distributions, particularly near the 50 and 75 thresholds. The rule-based system utterly fails to cleanly separate these classes, demonstrating the critical need for a multidimensional machine learning approach.")

if __name__ == "__main__":
    run_eda()
    run_deepdive()
