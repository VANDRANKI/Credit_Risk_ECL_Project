"""
CECL Credit Risk Modeling - Master Execution Script
====================================================
This script runs the complete analysis pipeline.

Author: Prabhu Vandrap
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import joblib

warnings.filterwarnings('ignore')

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data_raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data_processed')
OUTPUTS_FIGURES = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
OUTPUTS_MODELS = os.path.join(PROJECT_ROOT, 'outputs', 'models')

# Create directories if they don't exist
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(OUTPUTS_FIGURES, exist_ok=True)
os.makedirs(OUTPUTS_MODELS, exist_ok=True)

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 11

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print("CECL CREDIT RISK MODELING - AGRICULTURAL LOAN PORTFOLIO")
print("="*70)

# ============================================================================
# STEP 1: DATA ACQUISITION AND EDA
# ============================================================================
print("\n" + "="*70)
print("STEP 1: DATA ACQUISITION AND EDA")
print("="*70)

# Load data
DATA_PATH = os.path.join(DATA_RAW, 'LC_loans_granting_model_dataset.csv')
print(f"\nLoading data from: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH)
print(f"Raw data shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

# Data cleaning
print("\nCleaning data...")
df = df_raw.copy()

# Remove invalid loan amounts
df = df[df['loan_amnt'] > 0]

# Clean FICO scores (valid range: 300-850)
df = df[(df['fico_n'] >= 300) & (df['fico_n'] <= 850)]

# Clean income
df = df[(df['revenue'] > 0) & (df['revenue'] <= 10000000)]

# Clean DTI
df = df[df['dti_n'] >= 0]
df['dti_n'] = df['dti_n'].clip(upper=100)

# Key columns for modeling
key_columns = ['loan_amnt', 'revenue', 'dti_n', 'fico_n', 'purpose',
               'home_ownership_n', 'addr_state', 'emp_length', 'Default']
df = df.dropna(subset=key_columns)

# Parse dates
df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
df['issue_year'] = df['issue_date'].dt.year
df['issue_month'] = df['issue_date'].dt.month

# Create FICO buckets
df['fico_bucket'] = pd.cut(df['fico_n'],
                           bins=[300, 579, 619, 659, 699, 739, 779, 850],
                           labels=['300-579', '580-619', '620-659', '660-699',
                                   '700-739', '740-779', '780-850'])

# Create income buckets
df['income_bucket'] = pd.cut(df['revenue'],
                             bins=[0, 40000, 60000, 80000, 100000, 150000, 10000000],
                             labels=['<$40K', '$40K-$60K', '$60K-$80K',
                                     '$80K-$100K', '$100K-$150K', '>$150K'])

print(f"Cleaned data shape: {df.shape[0]:,} rows")
print(f"Default rate: {df['Default'].mean()*100:.2f}%")

# Save cleaned data
df_clean = df[['id', 'issue_date', 'issue_year', 'issue_month',
               'loan_amnt', 'revenue', 'dti_n', 'fico_n',
               'emp_length', 'purpose', 'home_ownership_n', 'addr_state',
               'fico_bucket', 'income_bucket', 'Default']].copy()
df_clean.to_csv(os.path.join(DATA_PROCESSED, 'loans_cleaned.csv'), index=False)
print(f"Cleaned data saved to: data_processed/loans_cleaned.csv")

# ============================================================================
# STEP 2: AGRICULTURAL PORTFOLIO SEGMENTATION
# ============================================================================
print("\n" + "="*70)
print("STEP 2: AGRICULTURAL PORTFOLIO SEGMENTATION")
print("="*70)

# Define agricultural criteria
AGRI_PURPOSES = ['small_business']
AGRI_STATES = ['CA', 'IA', 'NE', 'TX', 'MN', 'IL', 'KS', 'WI', 'IN', 'NC']

# Create flags
df['is_agri_portfolio'] = ((df['purpose'].isin(AGRI_PURPOSES)) &
                           (df['addr_state'].isin(AGRI_STATES))).astype(int)
df['is_agri_state'] = df['addr_state'].isin(AGRI_STATES).astype(int)
df['is_small_business'] = df['purpose'].isin(AGRI_PURPOSES).astype(int)

print(f"\nAgricultural portfolio loans: {df['is_agri_portfolio'].sum():,} ({df['is_agri_portfolio'].mean()*100:.2f}%)")
print(f"Agricultural portfolio default rate: {df[df['is_agri_portfolio']==1]['Default'].mean()*100:.2f}%")
print(f"Non-agricultural default rate: {df[df['is_agri_portfolio']==0]['Default'].mean()*100:.2f}%")

# Save with agri flag
df.to_csv(os.path.join(DATA_PROCESSED, 'loans_with_agri_flag.csv'), index=False)
print(f"Data with agri flag saved to: data_processed/loans_with_agri_flag.csv")

# ============================================================================
# STEP 3: PD MODEL DEVELOPMENT
# ============================================================================
print("\n" + "="*70)
print("STEP 3: PD MODEL DEVELOPMENT")
print("="*70)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("Using XGBoost for advanced model")
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available, using RandomForest")

# Feature engineering
print("\nEngineering features...")

# Derived features
df['loan_to_income'] = df['loan_amnt'] / (df['revenue'] + 1)
df['income_per_dti'] = df['revenue'] / (df['dti_n'] + 1)

# FICO bucket numeric
fico_bucket_map = {'300-579': 1, '580-619': 2, '620-659': 3, '660-699': 4,
                   '700-739': 5, '740-779': 6, '780-850': 7}
df['fico_bucket_num'] = df['fico_bucket'].astype(str).map(fico_bucket_map)

# Employment length numeric
emp_length_map = {'< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
                  '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                  '8 years': 8, '9 years': 9, '10+ years': 10}
df['emp_length_num'] = df['emp_length'].map(emp_length_map)
df['emp_length_num'] = df['emp_length_num'].fillna(df['emp_length_num'].median())

# Handle infinities
df['loan_to_income'] = df['loan_to_income'].replace([np.inf, -np.inf], np.nan)
df['income_per_dti'] = df['income_per_dti'].replace([np.inf, -np.inf], np.nan)
df['loan_to_income'] = df['loan_to_income'].fillna(df['loan_to_income'].median())
df['income_per_dti'] = df['income_per_dti'].fillna(df['income_per_dti'].median())
df['fico_bucket_num'] = pd.to_numeric(df['fico_bucket_num'], errors='coerce')
df['fico_bucket_num'] = df['fico_bucket_num'].fillna(4)  # Default to middle bucket

# One-hot encode categorical features
top_purposes = df['purpose'].value_counts().head(10).index.tolist()
df['purpose_grouped'] = df['purpose'].apply(lambda x: x if x in top_purposes else 'other')
purpose_dummies = pd.get_dummies(df['purpose_grouped'], prefix='purpose')
home_dummies = pd.get_dummies(df['home_ownership_n'], prefix='home')
df = pd.concat([df, purpose_dummies, home_dummies], axis=1)

# Define features
NUMERIC_FEATURES = ['loan_amnt', 'revenue', 'dti_n', 'fico_n',
                    'loan_to_income', 'income_per_dti', 'fico_bucket_num', 'emp_length_num']
DUMMY_FEATURES = list(purpose_dummies.columns) + list(home_dummies.columns)
ALL_FEATURES = NUMERIC_FEATURES + DUMMY_FEATURES

print(f"Total features: {len(ALL_FEATURES)}")

# Prepare data
X = df[ALL_FEATURES].copy()
y = df['Default'].copy()

# Time-based split
year_counts = df.groupby('issue_year').size().cumsum()
split_threshold = len(df) * 0.70
split_year = year_counts[year_counts <= split_threshold].index[-1]

train_mask = df['issue_year'] <= split_year
test_mask = df['issue_year'] > split_year

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nTrain set: {len(X_train):,} samples (years <= {split_year})")
print(f"Test set: {len(X_test):,} samples (years > {split_year})")

# Scale features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000,
                               class_weight='balanced', solver='lbfgs')
lr_model.fit(X_train_scaled, y_train)
lr_pred_test = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred_test)
print(f"Logistic Regression Test AUC: {lr_auc:.4f}")

# Train Advanced Model
print("\nTraining Advanced Model...")
if XGB_AVAILABLE:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    advanced_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, eval_metric='auc',
        use_label_encoder=False
    )
    MODEL_NAME = 'XGBoost'
else:
    advanced_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=20,
        min_samples_leaf=10, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    MODEL_NAME = 'RandomForest'

advanced_model.fit(X_train, y_train)
adv_pred_test = advanced_model.predict_proba(X_test)[:, 1]
adv_auc = roc_auc_score(y_test, adv_pred_test)
print(f"{MODEL_NAME} Test AUC: {adv_auc:.4f}")

# Select best model
if adv_auc > lr_auc:
    SELECTED_MODEL = advanced_model
    SELECTED_MODEL_NAME = MODEL_NAME
    SELECTED_AUC = adv_auc
    USE_SCALED = False
else:
    SELECTED_MODEL = lr_model
    SELECTED_MODEL_NAME = 'Logistic Regression'
    SELECTED_AUC = lr_auc
    USE_SCALED = True

print(f"\nSelected Model: {SELECTED_MODEL_NAME} (AUC: {SELECTED_AUC:.4f})")

# Generate PD predictions for all loans
print("\nGenerating PD predictions...")
if USE_SCALED:
    X_all_scaled = X.copy()
    X_all_scaled[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
    df['pd_hat'] = SELECTED_MODEL.predict_proba(X_all_scaled)[:, 1]
else:
    df['pd_hat'] = SELECTED_MODEL.predict_proba(X)[:, 1]

print(f"Mean PD: {df['pd_hat'].mean():.4f}")
print(f"Median PD: {df['pd_hat'].median():.4f}")

# Save models
joblib.dump(SELECTED_MODEL, os.path.join(OUTPUTS_MODELS, 'pd_model_selected.joblib'))
joblib.dump(scaler, os.path.join(OUTPUTS_MODELS, 'feature_scaler.joblib'))

model_metadata = {
    'selected_model': SELECTED_MODEL_NAME,
    'test_auc': float(SELECTED_AUC),
    'features': ALL_FEATURES,
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test))
}
with open(os.path.join(OUTPUTS_MODELS, 'model_metadata.json'), 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"Models saved to: outputs/models/")

# Save data with PD
df.to_csv(os.path.join(DATA_PROCESSED, 'loans_with_pd.csv'), index=False)

# ============================================================================
# STEP 4: LGD AND EAD ESTIMATION
# ============================================================================
print("\n" + "="*70)
print("STEP 4: LGD AND EAD ESTIMATION")
print("="*70)

# LGD assumption (Basel II standard for unsecured)
LGD_ASSUMPTION = 0.45
df['lgd_est'] = LGD_ASSUMPTION

# EAD (loan amount)
df['ead_est'] = df['loan_amnt']

print(f"\nLGD Assumption: {LGD_ASSUMPTION*100:.0f}%")
print(f"Total EAD: ${df['ead_est'].sum():,.0f}")
print(f"Mean EAD: ${df['ead_est'].mean():,.0f}")

# Save
df.to_csv(os.path.join(DATA_PROCESSED, 'loans_with_lgd_ead.csv'), index=False)

# ============================================================================
# STEP 5: ECL COMPUTATION
# ============================================================================
print("\n" + "="*70)
print("STEP 5: ECL COMPUTATION")
print("="*70)

# Calculate ECL
df['ecl_est'] = df['pd_hat'] * df['lgd_est'] * df['ead_est']
df['ecl_rate'] = df['ecl_est'] / df['ead_est']

# Portfolio metrics
total_ecl = df['ecl_est'].sum()
total_ead = df['ead_est'].sum()
portfolio_ecl_rate = total_ecl / total_ead * 100

# Agricultural segment
agri_ecl = df[df['is_agri_portfolio']==1]['ecl_est'].sum()
agri_ead = df[df['is_agri_portfolio']==1]['ead_est'].sum()
agri_ecl_rate = agri_ecl / agri_ead * 100

# Non-agricultural segment
non_agri_ecl = df[df['is_agri_portfolio']==0]['ecl_est'].sum()
non_agri_ead = df[df['is_agri_portfolio']==0]['ead_est'].sum()
non_agri_ecl_rate = non_agri_ecl / non_agri_ead * 100

print(f"\nPORTFOLIO ECL SUMMARY")
print(f"-"*50)
print(f"Total Portfolio ECL:     ${total_ecl:,.0f}")
print(f"Portfolio ECL Rate:      {portfolio_ecl_rate:.4f}%")
print(f"\nAgricultural ECL:        ${agri_ecl:,.0f}")
print(f"Agricultural ECL Rate:   {agri_ecl_rate:.4f}%")
print(f"\nNon-Agricultural ECL:    ${non_agri_ecl:,.0f}")
print(f"Non-Agricultural ECL Rate: {non_agri_ecl_rate:.4f}%")

# Save
df.to_csv(os.path.join(DATA_PROCESSED, 'loans_with_ecl.csv'), index=False)

# Save ECL metadata
ecl_metadata = {
    'total_loans': int(len(df)),
    'total_ead': float(total_ead),
    'total_ecl': float(total_ecl),
    'portfolio_ecl_rate': float(portfolio_ecl_rate),
    'agricultural_ecl': float(agri_ecl),
    'agricultural_ecl_rate': float(agri_ecl_rate),
    'non_agricultural_ecl': float(non_agri_ecl),
    'non_agricultural_ecl_rate': float(non_agri_ecl_rate)
}
with open(os.path.join(OUTPUTS_MODELS, 'ecl_metadata.json'), 'w') as f:
    json.dump(ecl_metadata, f, indent=2)

# ============================================================================
# STEP 6: STRESS TESTING
# ============================================================================
print("\n" + "="*70)
print("STEP 6: STRESS TESTING")
print("="*70)

# Define scenarios
SCENARIOS = {
    'Baseline': {'income_change': 0.00, 'dti_change': 0.00},
    'Moderate Stress': {'income_change': -0.10, 'dti_change': 0.15},
    'Severe Stress': {'income_change': -0.20, 'dti_change': 0.30},
    'Agricultural Crisis': {'income_change': -0.25, 'dti_change': 0.35}
}

# Store baseline metrics
baseline_total_ecl = total_ecl
baseline_agri_ecl = agri_ecl
baseline_non_agri_ecl = non_agri_ecl

stress_results = {}

print("\nApplying stress scenarios...")

for scenario_name, params in SCENARIOS.items():
    income_change = params['income_change']
    dti_change = params['dti_change']

    if scenario_name == 'Baseline':
        pd_stressed = df['pd_hat'].copy()
    else:
        # Calculate PD multiplier
        pd_multiplier = (1 - income_change) * (1 + abs(dti_change) * 0.5)
        pd_stressed = (df['pd_hat'] * pd_multiplier).clip(upper=0.95)

    # Calculate stressed ECL
    ecl_stressed = pd_stressed * df['lgd_est'] * df['ead_est']

    # Aggregate
    total_ecl_stressed = ecl_stressed.sum()
    agri_ecl_stressed = ecl_stressed[df['is_agri_portfolio']==1].sum()
    non_agri_ecl_stressed = ecl_stressed[df['is_agri_portfolio']==0].sum()

    # Calculate changes
    total_change = (total_ecl_stressed - baseline_total_ecl) / baseline_total_ecl * 100
    agri_change = (agri_ecl_stressed - baseline_agri_ecl) / baseline_agri_ecl * 100
    non_agri_change = (non_agri_ecl_stressed - baseline_non_agri_ecl) / baseline_non_agri_ecl * 100

    stress_results[scenario_name] = {
        'total_ecl': float(total_ecl_stressed),
        'total_ecl_change': float(total_change),
        'agri_ecl': float(agri_ecl_stressed),
        'agri_ecl_change': float(agri_change),
        'non_agri_ecl': float(non_agri_ecl_stressed),
        'non_agri_ecl_change': float(non_agri_change)
    }

    print(f"\n{scenario_name}:")
    print(f"  Total ECL: ${total_ecl_stressed:,.0f} ({total_change:+.1f}%)")
    print(f"  Agri ECL:  ${agri_ecl_stressed:,.0f} ({agri_change:+.1f}%)")

# Save stress test results
with open(os.path.join(OUTPUTS_MODELS, 'stress_test_results.json'), 'w') as f:
    json.dump(stress_results, f, indent=2)

# ============================================================================
# GENERATE KEY VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Set clean style - no grid
plt.style.use('seaborn-v0_8-white')

# Figure 1: Default Rate by FICO
fig, ax = plt.subplots(figsize=(12, 6))
fico_default = df.groupby('fico_bucket', observed=True)['Default'].mean() * 100
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(fico_default)))
bars = ax.bar(range(len(fico_default)), fico_default.values, color=colors, edgecolor='black')
ax.set_xticks(range(len(fico_default)))
ax.set_xticklabels(fico_default.index, rotation=45, fontweight='bold')
ax.set_xlabel('FICO Score Bucket', fontweight='bold', fontsize=12)
ax.set_ylabel('Default Rate (%)', fontweight='bold', fontsize=12)
ax.set_title('Default Rate by FICO Score Bucket', fontweight='bold', fontsize=14)
ax.tick_params(axis='both', labelsize=10)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, fico_default.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, '02_default_rate_by_fico.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 02_default_rate_by_fico.png")

# Figure 2: ECL by Segment
fig, ax = plt.subplots(figsize=(10, 6))
segments = ['Agricultural', 'Non-Agricultural']
ecls = [agri_ecl/1e6, non_agri_ecl/1e6]
colors = ['#2ecc71', '#3498db']
bars = ax.bar(segments, ecls, color=colors, edgecolor='black')
ax.set_ylabel('Total ECL ($ Millions)', fontweight='bold', fontsize=12)
ax.set_title('Total ECL by Portfolio Segment', fontweight='bold', fontsize=14)
ax.tick_params(axis='both', labelsize=10)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, ecls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${val:.1f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, '17_ecl_by_segment.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 17_ecl_by_segment.png")

# Figure 3: Stress Test Results
fig, ax = plt.subplots(figsize=(12, 6))
scenarios_list = list(SCENARIOS.keys())
ecl_changes = [stress_results[s]['total_ecl_change'] for s in scenarios_list]
colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
bars = ax.bar(range(len(scenarios_list)), ecl_changes, color=colors, edgecolor='black')
ax.set_xticks(range(len(scenarios_list)))
ax.set_xticklabels(scenarios_list, rotation=15, fontweight='bold')
ax.set_ylabel('ECL Change from Baseline (%)', fontweight='bold', fontsize=12)
ax.set_title('ECL Impact Under Stress Scenarios', fontweight='bold', fontsize=14)
ax.tick_params(axis='both', labelsize=10)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bar, val in zip(bars, ecl_changes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:+.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, '22_stress_test_results.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 22_stress_test_results.png")

# Figure 4: Agricultural Vulnerability
fig, ax = plt.subplots(figsize=(12, 6))
stress_scenarios = ['Moderate Stress', 'Severe Stress', 'Agricultural Crisis']
agri_changes = [stress_results[s]['agri_ecl_change'] for s in stress_scenarios]
non_agri_changes = [stress_results[s]['non_agri_ecl_change'] for s in stress_scenarios]
x = np.arange(len(stress_scenarios))
width = 0.35
bars1 = ax.bar(x - width/2, agri_changes, width, label='Agricultural', color='#e74c3c', edgecolor='black')
bars2 = ax.bar(x + width/2, non_agri_changes, width, label='Non-Agricultural', color='#3498db', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(stress_scenarios, rotation=15, fontweight='bold')
ax.set_ylabel('ECL Change from Baseline (%)', fontweight='bold', fontsize=12)
ax.set_title('ECL Sensitivity: Agricultural vs Non-Agricultural', fontweight='bold', fontsize=14)
ax.tick_params(axis='both', labelsize=10)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_FIGURES, '23_agri_vulnerability.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 23_agri_vulnerability.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE - FINAL SUMMARY")
print("="*70)

print(f"""
PORTFOLIO OVERVIEW
------------------
Total Loans:           {len(df):,}
Total Exposure (EAD):  ${total_ead:,.0f}
Total ECL:             ${total_ecl:,.0f}
Portfolio ECL Rate:    {portfolio_ecl_rate:.4f}%

AGRICULTURAL SEGMENT
--------------------
Agricultural Loans:    {df['is_agri_portfolio'].sum():,}
Agricultural ECL:      ${agri_ecl:,.0f}
Agricultural ECL Rate: {agri_ecl_rate:.4f}%

MODEL PERFORMANCE
-----------------
Selected Model:        {SELECTED_MODEL_NAME}
Test AUC:              {SELECTED_AUC:.4f}
Test Gini:             {(2*SELECTED_AUC - 1):.4f}

STRESS TESTING
--------------
Severe Stress ECL Change:      {stress_results['Severe Stress']['total_ecl_change']:+.1f}%
Agricultural Crisis ECL Change: {stress_results['Agricultural Crisis']['total_ecl_change']:+.1f}%

OUTPUT FILES
------------
Data:   data_processed/loans_with_ecl.csv
Models: outputs/models/
Figures: outputs/figures/
""")

print("="*70)
print("ALL ANALYSIS COMPLETE!")
print("="*70)
