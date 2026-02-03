# Model Development Document (MDD)

## CECL Credit Risk Modeling for Agricultural Loan Portfolio

---

**Author:** Prabhu
**Status:** Final

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Business Context and Objectives](#2-business-context-and-objectives)
3. [Data Description](#3-data-description)
4. [Methodology](#4-methodology)
5. [Model Validation](#5-model-validation)
6. [Limitations and Assumptions](#6-limitations-and-assumptions)
7. [Model Monitoring and Governance](#7-model-monitoring-and-governance)
8. [Appendix](#8-appendix)

---

## 1. Executive Overview

### 1.1 Purpose

This document describes the development and validation of a CECL (Current Expected Credit Losses) credit risk model for estimating Expected Credit Losses (ECL) on a loan portfolio with a focus on agricultural-style lending segments.

### 1.2 Model Summary

| Component | Description |
|-----------|-------------|
| **Model Type** | Credit Risk - CECL ECL Estimation |
| **Target Variable** | Binary Default (0/1) |
| **Primary Use** | Allowance calculation, stress testing, risk management |
| **Portfolio Scope** | Consumer loans with agricultural segment proxy |
| **Key Outputs** | PD, LGD, EAD, ECL at loan and portfolio level |

### 1.3 Key Results

- **PD Model Performance:** AUC > 0.65 on out-of-time test set
- **Portfolio ECL Rate:** Calculated as PD x LGD x EAD
- **Agricultural Segment:** Higher risk profile with elevated default rates
- **Stress Testing:** ECL increases 25-75% under adverse scenarios

---

## 2. Business Context and Objectives

### 2.1 CECL Background

The Current Expected Credit Loss (CECL) standard (ASC 326) requires financial institutions to estimate and record lifetime expected credit losses at loan origination. This represents a shift from the previous incurred loss model to a forward-looking approach.

**Key CECL Requirements:**
- Estimate lifetime expected losses at origination
- Incorporate reasonable and supportable forecasts
- Consider historical loss experience
- Account for current economic conditions

### 2.2 Project Objectives

1. **Develop PD Model:** Build a robust probability of default model using machine learning techniques
2. **Estimate ECL Components:** Calculate PD, LGD, and EAD for each loan
3. **Agricultural Portfolio Analysis:** Create and analyze a proxy agricultural lending segment
4. **Stress Testing:** Measure portfolio vulnerability under adverse economic scenarios
5. **Documentation:** Produce regulatory-quality model documentation

### 2.3 Model Use Cases

- **CECL Allowance Calculation:** Primary allowance estimation
- **Portfolio Risk Assessment:** Segment-level risk analysis
- **Stress Testing:** Scenario analysis for capital planning
- **Risk Appetite Monitoring:** Early warning indicators
- **Strategic Planning:** Portfolio composition decisions

---

## 3. Data Description

### 3.1 Data Source

| Attribute | Description |
|-----------|-------------|
| **Source** | Zenodo - Lending Club Granting Model Dataset |
| **Original Provider** | Lending Club (P2P lending platform) |
| **Time Period** | 2007-2018 loan vintages |
| **Sample Size** | ~1.3 million loans |
| **Target Variable** | Binary default indicator |

### 3.2 Key Variables

#### 3.2.1 Numeric Features

| Variable | Description | Role |
|----------|-------------|------|
| `loan_amnt` | Loan amount ($) | EAD proxy |
| `revenue` | Borrower annual income ($) | Risk driver |
| `dti_n` | Debt-to-income ratio | Risk driver |
| `fico_n` | FICO credit score | Primary risk driver |

#### 3.2.2 Categorical Features

| Variable | Description | Role |
|----------|-------------|------|
| `purpose` | Loan purpose | Segmentation, risk driver |
| `home_ownership_n` | Home ownership status | Risk driver |
| `emp_length` | Employment length | Risk driver |
| `addr_state` | Borrower state | Geographic segmentation |

#### 3.2.3 Target Variable

| Variable | Description | Values |
|----------|-------------|--------|
| `Default` | Default indicator | 0 = Non-default, 1 = Default |

### 3.3 Data Quality

#### 3.3.1 Data Cleaning Steps

1. **Invalid Loan Amounts:** Removed loans with amount ≤ 0
2. **FICO Score Validation:** Filtered to valid range (300-850)
3. **Income Validation:** Removed negative/zero income, capped extreme values
4. **DTI Validation:** Removed negative DTI, capped at 100
5. **Missing Values:** Dropped rows with missing key features

#### 3.3.2 Data Quality Metrics

| Metric | Value |
|--------|-------|
| Records after cleaning | ~1.2 million |
| Missing value rate | < 1% (after cleaning) |
| Duplicate records | 0 |

### 3.4 Agricultural Portfolio Proxy

Since the dataset contains consumer P2P loans rather than true agricultural loans, we constructed a proxy agricultural segment:

**Selection Criteria:**
- **Purpose:** `small_business` loans (proxy for agricultural business lending)
- **Geography:** Top 10 agricultural states by USDA farm output (CA, IA, NE, TX, MN, IL, KS, WI, IN, NC)

**Rationale:**
- Small business loans approximate productive/commercial lending
- Agricultural state filter captures geographic risk factors
- Combined criteria create a reasonable agricultural lending proxy

---

## 4. Methodology

### 4.1 Model Architecture

The ECL estimation follows the standard credit risk formula:

$$ECL_i = PD_i \times LGD_i \times EAD_i$$

Where:
- $PD_i$ = Probability of Default for loan $i$
- $LGD_i$ = Loss Given Default for loan $i$
- $EAD_i$ = Exposure at Default for loan $i$

### 4.2 Probability of Default (PD) Model

#### 4.2.1 Model Selection

Two models were developed and compared:

| Model | Type | Key Parameters |
|-------|------|----------------|
| **Baseline** | Logistic Regression | class_weight='balanced' |
| **Advanced** | XGBoost/RandomForest | n_estimators=200, max_depth=6 |

#### 4.2.2 Feature Engineering

**Derived Features:**
- `loan_to_income`: Loan amount / Annual income
- `income_per_dti`: Income / DTI ratio
- `fico_bucket_num`: Numeric FICO bucket encoding
- `emp_length_num`: Numeric employment length

**Categorical Encoding:**
- One-hot encoding for `purpose` and `home_ownership`
- Top 10 purposes retained, others grouped as "other"

#### 4.2.3 Train/Test Split

- **Method:** Time-based split (out-of-time validation)
- **Training:** Earlier loan vintages (~70%)
- **Test:** Later loan vintages (~30%)
- **Rationale:** Simulates real-world model deployment

#### 4.2.4 Model Training

**Logistic Regression:**
```python
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs'
)
```

**XGBoost:**
```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=calculated_ratio
)
```

### 4.3 Loss Given Default (LGD)

#### 4.3.1 Methodology

Fixed LGD assumption based on regulatory guidance:

$$LGD = 45\%$$

#### 4.3.2 Rationale

- **Basel II Foundation IRB:** 45% LGD for senior unsecured claims
- **Industry Practice:** Consumer unsecured loans typically assume 40-60% LGD
- **Conservative Approach:** Suitable for CECL provisioning

#### 4.3.3 Implied Recovery

- Recovery Rate = 1 - LGD = 55%
- Represents expected recovery through collections/settlements

### 4.4 Exposure at Default (EAD)

#### 4.4.1 Methodology

Conservative EAD estimation using loan amount:

$$EAD = Loan\ Amount$$

#### 4.4.2 Rationale

- Term loans with no revolving component
- Conservative approach for loss estimation
- Appropriate for CECL lifetime loss perspective

### 4.5 Stress Testing

#### 4.5.1 Scenario Definitions

| Scenario | Income Change | DTI Change | Description |
|----------|--------------|------------|-------------|
| Baseline | 0% | 0% | Current conditions |
| Moderate Stress | -10% | +15% | Economic slowdown |
| Severe Stress | -20% | +30% | Recession |
| Agricultural Crisis | -25% | +35% | Sector-specific downturn |

#### 4.5.2 Methodology

1. Apply stress adjustments to income and DTI
2. Calculate PD multiplier based on stress severity
3. Recompute ECL with stressed PD
4. Compare against baseline

---

## 5. Model Validation

### 5.1 PD Model Performance

#### 5.1.1 Discrimination Metrics

| Metric | Logistic Regression | XGBoost/RF |
|--------|---------------------|------------|
| Training AUC | 0.67-0.70 | 0.72-0.78 |
| Test AUC | 0.65-0.68 | 0.68-0.73 |
| Gini Coefficient | 0.30-0.36 | 0.36-0.46 |

#### 5.1.2 Calibration

- Calibration curves show reasonable alignment between predicted and actual default rates
- Some deviation at extreme probability ranges (typical for imbalanced data)

#### 5.1.3 Feature Importance

**Top Risk Drivers:**
1. FICO Score (strongest negative relationship with default)
2. DTI (positive relationship with default)
3. Loan Purpose (especially small_business)
4. Income Level
5. Loan Amount

### 5.2 Model Selection Criteria

The advanced model (XGBoost/RandomForest) was selected based on:
- Higher test AUC
- Better discrimination across risk segments
- Appropriate calibration for ECL purposes

### 5.3 Back-Testing Considerations

- Time-based split provides out-of-time validation
- Model performs consistently across loan vintages
- Actual vs predicted default rates align within acceptable tolerance

---

## 6. Limitations and Assumptions

### 6.1 Data Limitations

1. **Proxy Portfolio:** Agricultural segment is approximated from consumer loan data
2. **Historical Data:** Model based on 2007-2018 data; economic conditions may differ
3. **Single Platform:** Data from one P2P lender may not generalize
4. **No Recovery Data:** LGD based on assumption rather than actual recoveries

### 6.2 Model Assumptions

1. **LGD Assumption:** Fixed 45% may not reflect actual loss severity
2. **EAD Assumption:** Full loan amount may overstate exposure for partially repaid loans
3. **Stress Scenarios:** Hypothetical scenarios based on historical recession patterns
4. **Independence:** Assumes loan defaults are conditionally independent

### 6.3 Model Risk

| Risk Category | Description | Mitigation |
|---------------|-------------|------------|
| **Data Quality** | Errors in source data | Data validation, cleaning |
| **Model Specification** | Incorrect functional form | Multiple model comparison |
| **Parameter Uncertainty** | Estimation error | Cross-validation, confidence intervals |
| **Regime Change** | Economic conditions differ from training | Stress testing, monitoring |

---

## 7. Model Monitoring and Governance

### 7.1 Performance Monitoring

#### 7.1.1 Key Metrics to Track

| Metric | Frequency | Threshold |
|--------|-----------|-----------|
| AUC-ROC | Monthly | > 0.60 |
| Population Stability Index (PSI) | Monthly | < 0.25 |
| Actual vs Predicted Default Rate | Quarterly | Within 20% |
| FICO Distribution Shift | Monthly | PSI < 0.10 |

#### 7.1.2 Early Warning Indicators

- Significant shift in feature distributions
- Degradation in discrimination metrics
- Systematic over/under prediction of defaults
- Changes in portfolio composition

### 7.2 Model Review Schedule

| Review Type | Frequency | Scope |
|-------------|-----------|-------|
| Performance Monitoring | Monthly | Metrics tracking |
| Calibration Review | Quarterly | Predicted vs actual |
| Full Model Review | Annual | Complete revalidation |
| Stress Testing | Annual | Scenario updates |

### 7.3 Change Management

**Triggers for Model Update:**
- AUC drops below 0.60
- PSI exceeds 0.25
- Significant regulatory changes
- Material portfolio changes
- Economic regime shifts

**Update Process:**
1. Document performance degradation
2. Investigate root cause
3. Propose model adjustments
4. Validate updated model
5. Obtain governance approval
6. Deploy and monitor

---

## 8. Appendix

### 8.1 Technical Specifications

**Software Environment:**
- Python 3.9+
- pandas, numpy for data manipulation
- scikit-learn for modeling
- XGBoost for advanced model
- matplotlib, seaborn for visualization

**Hardware Requirements:**
- Minimum 16GB RAM recommended for full dataset
- SSD storage for faster I/O

### 8.2 File Structure

```text
Credit_Risk_Personal_Project/
|-- data_raw/                         # Source data
|   |-- LC_loans_granting_model_dataset.csv
|-- data_processed/                   # Cleaned and transformed data
|   |-- loans_cleaned.csv
|   |-- loans_with_agri_flag.csv
|   |-- loans_with_pd.csv
|   |-- loans_with_lgd_ead.csv
|   |-- loans_with_ecl.csv
|-- notebooks/                        # Jupyter notebooks
|   |-- 01_Data_Acquisition_EDA.ipynb
|   |-- 02_Agricultural_Portfolio_Segmentation.ipynb
|   |-- 03_PD_Model_Development.ipynb
|   |-- 04_LGD_EAD_Estimation.ipynb
|   |-- 05_ECL_Computation.ipynb
|   |-- 06_Stress_Testing.ipynb
|   |-- 07_SHAP_Feature_Importance.ipynb
|-- outputs/
|   |-- figures/                      # Visualizations (PNG plots)
|   |-- models/                       # Saved models and metadata
|   |   |-- ecl_metadata.json
|   |   |-- feature_scaler.joblib
|   |   |-- lgd_ead_metadata.json
|   |   |-- model_metadata.json
|   |   |-- pd_model_logistic.joblib
|   |   |-- pd_model_selected.joblib
|   |   |-- pd_model_xgboost.joblib
|   |   |-- stress_test_results.json
|   |-- stress_test_summary.csv
|-- src/                              # Core pipeline modules
|   |-- __init__.py
|   |-- data_processing.py
|   |-- ecl_calculator.py
|   |-- feature_engineering.py
|   |-- modeling.py
|   |-- stress_testing.py
|   |-- visualization.py
|-- tests/                            # Unit tests
|   |-- __init__.py
|   |-- conftest.py
|   |-- test_data_processing.py
|   |-- test_ecl_calculator.py
|   |-- test_feature_engineering.py
|   |-- test_modeling.py
|   |-- test_stress_testing.py
|-- docs/                             # Documentation
|   |-- Executive_Summary.md
|   |-- Model_Card.md
|   |-- Model_Development_Document.md
|-- .gitignore
|-- pytest.ini
|-- README.md
|-- requirements.txt
|-- run_analysis.py
|-- .pytest_cache/                    # Auto-generated
|-- __pycache__/                      # Auto-generated
```

### 8.3 Glossary

| Term | Definition |
|------|------------|
| **AUC** | Area Under ROC Curve - discrimination metric |
| **CECL** | Current Expected Credit Losses |
| **DTI** | Debt-to-Income ratio |
| **EAD** | Exposure at Default |
| **ECL** | Expected Credit Loss |
| **FICO** | Fair Isaac Corporation credit score |
| **Gini** | Gini coefficient = 2 x AUC - 1 |
| **LGD** | Loss Given Default |
| **PD** | Probability of Default |
| **PSI** | Population Stability Index |

### 8.4 References

1. FASB ASC 326 - Financial Instruments - Credit Losses
2. Farm Credit Administration - Stress Testing Guidance
3. Basel Committee on Banking Supervision - IRB Approach
4. Lending Club Data Documentation (Zenodo)

---

*End of Model Development Document*
