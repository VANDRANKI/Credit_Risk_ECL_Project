# Model Card: CECL PD Model

**Author:** Prabhu

---

## Model Overview

| Attribute | Value |
|-----------|-------|
| **Model Name** | CECL Probability of Default (PD) Model |
| **Model Type** | Binary Classification |
| **Algorithm** | XGBoost Gradient Boosting |
| **Target Variable** | Loan Default (0/1) |
| **Primary Use** | CECL Expected Credit Loss Estimation |

---

## Intended Use

### Primary Use Cases
- **CECL Compliance**: Estimate lifetime expected credit losses for consumer loan portfolios
- **Risk Assessment**: Evaluate default probability for loan origination decisions
- **Portfolio Monitoring**: Track credit quality trends across segments
- **Stress Testing**: Apply macroeconomic scenarios to PD estimates

### Intended Users
- Credit Risk Analysts
- Model Validation Teams
- Portfolio Managers
- Regulatory Compliance Officers

### Out-of-Scope Uses
- Real-time automated lending decisions without human oversight
- High-frequency trading applications
- Non-consumer loan portfolios (commercial, corporate)

---

## Training Data

### Data Source
- **Dataset**: LendingClub Loan Data (2007-2018)
- **Records**: ~1.3 million loans after cleaning
- **Time Period**: 2007 to Q4 2018

### Data Characteristics

| Feature Category | Variables |
|------------------|-----------|
| **Borrower Profile** | Annual Income, Employment Length, Home Ownership |
| **Credit History** | FICO Score, DTI Ratio |
| **Loan Terms** | Loan Amount, Purpose, Interest Rate |
| **Geography** | State (for agricultural segment proxy) |

### Target Definition
Default is defined as any loan reaching one of these statuses:
- Charged Off
- Default
- Late (31-120 days)
- Late (16-30 days)

### Data Split
| Set | Proportion | Purpose |
|-----|------------|---------|
| Training | 70% | Model fitting |
| Test | 30% | Performance evaluation |

*Split Method: Time-based (earlier loans for training)*

---

## Model Architecture

### Algorithm Selection
**XGBoost Classifier** was selected over Logistic Regression based on:
- Higher AUC-ROC on test set
- Better handling of non-linear relationships
- Robust to feature interactions

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Balance of performance and overfitting |
| `max_depth` | 6 | Prevents over-complex trees |
| `learning_rate` | 0.1 | Standard learning rate |
| `subsample` | 0.8 | Reduces overfitting |
| `colsample_bytree` | 0.8 | Feature subsampling |
| `scale_pos_weight` | Auto | Handles class imbalance |

### Features Used

| Feature | Description | Type |
|---------|-------------|------|
| `fico_n` | FICO credit score | Numeric |
| `dti_n` | Debt-to-income ratio (%) | Numeric |
| `loan_amnt` | Loan amount ($) | Numeric |
| `revenue` | Annual income ($) | Numeric |
| `loan_to_income` | Loan amount / Income ratio | Derived |
| `income_per_dti` | Income / DTI ratio | Derived |
| `fico_bucket_num` | FICO score bucket (1-7) | Encoded |
| `emp_length_num` | Employment length (years) | Encoded |

---

## Performance Metrics

### Overall Model Performance

| Metric | Training | Test |
|--------|----------|------|
| **AUC-ROC** | ~0.70 | ~0.67 |
| **Gini** | ~0.40 | ~0.34 |
| **Brier Score** | ~0.12 | ~0.13 |

### Performance by FICO Segment

| FICO Bucket | Default Rate | Model Avg PD | Alignment |
|-------------|--------------|--------------|-----------|
| 300-579 | ~50% | High | Good |
| 580-619 | ~40% | Medium-High | Good |
| 620-659 | ~30% | Medium | Good |
| 660-699 | ~22% | Medium | Good |
| 700-739 | ~15% | Low-Medium | Good |
| 740-779 | ~10% | Low | Good |
| 780-850 | ~5% | Very Low | Good |

### Discrimination Analysis
- Model ranks high-risk borrowers above low-risk borrowers effectively
- FICO score is the primary driver of predictions
- DTI ratio provides secondary discrimination power

---

## Limitations

### Known Limitations

1. **Data Vintage**: Training data from 2007-2018 may not reflect current economic conditions
2. **Geographic Proxy**: Agricultural portfolio identified by state and purpose, not actual farm loans
3. **Point-in-Time**: Model predicts lifetime PD, assumes stable economic conditions
4. **Feature Availability**: Requires complete data for all 8 features

### Performance Limitations

| Scenario | Limitation |
|----------|------------|
| **Extreme FICO** | Limited calibration data for FICO < 550 |
| **High Income** | Model may underpredict for income > $500K |
| **New Products** | Not validated for non-personal loan types |

### Assumptions

- LGD is fixed at 45% (Basel II Foundation IRB standard)
- EAD equals full loan amount (conservative assumption)
- Default is binary, no partial recovery modeling

---

## Ethical Considerations

### Fairness Assessment

The model uses FICO scores, which may contain embedded historical biases. Considerations:

1. **Protected Classes**: Model does not use race, gender, age, or religion
2. **Proxy Variables**: State and income may correlate with protected characteristics
3. **Disparate Impact**: Regular monitoring recommended for approval rate disparities

### Mitigation Strategies

| Risk | Mitigation |
|------|------------|
| Proxy discrimination | Exclude zip code, use state only for agricultural proxy |
| Economic bias | Regular recalibration with recent data |
| Model opacity | SHAP explanations for every prediction |

### Transparency

- All predictions can be explained via SHAP values
- Feature contributions are documented for regulatory review
- Model documentation follows SR 11-7 requirements

---

## Model Governance

### Validation Requirements

| Validation Type | Frequency | Owner |
|-----------------|-----------|-------|
| Conceptual Soundness | Initial + Annual | Model Validation |
| Ongoing Monitoring | Monthly | Model Risk |
| Outcome Analysis | Quarterly | Credit Risk |
| Sensitivity Testing | Semi-Annual | Model Validation |

### Monitoring Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| AUC Degradation | > 5% decline | Recalibration trigger |
| PSI (Population Stability) | > 0.25 | Data drift investigation |
| Default Rate Alignment | > 20% variance | Model review |

### Version Control

| Version | Changes | Effective |
|---------|---------|-----------|
| 1.0 | Initial deployment | Current |

---

## ECL Application

### ECL Formula

```
ECL = PD × LGD × EAD
```

Where:
- **PD**: Model-predicted probability of default
- **LGD**: 45% fixed (Basel II Foundation IRB)
- **EAD**: Full loan amount

### Stress Testing

Model supports stress scenarios:

| Scenario | PD Multiplier | Additional LGD |
|----------|---------------|----------------|
| Baseline | 1.0x | +0% |
| Moderate | 1.25x | +5% |
| Severe | 1.50x | +10% |
| Agricultural Crisis | 1.75x (Ag only) | +10% |

---

## Reproducibility

### Technical Requirements

```
Python >= 3.8
pandas >= 1.5.0
numpy >= 1.23.0
scikit-learn >= 1.2.0
xgboost >= 1.7.0
shap >= 0.42.0
```

### Model Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Trained Model | `outputs/models/` | `.joblib` |
| Feature List | `src/feature_engineering.py` | Python |
| Training Script | `run_analysis.py` | Python |

---

**Contact:** Prabhu | Data Analyst | Data Specialist | Quantitative Risk Analyst

---
