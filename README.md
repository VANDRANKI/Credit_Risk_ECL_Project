# CECL Credit Risk Modeling and Stress Testing for Agricultural Loan Portfolio

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## Overview

This project implements an end-to-end **CECL (Current Expected Credit Losses)** credit risk modeling framework with stress testing capabilities, focusing on an agricultural-style loan portfolio segment. The project demonstrates quantitative risk analysis skills including PD/LGD/EAD modeling, ECL computation, and scenario-based stress testing.

### Key Features

- **PD Modeling:** Probability of Default models using Logistic Regression and XGBoost
- **ECL Framework:** Complete PD x LGD x EAD expected loss calculation
- **Agricultural Portfolio:** Proxy agricultural segment analysis with regional focus
- **Stress Testing:** Multiple macroeconomic stress scenarios with portfolio-level impact analysis
- **Professional Documentation:** Model Development Document (MDD) and Executive Summary

## Project Structure

```
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

## Dataset

- **Source:** Zenodo - Lending Club Granting Model Dataset
- **Size:** ~1.3 million loans
- **Time Period:** 2007-2018 loan vintages
- **Target Variable:** Binary default indicator (0/1)

### Key Variables

| Variable | Description |
|----------|-------------|
| `loan_amnt` | Loan amount ($) |
| `revenue` | Borrower annual income ($) |
| `dti_n` | Debt-to-income ratio |
| `fico_n` | FICO credit score |
| `purpose` | Loan purpose |
| `addr_state` | Borrower state |
| `Default` | Default indicator (0/1) |

## Methodology

### 1. Probability of Default (PD) Modeling

- **Baseline Model:** Logistic Regression with class balancing
- **Advanced Model:** XGBoost with hyperparameter tuning
- **Validation:** Time-based train/test split (out-of-time validation)
- **Performance:** AUC > 0.65 on test set

### 2. Loss Given Default (LGD)

- **Approach:** Fixed 45% LGD assumption
- **Rationale:** Basel II Foundation IRB standard for unsecured exposures

### 3. Exposure at Default (EAD)

- **Approach:** Loan amount as EAD (conservative)
- **Rationale:** Full exposure assumption for term loans

### 4. Expected Credit Loss (ECL)

```
ECL = PD x LGD x EAD
```

Portfolio-level ECL aggregated by segment for CECL provisioning.

### 5. Stress Testing

| Scenario | Income Change | DTI Change |
|----------|--------------|------------|
| Baseline | 0% | 0% |
| Moderate Stress | -10% | +15% |
| Severe Stress | -20% | +30% |
| Agricultural Crisis | -25% | +35% |

## Key Results

### Portfolio Risk Profile

| Segment | Default Rate | ECL Rate |
|---------|-------------|----------|
| Agricultural | ~26% | ~8.0% |
| Non-Agricultural | ~20% | ~7.0% |
| Total Portfolio | ~20% | ~7.2% |

### Stress Testing Impact

| Scenario | ECL Change |
|----------|------------|
| Moderate Stress | +26% |
| Severe Stress | +56% |
| Agricultural Crisis | +76% |

### Key Insights

1. Agricultural segment shows 1.3x higher default rate than non-agricultural
2. FICO score is the strongest predictor of default
3. Agricultural portfolio more sensitive to economic stress (1.3-1.5x higher ECL increase)
4. Small business loans in agricultural states represent highest risk concentration

## Installation

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cecl-agri-credit-risk.git
cd cecl-agri-credit-risk
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from [Zenodo](https://zenodo.org/records/11295916) and place in `data_raw/`

5. Run notebooks in order (01 through 06)

## Usage

### Running the Analysis

Execute notebooks sequentially:

```bash
jupyter notebook notebooks/01_Data_Acquisition_EDA.ipynb
```

Each notebook produces:
- Processed data files saved to `data_processed/`
- Visualizations saved to `outputs/figures/`
- Models saved to `outputs/models/`

### Viewing Results

- **Figures:** Check `outputs/figures/` for all visualizations
- **Documentation:** Read `docs/Executive_Summary.md` for key findings
- **Technical Details:** See `docs/Model_Development_Document.md`

## Documentation

| Document | Description |
|----------|-------------|
| [Model Development Document](docs/Model_Development_Document.md) | Technical methodology, validation, and governance |
| [Executive Summary](docs/Executive_Summary.md) | High-level findings for leadership |
| [Project Plan](PROJECT_PLAN_CECL_AGRI_PORTFOLIO.md) | Original project requirements |

## Skills Demonstrated

- **Credit Risk Modeling:** PD, LGD, EAD, ECL estimation
- **Machine Learning:** Logistic Regression, XGBoost, feature engineering
- **CECL Framework:** Lifetime expected loss modeling
- **Stress Testing:** Scenario analysis and sensitivity testing
- **Data Analysis:** EDA, data cleaning, visualization
- **Documentation:** Model governance and executive reporting
- **Python:** pandas, scikit-learn, XGBoost, matplotlib

## Regulatory Alignment

This project aligns with:
- **CECL (ASC 326):** Forward-looking expected loss estimation
- **Farm Credit Administration:** Agricultural lending stress testing guidance
- **Basel II/III:** IRB approach for credit risk

## Author

**Prabhu**
- Quantitative Risk Analyst
- Credit Risk Modeling | CECL | Stress Testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Lending Club for the original dataset
- Zenodo for data hosting
- Farm Credit Administration for stress testing guidance

---

*This project was developed as a demonstration of credit risk modeling capabilities for quantitative risk analyst roles.*
