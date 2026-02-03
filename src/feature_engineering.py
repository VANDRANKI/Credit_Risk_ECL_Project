"""
Feature Engineering Module
==========================

Functions for creating and transforming features for credit risk modeling.

Author: Prabhu
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


# Agricultural states (Top 10 by USDA farm output)
AGRI_STATES = ['CA', 'IA', 'NE', 'TX', 'MN', 'IL', 'KS', 'WI', 'IN', 'NC']

# Agricultural loan purposes
AGRI_PURPOSES = ['small_business']

# Employment length mapping
EMP_LENGTH_MAP = {
    '< 1 year': 0.5,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
}

# FICO bucket mapping
FICO_BUCKET_MAP = {
    '300-579': 1,
    '580-619': 2,
    '620-659': 3,
    '660-699': 4,
    '700-739': 5,
    '740-779': 6,
    '780-850': 7
}


def create_agricultural_flag(
    df: pd.DataFrame,
    agri_states: List[str] = AGRI_STATES,
    purpose_col: str = 'purpose',
    state_col: str = 'addr_state',
    agri_purposes: List[str] = AGRI_PURPOSES
) -> pd.DataFrame:
    """
    Create agricultural portfolio flag based on purpose and state.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    purpose_col : str
        Name of the loan purpose column.
    state_col : str
        Name of the state column.
    agri_purposes : List[str]
        List of purposes considered agricultural.
    agri_states : List[str]
        List of states considered agricultural.

    Returns
    -------
    pd.DataFrame
        Dataframe with agricultural portfolio flags added.
    """
    df = df.copy()

    # Main agricultural flag (both criteria)
    df['is_agri_portfolio'] = (
        (df[purpose_col].isin(agri_purposes)) &
        (df[state_col].isin(agri_states))
    ).astype(int)

    # Individual flags for analysis
    df['is_agri_state'] = df[state_col].isin(agri_states).astype(int)
    df['is_small_business'] = df[purpose_col].isin(agri_purposes).astype(int)

    return df


def create_loan_to_income_ratio(
    df: pd.DataFrame,
    loan_col: str = 'loan_amnt',
    income_col: str = 'revenue',
    output_col: str = 'loan_to_income'
) -> pd.DataFrame:
    """
    Create loan-to-income ratio feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    loan_col : str
        Name of the loan amount column.
    income_col : str
        Name of the income column.
    output_col : str
        Name of the output column.

    Returns
    -------
    pd.DataFrame
        Dataframe with loan-to-income ratio added.
    """
    df = df.copy()
    df[output_col] = df[loan_col] / (df[income_col] + 1)

    # Handle infinities
    df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan)
    df[output_col] = df[output_col].fillna(df[output_col].median())

    return df


def create_income_per_dti(
    df: pd.DataFrame,
    income_col: str = 'revenue',
    dti_col: str = 'dti_n',
    output_col: str = 'income_per_dti'
) -> pd.DataFrame:
    """
    Create income-per-DTI feature (higher is better creditworthiness).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    income_col : str
        Name of the income column.
    dti_col : str
        Name of the DTI column.
    output_col : str
        Name of the output column.

    Returns
    -------
    pd.DataFrame
        Dataframe with income-per-DTI ratio added.
    """
    df = df.copy()
    df[output_col] = df[income_col] / (df[dti_col] + 1)

    # Handle infinities
    df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan)
    df[output_col] = df[output_col].fillna(df[output_col].median())

    return df


def encode_employment_length(
    df: pd.DataFrame,
    input_col: str = 'emp_length',
    output_col: str = 'emp_length_num',
    mapping: Dict[str, float] = EMP_LENGTH_MAP
) -> pd.DataFrame:
    """
    Encode employment length as numeric feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    input_col : str
        Name of the employment length column.
    output_col : str
        Name of the output column.
    mapping : Dict[str, float]
        Mapping from string to numeric values.

    Returns
    -------
    pd.DataFrame
        Dataframe with numeric employment length.
    """
    df = df.copy()
    df[output_col] = df[input_col].map(mapping)
    df[output_col] = df[output_col].fillna(df[output_col].median())

    return df


def encode_fico_bucket(
    df: pd.DataFrame,
    input_col: str = 'fico_bucket',
    output_col: str = 'fico_bucket_num',
    mapping: Dict[str, int] = FICO_BUCKET_MAP
) -> pd.DataFrame:
    """
    Encode FICO bucket as numeric feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    input_col : str
        Name of the FICO bucket column.
    output_col : str
        Name of the output column.
    mapping : Dict[str, int]
        Mapping from bucket string to numeric values.

    Returns
    -------
    pd.DataFrame
        Dataframe with numeric FICO bucket.
    """
    df = df.copy()
    df[output_col] = df[input_col].astype(str).map(mapping)
    df[output_col] = pd.to_numeric(df[output_col], errors='coerce')
    df[output_col] = df[output_col].fillna(4)  # Default to middle bucket

    return df


def create_purpose_dummies(
    df: pd.DataFrame,
    input_col: str = 'purpose',
    top_n: int = 10,
    prefix: str = 'purpose'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create one-hot encoded purpose features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    input_col : str
        Name of the purpose column.
    top_n : int
        Number of top purposes to keep (rest grouped as 'other').
    prefix : str
        Prefix for dummy column names.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Dataframe with dummy columns and list of dummy column names.
    """
    df = df.copy()

    # Get top purposes
    top_purposes = df[input_col].value_counts().head(top_n).index.tolist()

    # Group rare purposes as 'other'
    df['purpose_grouped'] = df[input_col].apply(
        lambda x: x if x in top_purposes else 'other'
    )

    # Create dummies
    dummies = pd.get_dummies(df['purpose_grouped'], prefix=prefix)
    dummy_columns = list(dummies.columns)

    df = pd.concat([df, dummies], axis=1)

    return df, dummy_columns


def create_home_ownership_dummies(
    df: pd.DataFrame,
    input_col: str = 'home_ownership_n',
    prefix: str = 'home'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create one-hot encoded home ownership features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    input_col : str
        Name of the home ownership column.
    prefix : str
        Prefix for dummy column names.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Dataframe with dummy columns and list of dummy column names.
    """
    df = df.copy()

    dummies = pd.get_dummies(df[input_col], prefix=prefix)
    dummy_columns = list(dummies.columns)

    df = pd.concat([df, dummies], axis=1)

    return df, dummy_columns


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features (loan_to_income, income_per_dti).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with loan_amnt, revenue, dti_n columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with derived features added.
    """
    df = df.copy()

    # Loan to income ratio
    df['loan_to_income'] = df['loan_amnt'] / (df['revenue'] + 1)
    df['loan_to_income'] = df['loan_to_income'].replace([np.inf, -np.inf], np.nan)
    df['loan_to_income'] = df['loan_to_income'].fillna(df['loan_to_income'].median())

    # Income per DTI
    df['income_per_dti'] = df['revenue'] / (df['dti_n'] + 1)
    df['income_per_dti'] = df['income_per_dti'].replace([np.inf, -np.inf], np.nan)
    df['income_per_dti'] = df['income_per_dti'].fillna(df['income_per_dti'].median())

    return df


def encode_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias for encode_employment_length for API compatibility.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with emp_length column.

    Returns
    -------
    pd.DataFrame
        Dataframe with emp_length_num column.
    """
    return encode_employment_length(df)


def create_feature_matrix(
    df: pd.DataFrame,
    features: List[str],
    target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix and target vector for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : List[str]
        List of feature column names.
    target : str
        Name of target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    df_clean = df.dropna(subset=features + [target]).copy()

    X = df_clean[features].copy()
    y = df_clean[target].copy()

    return X, y


def engineer_all_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Apply all feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with cleaned data.

    Returns
    -------
    Tuple[pd.DataFrame, List[str], List[str]]
        - Dataframe with all engineered features
        - List of numeric feature names
        - List of dummy feature names
    """
    # Create derived features
    df = create_loan_to_income_ratio(df)
    df = create_income_per_dti(df)
    df = encode_employment_length(df)
    df = encode_fico_bucket(df)

    # Create agricultural flag
    df = create_agricultural_flag(df)

    # Create dummy features
    df, purpose_dummies = create_purpose_dummies(df)
    df, home_dummies = create_home_ownership_dummies(df)

    # Define feature lists
    numeric_features = [
        'loan_amnt', 'revenue', 'dti_n', 'fico_n',
        'loan_to_income', 'income_per_dti', 'fico_bucket_num', 'emp_length_num'
    ]
    dummy_features = purpose_dummies + home_dummies

    return df, numeric_features, dummy_features
