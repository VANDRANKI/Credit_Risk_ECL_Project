"""
Data Processing Module
======================

Functions for loading, cleaning, and validating loan data.

Author: Prabhu
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict


def load_loan_data(filepath: str) -> pd.DataFrame:
    """
    Load loan data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing loan data.

    Returns
    -------
    pd.DataFrame
        Raw loan data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    df = pd.read_csv(filepath)
    return df


def validate_loan_amounts(df: pd.DataFrame, column: str = 'loan_amnt') -> pd.DataFrame:
    """
    Remove rows with invalid loan amounts (<=0).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the loan amount column.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with valid loan amounts.
    """
    initial_count = len(df)
    df_clean = df[df[column] > 0].copy()
    removed = initial_count - len(df_clean)

    if removed > 0:
        print(f"Removed {removed:,} rows with invalid loan amounts")

    return df_clean


def validate_fico_scores(
    df: pd.DataFrame,
    column: str = 'fico_n',
    min_score: int = 300,
    max_score: int = 850
) -> pd.DataFrame:
    """
    Filter FICO scores to valid range.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the FICO score column.
    min_score : int
        Minimum valid FICO score (default: 300).
    max_score : int
        Maximum valid FICO score (default: 850).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with valid FICO scores.
    """
    initial_count = len(df)
    df_clean = df[(df[column] >= min_score) & (df[column] <= max_score)].copy()
    removed = initial_count - len(df_clean)

    if removed > 0:
        print(f"Removed {removed:,} rows with invalid FICO scores")

    return df_clean


def validate_income(
    df: pd.DataFrame,
    column: str = 'revenue',
    min_income: float = 0,
    max_income: float = 10_000_000
) -> pd.DataFrame:
    """
    Filter income values to valid range.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the income column.
    min_income : float
        Minimum valid income (exclusive).
    max_income : float
        Maximum valid income (inclusive).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with valid income values.
    """
    initial_count = len(df)
    df_clean = df[(df[column] > min_income) & (df[column] <= max_income)].copy()
    removed = initial_count - len(df_clean)

    if removed > 0:
        print(f"Removed {removed:,} rows with invalid income values")

    return df_clean


def validate_dti(
    df: pd.DataFrame,
    column: str = 'dti_n',
    max_dti: float = 100
) -> pd.DataFrame:
    """
    Validate and cap DTI (Debt-to-Income) ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the DTI column.
    max_dti : float
        Maximum DTI value to cap at.

    Returns
    -------
    pd.DataFrame
        Dataframe with validated DTI values.
    """
    df_clean = df[df[column] >= 0].copy()
    df_clean[column] = df_clean[column].clip(upper=max_dti)

    return df_clean


def clean_loan_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply all data cleaning steps to loan data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loan data.
    required_columns : List[str], optional
        Columns required for modeling. Rows with missing values
        in these columns will be dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned loan data.
    """
    print("Starting data cleaning...")
    print(f"Initial shape: {df.shape}")

    # Apply validation steps
    df_clean = validate_loan_amounts(df)
    df_clean = validate_fico_scores(df_clean)
    df_clean = validate_income(df_clean)
    df_clean = validate_dti(df_clean)

    # Drop rows with missing required columns
    if required_columns:
        df_clean = df_clean.dropna(subset=required_columns)
        print(f"Dropped rows with missing values in required columns")

    print(f"Final shape: {df_clean.shape}")
    print(f"Rows removed: {len(df) - len(df_clean):,} ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")

    return df_clean


def create_fico_buckets(
    df: pd.DataFrame,
    column: str = 'fico_n',
    output_column: str = 'fico_bucket'
) -> pd.DataFrame:
    """
    Create FICO score buckets for risk segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the FICO score column.
    output_column : str
        Name of the output bucket column.

    Returns
    -------
    pd.DataFrame
        Dataframe with FICO bucket column added.
    """
    bins = [300, 579, 619, 659, 699, 739, 779, 850]
    labels = ['300-579', '580-619', '620-659', '660-699',
              '700-739', '740-779', '780-850']

    df = df.copy()
    df[output_column] = pd.cut(df[column], bins=bins, labels=labels)

    return df


def create_income_buckets(
    df: pd.DataFrame,
    column: str = 'revenue',
    output_column: str = 'income_bucket'
) -> pd.DataFrame:
    """
    Create income buckets for risk segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the income column.
    output_column : str
        Name of the output bucket column.

    Returns
    -------
    pd.DataFrame
        Dataframe with income bucket column added.
    """
    bins = [0, 40000, 60000, 80000, 100000, 150000, float('inf')]
    labels = ['<$40K', '$40K-$60K', '$60K-$80K',
              '$80K-$100K', '$100K-$150K', '>$150K']

    df = df.copy()
    df[output_column] = pd.cut(df[column], bins=bins, labels=labels)

    return df


def parse_issue_date(
    df: pd.DataFrame,
    column: str = 'issue_d',
    date_format: str = '%b-%Y'
) -> pd.DataFrame:
    """
    Parse issue date and extract year/month components.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the date column.
    date_format : str
        Format of the date string.

    Returns
    -------
    pd.DataFrame
        Dataframe with parsed date columns.
    """
    df = df.copy()
    df['issue_date'] = pd.to_datetime(df[column], format=date_format)
    df['issue_year'] = df['issue_date'].dt.year
    df['issue_month'] = df['issue_date'].dt.month

    return df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for loan data.

    Parameters
    ----------
    df : pd.DataFrame
        Loan dataframe.

    Returns
    -------
    Dict
        Dictionary containing summary statistics.
    """
    summary = {
        'total_loans': len(df),
        'total_exposure': df['loan_amnt'].sum() if 'loan_amnt' in df.columns else None,
        'default_rate': df['Default'].mean() if 'Default' in df.columns else None,
        'avg_fico': df['fico_n'].mean() if 'fico_n' in df.columns else None,
        'avg_dti': df['dti_n'].mean() if 'dti_n' in df.columns else None,
        'avg_income': df['revenue'].mean() if 'revenue' in df.columns else None,
        'avg_loan_amount': df['loan_amnt'].mean() if 'loan_amnt' in df.columns else None,
    }

    return summary


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str]
) -> bool:
    """
    Validate that a dataframe has required columns and is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    required_columns : List[str]
        List of required column names.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If dataframe is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return True


def clean_numeric_column(
    df: pd.DataFrame,
    column: str,
    clip_percentile: Optional[float] = None
) -> pd.DataFrame:
    """
    Clean a numeric column by replacing infinities and optionally clipping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the column to clean.
    clip_percentile : float, optional
        Percentile for clipping extreme values.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column.
    """
    df = df.copy()

    # Replace infinities with NaN
    df[column] = df[column].replace([np.inf, -np.inf], np.nan)

    # Clip extreme values if requested
    if clip_percentile is not None:
        upper = df[column].quantile(clip_percentile / 100)
        df[column] = df[column].clip(upper=upper)

    return df


def handle_missing_values(
    df: pd.DataFrame,
    column: str,
    strategy: str = 'median',
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Name of the column.
    strategy : str
        Strategy for filling: 'median', 'mean', or 'constant'.
    fill_value : float, optional
        Value to use if strategy is 'constant'.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled.
    """
    df = df.copy()

    if strategy == 'median':
        df[column] = df[column].fillna(df[column].median())
    elif strategy == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == 'constant':
        df[column] = df[column].fillna(fill_value)

    return df


def create_time_split(
    df: pd.DataFrame,
    date_col: str,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create time-based train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column.
    test_ratio : float
        Proportion of data for test set.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test dataframes.
    """
    df = df.sort_values(date_col).copy()

    split_idx = int(len(df) * (1 - test_ratio))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test
