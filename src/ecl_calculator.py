"""
ECL Calculator Module
=====================

Functions for computing Expected Credit Loss (ECL) under CECL framework.

ECL Formula: ECL = PD x LGD x EAD

Where:
- PD = Probability of Default
- LGD = Loss Given Default
- EAD = Exposure at Default

Author: Prabhu
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


# Default LGD assumption (Basel II Foundation IRB for unsecured)
DEFAULT_LGD = 0.45


def estimate_ead(
    df: pd.DataFrame,
    loan_amount_col: str = 'loan_amnt',
    output_col: str = 'ead_est'
) -> pd.DataFrame:
    """
    Estimate Exposure at Default (EAD).

    Uses loan amount as conservative EAD estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    loan_amount_col : str
        Name of the loan amount column.
    output_col : str
        Name of the output EAD column.

    Returns
    -------
    pd.DataFrame
        Dataframe with EAD estimates.
    """
    df = df.copy()
    df[output_col] = df[loan_amount_col]

    return df


def calculate_ead(
    df: pd.DataFrame,
    loan_amount_col: str = 'loan_amnt',
    ccf: float = 1.0
) -> pd.DataFrame:
    """
    Calculate Exposure at Default (EAD) with optional credit conversion factor.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    loan_amount_col : str
        Name of the loan amount column.
    ccf : float
        Credit conversion factor (default 1.0 = full exposure).

    Returns
    -------
    pd.DataFrame
        Dataframe with EAD column.
    """
    df = df.copy()
    df['ead'] = df[loan_amount_col] * ccf

    return df


def estimate_lgd(
    df: pd.DataFrame,
    lgd_value: float = DEFAULT_LGD,
    output_col: str = 'lgd_est'
) -> pd.DataFrame:
    """
    Estimate Loss Given Default (LGD).

    Uses fixed LGD assumption based on Basel II standards.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    lgd_value : float
        LGD assumption (default: 0.45 = 45%).
    output_col : str
        Name of the output LGD column.

    Returns
    -------
    pd.DataFrame
        Dataframe with LGD estimates.
    """
    df = df.copy()
    df[output_col] = lgd_value

    return df


def calculate_lgd(
    df: pd.DataFrame,
    fixed_lgd: Optional[float] = None,
    lgd_by_segment: Optional[Dict[str, float]] = None,
    segment_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate Loss Given Default (LGD).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fixed_lgd : float, optional
        Fixed LGD value for all loans.
    lgd_by_segment : Dict[str, float], optional
        LGD values by segment.
    segment_col : str, optional
        Column containing segment values.

    Returns
    -------
    pd.DataFrame
        Dataframe with LGD column.
    """
    df = df.copy()

    if fixed_lgd is not None:
        df['lgd'] = fixed_lgd
    elif lgd_by_segment is not None and segment_col is not None:
        df['lgd'] = df[segment_col].map(lgd_by_segment)
    else:
        df['lgd'] = DEFAULT_LGD

    return df


def calculate_ecl(
    df: pd.DataFrame,
    pd_col: str = 'pd_hat',
    lgd_col: str = 'lgd',
    ead_col: str = 'ead',
    output_col: str = 'ecl_est'
) -> pd.DataFrame:
    """
    Calculate Expected Credit Loss for each loan.

    ECL = PD x LGD x EAD

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with PD, LGD, and EAD columns.
    pd_col : str
        Name of the PD column.
    lgd_col : str
        Name of the LGD column.
    ead_col : str
        Name of the EAD column.
    output_col : str
        Name of the output ECL column.

    Returns
    -------
    pd.DataFrame
        Dataframe with ECL estimates.
    """
    df = df.copy()
    df[output_col] = df[pd_col] * df[lgd_col] * df[ead_col]

    return df


def calculate_ecl_rate(
    df: pd.DataFrame,
    ecl_col: str = 'ecl_est',
    ead_col: str = 'ead_est',
    output_col: str = 'ecl_rate'
) -> pd.DataFrame:
    """
    Calculate ECL rate (ECL as percentage of EAD).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ECL and EAD columns.
    ecl_col : str
        Name of the ECL column.
    ead_col : str
        Name of the EAD column.
    output_col : str
        Name of the output ECL rate column.

    Returns
    -------
    pd.DataFrame
        Dataframe with ECL rate.
    """
    df = df.copy()
    df[output_col] = df[ecl_col] / df[ead_col]

    return df


def get_portfolio_ecl_summary(
    df: pd.DataFrame,
    ecl_col: str = 'ecl_est',
    ead_col: str = 'ead_est',
    pd_col: str = 'pd_hat',
    default_col: str = 'Default'
) -> Dict:
    """
    Calculate portfolio-level ECL summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ECL components.
    ecl_col : str
        Name of the ECL column.
    ead_col : str
        Name of the EAD column.
    pd_col : str
        Name of the PD column.
    default_col : str
        Name of the default indicator column.

    Returns
    -------
    Dict
        Portfolio ECL summary statistics.
    """
    summary = {
        'total_loans': len(df),
        'total_ead': df[ead_col].sum(),
        'total_ecl': df[ecl_col].sum(),
        'ecl_rate': df[ecl_col].sum() / df[ead_col].sum() * 100,
        'mean_pd': df[pd_col].mean(),
        'median_pd': df[pd_col].median(),
        'mean_ecl': df[ecl_col].mean(),
        'median_ecl': df[ecl_col].median(),
        'actual_default_rate': df[default_col].mean() * 100 if default_col in df.columns else None
    }

    return summary


def get_segment_ecl_summary(
    df: pd.DataFrame,
    segment_col: str,
    ecl_col: str = 'ecl_est',
    ead_col: str = 'ead_est',
    pd_col: str = 'pd_hat',
    default_col: str = 'Default'
) -> pd.DataFrame:
    """
    Calculate ECL summary by segment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ECL components.
    segment_col : str
        Name of the segment column.
    ecl_col : str
        Name of the ECL column.
    ead_col : str
        Name of the EAD column.
    pd_col : str
        Name of the PD column.
    default_col : str
        Name of the default indicator column.

    Returns
    -------
    pd.DataFrame
        ECL summary by segment.
    """
    summary = df.groupby(segment_col).agg(
        loan_count=('id', 'count') if 'id' in df.columns else (ecl_col, 'count'),
        total_ead=(ead_col, 'sum'),
        total_ecl=(ecl_col, 'sum'),
        mean_pd=(pd_col, 'mean'),
        actual_default_rate=(default_col, 'mean')
    )

    summary['ecl_rate'] = summary['total_ecl'] / summary['total_ead'] * 100
    summary['ead_pct'] = summary['total_ead'] / summary['total_ead'].sum() * 100
    summary['ecl_pct'] = summary['total_ecl'] / summary['total_ecl'].sum() * 100

    return summary


def calculate_ecl_by_bucket(
    df: pd.DataFrame,
    bucket_col: str,
    ecl_col: str = 'ecl_est',
    ead_col: str = 'ead_est'
) -> pd.DataFrame:
    """
    Calculate ECL aggregated by risk bucket.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    bucket_col : str
        Name of the bucket column (e.g., 'fico_bucket').
    ecl_col : str
        Name of the ECL column.
    ead_col : str
        Name of the EAD column.

    Returns
    -------
    pd.DataFrame
        ECL summary by bucket.
    """
    summary = df.groupby(bucket_col, observed=True).agg(
        loan_count=(ecl_col, 'count'),
        total_ead=(ead_col, 'sum'),
        total_ecl=(ecl_col, 'sum')
    )

    summary['ecl_rate'] = summary['total_ecl'] / summary['total_ead'] * 100
    summary['ecl_pct'] = summary['total_ecl'] / summary['total_ecl'].sum() * 100

    return summary


def create_pd_bands(
    df: pd.DataFrame,
    pd_col: str = 'pd_hat',
    output_col: str = 'pd_band'
) -> pd.DataFrame:
    """
    Create PD risk bands for portfolio segmentation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    pd_col : str
        Name of the PD column.
    output_col : str
        Name of the output band column.

    Returns
    -------
    pd.DataFrame
        Dataframe with PD band column.
    """
    df = df.copy()

    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
    labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>30%']

    df[output_col] = pd.cut(df[pd_col], bins=bins, labels=labels)

    return df


def aggregate_ecl_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    ecl_col: str = 'ecl_est'
) -> pd.Series:
    """
    Aggregate ECL by segment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    segment_col : str
        Column containing segment values.
    ecl_col : str
        Column containing ECL values.

    Returns
    -------
    pd.Series
        ECL totals by segment.
    """
    return df.groupby(segment_col)[ecl_col].sum()


def calculate_portfolio_ecl_rate(
    df: pd.DataFrame,
    ecl_col: str = 'ecl_est',
    ead_col: str = 'ead'
) -> float:
    """
    Calculate portfolio-level ECL rate.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    ecl_col : str
        Column containing ECL values.
    ead_col : str
        Column containing EAD values.

    Returns
    -------
    float
        ECL rate (ECL / EAD).
    """
    total_ecl = df[ecl_col].sum()
    total_ead = df[ead_col].sum()

    return total_ecl / total_ead if total_ead > 0 else 0.0


def compute_full_ecl(
    df: pd.DataFrame,
    pd_col: str = 'pd_hat',
    lgd_value: float = DEFAULT_LGD
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute full ECL with all components.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with PD predictions and loan amounts.
    pd_col : str
        Name of the PD column.
    lgd_value : float
        LGD assumption.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Dataframe with ECL and portfolio summary.
    """
    # Estimate components
    df = estimate_ead(df)
    df = estimate_lgd(df, lgd_value)
    df = calculate_ecl(df, pd_col=pd_col, lgd_col='lgd_est', ead_col='ead_est')
    df = calculate_ecl_rate(df)
    df = create_pd_bands(df, pd_col=pd_col)

    # Get summary
    summary = get_portfolio_ecl_summary(df)

    return df, summary
