"""
Stress Testing Module
=====================

Functions for applying macroeconomic stress scenarios to credit portfolios.

Author: Prabhu
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# Default stress scenarios
DEFAULT_SCENARIOS = {
    'Baseline': {
        'income_change': 0.00,
        'dti_change': 0.00,
        'description': 'Current economic conditions'
    },
    'Moderate Stress': {
        'income_change': -0.10,
        'dti_change': 0.15,
        'description': 'Economic slowdown with moderate unemployment increase'
    },
    'Severe Stress': {
        'income_change': -0.20,
        'dti_change': 0.30,
        'description': 'Recession-like conditions with significant income decline'
    },
    'Agricultural Crisis': {
        'income_change': -0.25,
        'dti_change': 0.35,
        'description': 'Severe agricultural sector downturn'
    }
}


def calculate_pd_multiplier(
    income_change: float,
    dti_change: float
) -> float:
    """
    Calculate PD multiplier based on stress parameters.

    Higher income decline and DTI increase lead to higher PD.

    Parameters
    ----------
    income_change : float
        Percentage change in income (negative for decline).
    dti_change : float
        Percentage change in DTI (positive for increase).

    Returns
    -------
    float
        PD multiplier.
    """
    # PD increases when income falls and DTI rises
    multiplier = (1 - income_change) * (1 + abs(dti_change) * 0.5)

    return multiplier


def apply_stress_scenario(
    df: pd.DataFrame,
    scenario: Dict,
    agri_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply stress scenario to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with pd_hat and lgd columns.
    scenario : Dict
        Scenario parameters with pd_multiplier, lgd_add, and optionally agri_pd_multiplier.
    agri_col : str, optional
        Column indicating agricultural loans.

    Returns
    -------
    pd.DataFrame
        Dataframe with stressed PD and LGD.
    """
    df = df.copy()

    pd_multiplier = scenario.get('pd_multiplier', 1.0)
    lgd_add = scenario.get('lgd_add', 0.0)
    agri_pd_multiplier = scenario.get('agri_pd_multiplier', None)

    # Apply PD stress
    df['pd_stressed'] = df['pd_hat'] * pd_multiplier

    # Apply agricultural-specific stress if specified
    if agri_col and agri_pd_multiplier:
        agri_mask = df[agri_col] == 1
        df.loc[agri_mask, 'pd_stressed'] = df.loc[agri_mask, 'pd_hat'] * agri_pd_multiplier

    # Cap PD at 1.0
    df['pd_stressed'] = df['pd_stressed'].clip(upper=1.0)

    # Apply LGD stress if lgd column exists
    if 'lgd' in df.columns:
        df['lgd_stressed'] = (df['lgd'] + lgd_add).clip(upper=1.0)

    return df


def calculate_ecl_change(baseline_ecl: float, stressed_ecl: float) -> float:
    """
    Calculate percentage change in ECL.

    Parameters
    ----------
    baseline_ecl : float
        Baseline ECL amount.
    stressed_ecl : float
        Stressed ECL amount.

    Returns
    -------
    float
        Percentage change.
    """
    if baseline_ecl == 0:
        return 0.0
    return (stressed_ecl - baseline_ecl) / baseline_ecl * 100


def run_stress_test_suite(
    df: pd.DataFrame,
    scenarios: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Run all stress scenarios and return results.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ecl_est column.
    scenarios : Dict[str, Dict]
        Dictionary of scenario names to parameters.

    Returns
    -------
    Dict[str, Dict]
        Results for each scenario.
    """
    results = {}
    baseline_ecl = df['ecl_est'].sum()

    for scenario_name, params in scenarios.items():
        # Apply stress
        df_stressed = apply_stress_scenario(df.copy(), params)

        # Calculate stressed ECL
        if 'lgd_stressed' in df_stressed.columns:
            df_stressed['ecl_stressed'] = df_stressed['pd_stressed'] * df_stressed['lgd_stressed'] * df_stressed['ead']
        else:
            df_stressed['ecl_stressed'] = df_stressed['pd_stressed'] * df_stressed['lgd'] * df_stressed['ead']

        stressed_ecl = df_stressed['ecl_stressed'].sum()

        results[scenario_name] = {
            'total_ecl': stressed_ecl,
            'total_ecl_change': calculate_ecl_change(baseline_ecl, stressed_ecl)
        }

    return results


def calculate_segment_stress_impact(
    df: pd.DataFrame,
    segment_col: str,
    baseline_col: str = 'ecl_est',
    stressed_col: str = 'ecl_stressed'
) -> Dict:
    """
    Calculate stress impact by segment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with baseline and stressed ECL.
    segment_col : str
        Column indicating segment.
    baseline_col : str
        Column with baseline ECL.
    stressed_col : str
        Column with stressed ECL.

    Returns
    -------
    Dict
        Impact metrics by segment.
    """
    results = {}

    for segment_value in df[segment_col].unique():
        mask = df[segment_col] == segment_value
        baseline = df.loc[mask, baseline_col].sum()
        stressed = df.loc[mask, stressed_col].sum()

        change = calculate_ecl_change(baseline, stressed)

        if segment_value == 1:
            results['agri_ecl_change'] = change
        else:
            results[f'segment_{segment_value}_change'] = change
        results[segment_value] = change

    return results


def apply_stress_to_pd(
    df: pd.DataFrame,
    income_change: float,
    dti_change: float,
    pd_col: str = 'pd_hat',
    output_col: str = 'pd_stressed',
    max_pd: float = 0.95
) -> pd.DataFrame:
    """
    Apply stress scenario to PD estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with PD column.
    income_change : float
        Percentage change in income.
    dti_change : float
        Percentage change in DTI.
    pd_col : str
        Name of the PD column.
    output_col : str
        Name of the output stressed PD column.
    max_pd : float
        Maximum PD cap.

    Returns
    -------
    pd.DataFrame
        Dataframe with stressed PD.
    """
    df = df.copy()

    if income_change == 0 and dti_change == 0:
        # Baseline - no change
        df[output_col] = df[pd_col]
    else:
        multiplier = calculate_pd_multiplier(income_change, dti_change)
        df[output_col] = (df[pd_col] * multiplier).clip(upper=max_pd)

    return df


def calculate_stressed_ecl(
    df: pd.DataFrame,
    pd_stressed_col: str = 'pd_stressed',
    lgd_col: str = 'lgd_stressed',
    ead_col: str = 'ead',
    output_col: str = 'ecl_stressed'
) -> pd.DataFrame:
    """
    Calculate ECL using stressed PD and LGD.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with stressed PD, LGD, and EAD.
    pd_stressed_col : str
        Name of the stressed PD column.
    lgd_col : str
        Name of the LGD column.
    ead_col : str
        Name of the EAD column.
    output_col : str
        Name of the output stressed ECL column.

    Returns
    -------
    pd.DataFrame
        Dataframe with stressed ECL.
    """
    df = df.copy()
    df[output_col] = df[pd_stressed_col] * df[lgd_col] * df[ead_col]

    return df


def run_stress_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    scenario_params: Dict,
    pd_col: str = 'pd_hat',
    lgd_col: str = 'lgd_est',
    ead_col: str = 'ead_est',
    segment_col: Optional[str] = 'is_agri_portfolio'
) -> Dict:
    """
    Run a single stress scenario and return results.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    scenario_name : str
        Name of the scenario.
    scenario_params : Dict
        Scenario parameters (income_change, dti_change).
    pd_col : str
        Name of the PD column.
    lgd_col : str
        Name of the LGD column.
    ead_col : str
        Name of the EAD column.
    segment_col : str, optional
        Column for segment analysis.

    Returns
    -------
    Dict
        Stress scenario results.
    """
    income_change = scenario_params['income_change']
    dti_change = scenario_params['dti_change']

    # Apply stress
    df_stressed = apply_stress_to_pd(df, income_change, dti_change, pd_col)
    df_stressed = calculate_stressed_ecl(df_stressed, lgd_col=lgd_col, ead_col=ead_col)

    # Calculate totals
    total_ecl = df_stressed['ecl_stressed'].sum()
    total_ead = df_stressed[ead_col].sum()

    results = {
        'scenario': scenario_name,
        'income_change': income_change,
        'dti_change': dti_change,
        'total_ecl': total_ecl,
        'total_ead': total_ead,
        'ecl_rate': total_ecl / total_ead * 100,
        'mean_pd_stressed': df_stressed['pd_stressed'].mean()
    }

    # Segment analysis if provided
    if segment_col and segment_col in df_stressed.columns:
        # Agricultural segment
        agri_mask = df_stressed[segment_col] == 1
        results['agri_ecl'] = df_stressed.loc[agri_mask, 'ecl_stressed'].sum()
        results['agri_ead'] = df_stressed.loc[agri_mask, ead_col].sum()
        results['agri_ecl_rate'] = results['agri_ecl'] / results['agri_ead'] * 100

        # Non-agricultural segment
        non_agri_mask = df_stressed[segment_col] == 0
        results['non_agri_ecl'] = df_stressed.loc[non_agri_mask, 'ecl_stressed'].sum()
        results['non_agri_ead'] = df_stressed.loc[non_agri_mask, ead_col].sum()
        results['non_agri_ecl_rate'] = results['non_agri_ecl'] / results['non_agri_ead'] * 100

    return results


def run_all_scenarios(
    df: pd.DataFrame,
    scenarios: Dict = DEFAULT_SCENARIOS,
    pd_col: str = 'pd_hat',
    lgd_col: str = 'lgd_est',
    ead_col: str = 'ead_est',
    segment_col: Optional[str] = 'is_agri_portfolio'
) -> Dict[str, Dict]:
    """
    Run all stress scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    scenarios : Dict
        Dictionary of scenarios.
    pd_col : str
        Name of the PD column.
    lgd_col : str
        Name of the LGD column.
    ead_col : str
        Name of the EAD column.
    segment_col : str, optional
        Column for segment analysis.

    Returns
    -------
    Dict[str, Dict]
        Results for all scenarios.
    """
    results = {}

    for scenario_name, params in scenarios.items():
        results[scenario_name] = run_stress_scenario(
            df, scenario_name, params,
            pd_col=pd_col, lgd_col=lgd_col, ead_col=ead_col,
            segment_col=segment_col
        )

    return results


def calculate_stress_changes(
    stress_results: Dict[str, Dict],
    baseline_scenario: str = 'Baseline'
) -> Dict[str, Dict]:
    """
    Calculate percentage changes from baseline for all scenarios.

    Parameters
    ----------
    stress_results : Dict[str, Dict]
        Results from run_all_scenarios.
    baseline_scenario : str
        Name of the baseline scenario.

    Returns
    -------
    Dict[str, Dict]
        Results with change percentages added.
    """
    baseline = stress_results[baseline_scenario]

    for scenario_name, results in stress_results.items():
        # Total ECL change
        results['total_ecl_change'] = (
            (results['total_ecl'] - baseline['total_ecl']) /
            baseline['total_ecl'] * 100
        )

        # Segment changes if available
        if 'agri_ecl' in results and 'agri_ecl' in baseline:
            results['agri_ecl_change'] = (
                (results['agri_ecl'] - baseline['agri_ecl']) /
                baseline['agri_ecl'] * 100
            )
            results['non_agri_ecl_change'] = (
                (results['non_agri_ecl'] - baseline['non_agri_ecl']) /
                baseline['non_agri_ecl'] * 100
            )

    return stress_results


def get_stress_summary_table(stress_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create summary table of stress test results.

    Parameters
    ----------
    stress_results : Dict[str, Dict]
        Results from run_all_scenarios with changes calculated.

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    rows = []

    for scenario_name, results in stress_results.items():
        row = {
            'Scenario': scenario_name,
            'Total ECL': results['total_ecl'],
            'ECL Rate (%)': results['ecl_rate'],
            'ECL Change (%)': results.get('total_ecl_change', 0)
        }

        if 'agri_ecl' in results:
            row['Agri ECL'] = results['agri_ecl']
            row['Agri ECL Change (%)'] = results.get('agri_ecl_change', 0)

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_incremental_ecl(
    stress_results: Dict[str, Dict],
    baseline_scenario: str = 'Baseline'
) -> Dict[str, float]:
    """
    Calculate incremental ECL (additional provision required) for each scenario.

    Parameters
    ----------
    stress_results : Dict[str, Dict]
        Results from run_all_scenarios.
    baseline_scenario : str
        Name of the baseline scenario.

    Returns
    -------
    Dict[str, float]
        Incremental ECL for each scenario.
    """
    baseline_ecl = stress_results[baseline_scenario]['total_ecl']

    incremental = {}
    for scenario_name, results in stress_results.items():
        incremental[scenario_name] = results['total_ecl'] - baseline_ecl

    return incremental


def assess_portfolio_vulnerability(
    stress_results: Dict[str, Dict],
    baseline_scenario: str = 'Baseline'
) -> Dict:
    """
    Assess portfolio vulnerability based on stress test results.

    Parameters
    ----------
    stress_results : Dict[str, Dict]
        Results from run_all_scenarios with changes calculated.
    baseline_scenario : str
        Name of the baseline scenario.

    Returns
    -------
    Dict
        Vulnerability assessment metrics.
    """
    baseline = stress_results[baseline_scenario]

    # Find worst scenario
    worst_scenario = max(
        [s for s in stress_results.keys() if s != baseline_scenario],
        key=lambda s: stress_results[s]['total_ecl_change']
    )
    worst = stress_results[worst_scenario]

    assessment = {
        'baseline_ecl': baseline['total_ecl'],
        'baseline_ecl_rate': baseline['ecl_rate'],
        'worst_scenario': worst_scenario,
        'worst_ecl': worst['total_ecl'],
        'worst_ecl_change': worst['total_ecl_change'],
        'max_incremental_ecl': worst['total_ecl'] - baseline['total_ecl']
    }

    # Agricultural vulnerability
    if 'agri_ecl' in baseline:
        assessment['agri_baseline_ecl_rate'] = baseline['agri_ecl_rate']
        assessment['agri_worst_ecl_change'] = worst.get('agri_ecl_change', 0)
        assessment['agri_sensitivity_ratio'] = (
            worst.get('agri_ecl_change', 0) / worst.get('non_agri_ecl_change', 1)
            if worst.get('non_agri_ecl_change', 0) != 0 else float('inf')
        )

    return assessment
