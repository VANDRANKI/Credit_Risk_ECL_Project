"""
Visualization Module
====================

Professional chart generation utilities for credit risk analysis.

Author: Prabhu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import os


def set_professional_style():
    """
    Set professional chart style with no grid and bold labels.
    """
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def apply_chart_formatting(ax, title: str, xlabel: str, ylabel: str):
    """
    Apply consistent formatting to chart axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to format.
    title : str
        Chart title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    """
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)

    # Bold tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Remove grid and unnecessary spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_default_rate_by_fico(
    df: pd.DataFrame,
    fico_col: str = 'fico_bucket',
    default_col: str = 'Default',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot default rate by FICO bucket.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fico_col : str
        Name of FICO bucket column.
    default_col : str
        Name of default indicator column.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    set_professional_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    fico_default = df.groupby(fico_col, observed=True)[default_col].mean() * 100
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(fico_default)))

    bars = ax.bar(range(len(fico_default)), fico_default.values,
                  color=colors, edgecolor='black')
    ax.set_xticks(range(len(fico_default)))
    ax.set_xticklabels(fico_default.index, rotation=45, fontweight='bold')

    apply_chart_formatting(ax, 'Default Rate by FICO Score Bucket',
                          'FICO Score Bucket', 'Default Rate (%)')

    # Add value labels
    for bar, val in zip(bars, fico_default.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_ecl_by_segment(
    df: pd.DataFrame,
    segment_col: str = 'is_agri_portfolio',
    ecl_col: str = 'ecl_est',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ECL by portfolio segment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    segment_col : str
        Name of segment column.
    ecl_col : str
        Name of ECL column.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    set_professional_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    agri_ecl = df[df[segment_col] == 1][ecl_col].sum() / 1e6
    non_agri_ecl = df[df[segment_col] == 0][ecl_col].sum() / 1e6

    segments = ['Agricultural', 'Non-Agricultural']
    ecls = [agri_ecl, non_agri_ecl]
    colors = ['#2ecc71', '#3498db']

    bars = ax.bar(segments, ecls, color=colors, edgecolor='black')

    apply_chart_formatting(ax, 'Total ECL by Portfolio Segment',
                          '', 'Total ECL ($ Millions)')

    # Add value labels
    for bar, val in zip(bars, ecls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${val:.1f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_stress_test_results(
    stress_results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot stress test ECL changes.

    Parameters
    ----------
    stress_results : Dict[str, Dict]
        Stress test results dictionary.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    set_professional_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    scenarios = list(stress_results.keys())
    ecl_changes = [stress_results[s].get('total_ecl_change', 0) for s in scenarios]
    colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad'][:len(scenarios)]

    bars = ax.bar(range(len(scenarios)), ecl_changes, color=colors, edgecolor='black')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=15, fontweight='bold')

    apply_chart_formatting(ax, 'ECL Impact Under Stress Scenarios',
                          '', 'ECL Change from Baseline (%)')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, ecl_changes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:+.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_agricultural_vulnerability(
    stress_results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot agricultural vs non-agricultural ECL sensitivity.

    Parameters
    ----------
    stress_results : Dict[str, Dict]
        Stress test results dictionary.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    set_professional_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Exclude baseline
    stress_scenarios = [s for s in stress_results.keys() if s != 'Baseline']
    agri_changes = [stress_results[s].get('agri_ecl_change', 0) for s in stress_scenarios]
    non_agri_changes = [stress_results[s].get('non_agri_ecl_change', 0) for s in stress_scenarios]

    x = np.arange(len(stress_scenarios))
    width = 0.35

    bars1 = ax.bar(x - width/2, agri_changes, width, label='Agricultural',
                   color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, non_agri_changes, width, label='Non-Agricultural',
                   color='#3498db', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(stress_scenarios, rotation=15, fontweight='bold')

    apply_chart_formatting(ax, 'ECL Sensitivity: Agricultural vs Non-Agricultural',
                          '', 'ECL Change from Baseline (%)')

    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred_proba : np.ndarray
        Predicted probabilities.
    model_name : str
        Name of the model for legend.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    set_professional_style()

    fig, ax = plt.subplots(figsize=(10, 8))

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')

    apply_chart_formatting(ax, 'ROC Curve',
                          'False Positive Rate', 'True Positive Rate')

    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Dataframe with 'feature' and 'importance' columns.
    top_n : int
        Number of top features to show.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    set_professional_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    top_features = importance_df.head(top_n).sort_values('importance')

    ax.barh(range(len(top_features)), top_features['importance'],
            color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontweight='bold')

    apply_chart_formatting(ax, f'Top {top_n} Features by Importance',
                          'Importance', '')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_pd_distribution(
    df: pd.DataFrame,
    pd_col: str = 'pd_hat',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot PD distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    pd_col : str
        Name of PD column.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Figure
        Figure object.
    """
    set_professional_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    df[pd_col].hist(bins=50, ax=ax, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(df[pd_col].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df[pd_col].mean():.4f}')

    apply_chart_formatting(ax, 'Distribution of Predicted Probability of Default',
                          'Predicted PD', 'Frequency')

    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
