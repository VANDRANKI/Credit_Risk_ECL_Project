"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for unit tests.

Author: Prabhu
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_loan_data():
    """Create sample loan data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'loan_amnt': np.random.uniform(5000, 50000, n_samples),
        'revenue': np.random.uniform(30000, 150000, n_samples),
        'dti_n': np.random.uniform(5, 35, n_samples),
        'fico_bucket': np.random.choice(
            ['300-579', '580-619', '620-659', '660-699', '700-739', '740-779', '780-850'],
            n_samples
        ),
        'emp_length': np.random.choice(
            ['< 1 year', '1 year', '2 years', '5 years', '10+ years'],
            n_samples
        ),
        'purpose': np.random.choice(
            ['debt_consolidation', 'credit_card', 'small_business', 'home_improvement'],
            n_samples
        ),
        'home_ownership_n': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples),
        'addr_state': np.random.choice(
            ['CA', 'TX', 'NY', 'FL', 'IA', 'NE'],
            n_samples
        ),
        'Default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'issue_d': pd.date_range('2018-01-01', periods=n_samples, freq='D')
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Create sample feature matrix for model testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.85, 0.15]))

    return X, y


@pytest.fixture
def sample_ecl_data():
    """Create sample data for ECL calculations."""
    np.random.seed(42)
    n_samples = 50

    data = {
        'loan_amnt': np.random.uniform(10000, 100000, n_samples),
        'pd_hat': np.random.uniform(0.01, 0.30, n_samples),
        'lgd': np.full(n_samples, 0.45),
        'ead': np.random.uniform(10000, 100000, n_samples),
        'is_agri_portfolio': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }

    return pd.DataFrame(data)


@pytest.fixture
def stress_scenarios():
    """Create sample stress test scenarios."""
    return {
        'Baseline': {'pd_multiplier': 1.0, 'lgd_add': 0.0},
        'Moderate Stress': {'pd_multiplier': 1.25, 'lgd_add': 0.05},
        'Severe Stress': {'pd_multiplier': 1.50, 'lgd_add': 0.10}
    }
