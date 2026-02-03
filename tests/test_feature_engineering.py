"""
Unit Tests for Feature Engineering Module
==========================================

Author: Prabhu
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    create_agricultural_flag,
    create_derived_features,
    encode_fico_bucket,
    encode_emp_length,
    create_feature_matrix
)


# Top 10 agricultural states
AG_STATES = ['CA', 'IA', 'NE', 'TX', 'MN', 'IL', 'KS', 'WI', 'IN', 'NC']


class TestCreateAgriculturalFlag:
    """Tests for create_agricultural_flag function."""

    def test_agricultural_flag_positive(self):
        """Test flag is 1 for agricultural loans."""
        df = pd.DataFrame({
            'purpose': ['small_business', 'small_business'],
            'addr_state': ['IA', 'NE']
        })
        result = create_agricultural_flag(df.copy(), AG_STATES)
        assert result['is_agri_portfolio'].sum() == 2

    def test_agricultural_flag_negative(self):
        """Test flag is 0 for non-agricultural loans."""
        df = pd.DataFrame({
            'purpose': ['debt_consolidation', 'credit_card'],
            'addr_state': ['NY', 'FL']
        })
        result = create_agricultural_flag(df.copy(), AG_STATES)
        assert result['is_agri_portfolio'].sum() == 0

    def test_agricultural_flag_partial(self):
        """Test flag logic with mixed data."""
        df = pd.DataFrame({
            'purpose': ['small_business', 'small_business', 'debt_consolidation'],
            'addr_state': ['CA', 'NY', 'IA']  # CA+small_biz=1, NY+small_biz=0, IA+debt=0
        })
        result = create_agricultural_flag(df.copy(), AG_STATES)
        assert result['is_agri_portfolio'].iloc[0] == 1
        assert result['is_agri_portfolio'].iloc[1] == 0
        assert result['is_agri_portfolio'].iloc[2] == 0


class TestCreateDerivedFeatures:
    """Tests for create_derived_features function."""

    def test_loan_to_income_ratio(self):
        """Test loan to income ratio calculation."""
        df = pd.DataFrame({
            'loan_amnt': [10000, 20000],
            'revenue': [50000, 100000],
            'dti_n': [10, 20]
        })
        result = create_derived_features(df.copy())

        expected_lti = [10000/50001, 20000/100001]  # +1 to avoid division by zero
        np.testing.assert_array_almost_equal(
            result['loan_to_income'].values,
            expected_lti,
            decimal=4
        )

    def test_income_per_dti(self):
        """Test income per DTI calculation."""
        df = pd.DataFrame({
            'loan_amnt': [10000],
            'revenue': [60000],
            'dti_n': [20]
        })
        result = create_derived_features(df.copy())
        expected = 60000 / 21  # +1 to avoid division by zero
        assert abs(result['income_per_dti'].iloc[0] - expected) < 0.01

    def test_handles_zero_revenue(self):
        """Test handling of zero revenue (no division by zero)."""
        df = pd.DataFrame({
            'loan_amnt': [10000],
            'revenue': [0],
            'dti_n': [10]
        })
        result = create_derived_features(df.copy())
        assert not np.isinf(result['loan_to_income'].iloc[0])


class TestEncodeFicoBucket:
    """Tests for encode_fico_bucket function."""

    def test_encode_all_buckets(self):
        """Test encoding all FICO buckets."""
        df = pd.DataFrame({
            'fico_bucket': ['300-579', '580-619', '620-659', '660-699',
                           '700-739', '740-779', '780-850']
        })
        result = encode_fico_bucket(df.copy())

        expected = [1, 2, 3, 4, 5, 6, 7]
        assert result['fico_bucket_num'].tolist() == expected

    def test_encode_unknown_bucket(self):
        """Test handling unknown FICO bucket."""
        df = pd.DataFrame({'fico_bucket': ['unknown_bucket']})
        result = encode_fico_bucket(df.copy())
        # Should fill with default value (4 = middle bucket)
        assert result['fico_bucket_num'].iloc[0] == 4


class TestEncodeEmpLength:
    """Tests for encode_emp_length function."""

    def test_encode_known_lengths(self):
        """Test encoding known employment lengths."""
        df = pd.DataFrame({
            'emp_length': ['< 1 year', '1 year', '5 years', '10+ years']
        })
        result = encode_emp_length(df.copy())

        expected = [0.5, 1.0, 5.0, 10.0]
        assert result['emp_length_num'].tolist() == expected

    def test_encode_missing_length(self):
        """Test handling missing employment length."""
        df = pd.DataFrame({'emp_length': [np.nan, '5 years']})
        result = encode_emp_length(df.copy())

        # Missing should be filled with median
        assert not result['emp_length_num'].isna().any()


class TestCreateFeatureMatrix:
    """Tests for create_feature_matrix function."""

    def test_feature_matrix_shape(self, sample_loan_data):
        """Test feature matrix has expected columns."""
        features = ['loan_amnt', 'revenue', 'dti_n']
        X, y = create_feature_matrix(sample_loan_data, features, 'Default')

        assert len(X.columns) == len(features)
        assert len(X) == len(sample_loan_data)
        assert len(y) == len(sample_loan_data)

    def test_feature_matrix_no_nulls(self, sample_loan_data):
        """Test feature matrix has no null values."""
        features = ['loan_amnt', 'revenue']
        X, y = create_feature_matrix(sample_loan_data, features, 'Default')

        assert X.isna().sum().sum() == 0
