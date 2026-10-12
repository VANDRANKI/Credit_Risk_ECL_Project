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
    create_feature_matrix,
    create_loan_to_income_ratio,
    create_income_per_dti,
    encode_employment_length,
    create_purpose_dummies,
    create_home_ownership_dummies,
    engineer_all_features
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


class TestRatioFeatures:

    def test_loan_to_income_uses_income_plus_one(self):
        df = pd.DataFrame({'loan_amnt': [1000.0], 'revenue': [99.0]})
        assert create_loan_to_income_ratio(df)['loan_to_income'].iloc[0] == pytest.approx(10.0)

    def test_infinite_ratio_is_replaced_by_the_median(self):
        # revenue = -1 makes the denominator zero, so the raw ratio is inf.
        df = pd.DataFrame({'loan_amnt': [100.0, 300.0, 500.0], 'revenue': [99.0, 299.0, -1.0]})
        out = create_loan_to_income_ratio(df)
        assert np.isfinite(out['loan_to_income']).all()
        assert out['loan_to_income'].iloc[2] == pytest.approx(1.0)  # median of [1.0, 1.0]

    def test_income_per_dti_uses_dti_plus_one(self):
        df = pd.DataFrame({'revenue': [5000.0], 'dti_n': [4.0]})
        assert create_income_per_dti(df)['income_per_dti'].iloc[0] == pytest.approx(1000.0)

    def test_infinite_income_per_dti_is_replaced_by_the_median(self):
        df = pd.DataFrame({'revenue': [100.0, 300.0, 50.0], 'dti_n': [1.0, 2.0, -1.0]})
        out = create_income_per_dti(df)
        assert np.isfinite(out['income_per_dti']).all()
        assert out['income_per_dti'].iloc[2] == pytest.approx(75.0)  # median of [50, 100]

    def test_ratio_helpers_do_not_modify_their_input(self):
        df = pd.DataFrame({'loan_amnt': [1.0], 'revenue': [1.0], 'dti_n': [1.0]})
        create_loan_to_income_ratio(df)
        create_income_per_dti(df)
        assert list(df.columns) == ['loan_amnt', 'revenue', 'dti_n']


class TestEncodeEmploymentLength:

    def test_known_labels_are_mapped(self):
        df = pd.DataFrame({'emp_length': ['< 1 year', '1 year', '10+ years']})
        out = encode_employment_length(df)
        assert out['emp_length_num'].tolist() == [0.5, 1, 10]

    def test_unknown_or_missing_labels_get_the_median(self):
        df = pd.DataFrame({'emp_length': ['2 years', '4 years', 'n/a', None]})
        out = encode_employment_length(df)
        assert out['emp_length_num'].tolist() == [2, 4, 3, 3]


class TestPurposeAndHomeDummies:

    def test_rare_purposes_are_grouped_as_other(self):
        df = pd.DataFrame({'purpose': ['a'] * 3 + ['b'] * 2 + ['c', 'd']})
        out, cols = create_purpose_dummies(df, top_n=2)
        assert sorted(cols) == ['purpose_a', 'purpose_b', 'purpose_other']
        assert out['purpose_grouped'].tolist() == ['a'] * 3 + ['b'] * 2 + ['other', 'other']
        assert out[cols].astype(int).sum(axis=1).eq(1).all()

    def test_no_other_column_when_everything_is_in_the_top_n(self):
        df = pd.DataFrame({'purpose': ['a', 'b', 'a']})
        _, cols = create_purpose_dummies(df, top_n=5)
        assert sorted(cols) == ['purpose_a', 'purpose_b']

    def test_home_ownership_has_one_column_per_value(self):
        df = pd.DataFrame({'home_ownership_n': ['RENT', 'OWN', 'RENT', 'MORTGAGE']})
        out, cols = create_home_ownership_dummies(df)
        assert sorted(cols) == ['home_MORTGAGE', 'home_OWN', 'home_RENT']
        assert out['home_RENT'].astype(int).tolist() == [1, 0, 1, 0]


class TestEngineerAllFeatures:

    def test_returns_frame_and_the_two_feature_lists(self, sample_loan_data):
        out, numeric, dummies = engineer_all_features(sample_loan_data)
        assert len(out) == len(sample_loan_data)
        assert {'loan_to_income', 'income_per_dti', 'fico_bucket_num', 'emp_length_num',
                'is_agri_portfolio'} <= set(out.columns)
        assert 'loan_to_income' in numeric and 'emp_length_num' in numeric
        for col in dummies:
            assert col in out.columns
        assert any(c.startswith('purpose_') for c in dummies)
        assert any(c.startswith('home_') for c in dummies)

    def test_engineered_numeric_features_have_no_missing_values(self, sample_loan_data):
        out, numeric, _ = engineer_all_features(sample_loan_data)
        engineered = ['loan_to_income', 'income_per_dti', 'fico_bucket_num', 'emp_length_num']
        assert not out[engineered].isna().any().any()


class TestCreateFeatureMatrix:

    def test_rows_missing_a_feature_or_the_target_are_dropped(self):
        df = pd.DataFrame({
            'f1': [1.0, np.nan, 3.0, 4.0],
            'f2': [1.0, 2.0, 3.0, 4.0],
            'y': [0, 1, np.nan, 1],
        })
        X, y = create_feature_matrix(df, ['f1', 'f2'], 'y')
        assert X.index.tolist() == [0, 3]
        assert y.tolist() == [0, 1]
