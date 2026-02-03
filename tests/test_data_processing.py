"""
Unit Tests for Data Processing Module
======================================

Author: Prabhu
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    validate_dataframe,
    clean_numeric_column,
    handle_missing_values,
    create_time_split
)


class TestValidateDataframe:
    """Tests for validate_dataframe function."""

    def test_valid_dataframe(self, sample_loan_data):
        """Test validation passes for valid dataframe."""
        required_cols = ['loan_amnt', 'revenue', 'Default']
        result = validate_dataframe(sample_loan_data, required_cols)
        assert result is True

    def test_missing_columns(self, sample_loan_data):
        """Test validation fails for missing columns."""
        required_cols = ['loan_amnt', 'nonexistent_column']
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(sample_loan_data, required_cols)

    def test_empty_dataframe(self):
        """Test validation fails for empty dataframe."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_dataframe(empty_df, ['col1'])


class TestCleanNumericColumn:
    """Tests for clean_numeric_column function."""

    def test_clean_valid_numbers(self):
        """Test cleaning valid numeric column."""
        df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = clean_numeric_column(df.copy(), 'value')
        assert result['value'].isna().sum() == 0

    def test_clean_with_infinities(self):
        """Test replacing infinities with NaN."""
        df = pd.DataFrame({'value': [1.0, np.inf, -np.inf, 4.0]})
        result = clean_numeric_column(df.copy(), 'value')
        assert not np.isinf(result['value']).any()

    def test_clean_with_outliers(self):
        """Test clipping extreme values."""
        df = pd.DataFrame({'value': [1.0, 2.0, 1000000.0, 4.0]})
        result = clean_numeric_column(df.copy(), 'value', clip_percentile=99)
        assert result['value'].max() < 1000000.0


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""

    def test_fill_numeric_median(self):
        """Test filling numeric NaN with median."""
        df = pd.DataFrame({'value': [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = handle_missing_values(df.copy(), 'value', strategy='median')
        assert result['value'].isna().sum() == 0
        assert result['value'].iloc[2] == 3.0  # median of 1,2,4,5

    def test_fill_numeric_mean(self):
        """Test filling numeric NaN with mean."""
        df = pd.DataFrame({'value': [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = handle_missing_values(df.copy(), 'value', strategy='mean')
        assert result['value'].isna().sum() == 0

    def test_fill_with_constant(self):
        """Test filling NaN with constant value."""
        df = pd.DataFrame({'value': [1.0, np.nan, 3.0]})
        result = handle_missing_values(df.copy(), 'value', strategy='constant', fill_value=0)
        assert result['value'].iloc[1] == 0


class TestCreateTimeSplit:
    """Tests for create_time_split function."""

    def test_split_by_date(self, sample_loan_data):
        """Test time-based train/test split."""
        train, test = create_time_split(
            sample_loan_data,
            date_col='issue_d',
            test_ratio=0.2
        )

        assert len(train) + len(test) == len(sample_loan_data)
        assert train['issue_d'].max() <= test['issue_d'].min()

    def test_split_preserves_data(self, sample_loan_data):
        """Test split preserves all data."""
        train, test = create_time_split(
            sample_loan_data,
            date_col='issue_d',
            test_ratio=0.3
        )

        total_rows = len(train) + len(test)
        assert total_rows == len(sample_loan_data)
