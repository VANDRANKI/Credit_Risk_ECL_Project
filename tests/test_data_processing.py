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
    create_time_split,
    create_fico_buckets,
    clean_loan_data,
    validate_loan_amounts,
    validate_fico_scores,
    validate_income,
    validate_dti,
    create_income_buckets,
    parse_issue_date,
    get_data_summary,
    load_loan_data
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


class TestCreateFicoBuckets:
    """Tests for create_fico_buckets function."""

    def test_minimum_score_is_bucketed(self):
        """FICO score of exactly 300 (the valid minimum) must land in the
        '300-579' bucket, not fall through to NaN.

        pd.cut's default bins are (left, right], so without
        include_lowest=True, a value equal to the leftmost bin edge (300)
        is excluded from every bin and becomes NaN.
        """
        df = pd.DataFrame({'fico_n': [300, 579, 580, 850]})
        result = create_fico_buckets(df.copy())

        assert result['fico_bucket'].isna().sum() == 0
        assert result.loc[result['fico_n'] == 300, 'fico_bucket'].iloc[0] == '300-579'

    def test_bucket_boundaries(self):
        """Spot-check bucket assignment across all boundary values."""
        df = pd.DataFrame({'fico_n': [300, 579, 580, 619, 620, 850]})
        result = create_fico_buckets(df.copy())

        expected = {
            300: '300-579',
            579: '300-579',
            580: '580-619',
            619: '580-619',
            620: '620-659',
            850: '780-850',
        }
        for fico, bucket in expected.items():
            assert result.loc[result['fico_n'] == fico, 'fico_bucket'].iloc[0] == bucket


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


class TestCleanLoanData:
    """clean_loan_data chains the four validators."""

    def test_empty_input_does_not_divide_by_zero(self):
        """Regression: the summary line computed removed/len(df) and raised
        ZeroDivisionError for an empty frame."""
        empty = pd.DataFrame({'loan_amnt': [], 'fico_n': [], 'revenue': [], 'dti_n': []})
        out = clean_loan_data(empty)
        assert out.empty

    def test_invalid_rows_are_removed_by_each_validator(self):
        df = pd.DataFrame({
            'loan_amnt': [10000, 0, 10000, 10000, 10000, 10000],
            'fico_n': [700, 700, 200, 700, 700, 700],
            'revenue': [50000, 50000, 50000, 0, 50000, 50000],
            'dti_n': [20, 20, 20, 20, -1, 20],
        })
        out = clean_loan_data(df)
        assert out.index.tolist() == [0, 5]


class TestValidators:
    """Range checks used by clean_loan_data; the boundary values are the point."""

    def test_loan_amount_must_be_positive(self):
        df = pd.DataFrame({'loan_amnt': [-5, 0, 1, 5000]})
        assert validate_loan_amounts(df)['loan_amnt'].tolist() == [1, 5000]

    def test_fico_range_is_inclusive(self):
        df = pd.DataFrame({'fico_n': [299, 300, 850, 851]})
        assert validate_fico_scores(df)['fico_n'].tolist() == [300, 850]

    def test_fico_range_is_configurable(self):
        df = pd.DataFrame({'fico_n': [500, 600, 700]})
        out = validate_fico_scores(df, min_score=600, max_score=650)
        assert out['fico_n'].tolist() == [600]

    def test_income_excludes_zero_and_includes_the_cap(self):
        df = pd.DataFrame({'revenue': [0, 1, 10_000_000, 10_000_001]})
        assert validate_income(df)['revenue'].tolist() == [1, 10_000_000]

    def test_dti_drops_negative_and_caps_large_values(self):
        df = pd.DataFrame({'dti_n': [-1, 0, 20, 500]})
        out = validate_dti(df)
        assert out['dti_n'].tolist()[:2] == [0, 20]
        assert len(out) == 3
        assert out['dti_n'].max() < 500

    def test_validators_do_not_modify_their_input(self):
        df = pd.DataFrame({'loan_amnt': [-5, 10]})
        validate_loan_amounts(df)
        assert df['loan_amnt'].tolist() == [-5, 10]

    def test_rows_with_missing_values_are_dropped_not_kept(self):
        """NaN compares False against every bound, so it must not slip through."""
        df = pd.DataFrame({'fico_n': [np.nan, 700.0]})
        assert validate_fico_scores(df)['fico_n'].tolist() == [700.0]


class TestCleanLoanDataRequiredColumns:

    def test_missing_required_values_are_dropped(self):
        df = pd.DataFrame({
            'loan_amnt': [10000, 10000],
            'fico_n': [700, 700],
            'revenue': [50000, 50000],
            'dti_n': [20, 20],
            'purpose': ['car', None],
        })
        out = clean_loan_data(df, required_columns=['purpose'])
        assert out['purpose'].tolist() == ['car']


class TestCreateIncomeBuckets:

    @pytest.mark.parametrize('income, bucket', [
        (1, '<$40K'),
        (40000, '<$40K'),
        (40001, '$40K-$60K'),
        (100000, '$80K-$100K'),
        (150000, '$100K-$150K'),
        (150001, '>$150K'),
    ])
    def test_bucket_edges(self, income, bucket):
        out = create_income_buckets(pd.DataFrame({'revenue': [income]}))
        assert out['income_bucket'].iloc[0] == bucket

    def test_input_is_not_modified(self):
        df = pd.DataFrame({'revenue': [50000]})
        create_income_buckets(df)
        assert 'income_bucket' not in df.columns


class TestParseIssueDate:

    def test_year_and_month_are_extracted(self):
        df = pd.DataFrame({'issue_d': ['Jan-2015', 'Dec-2016']})
        out = parse_issue_date(df)
        assert out['issue_year'].tolist() == [2015, 2016]
        assert out['issue_month'].tolist() == [1, 12]

    def test_unparseable_date_raises(self):
        with pytest.raises(ValueError):
            parse_issue_date(pd.DataFrame({'issue_d': ['not a date']}))


class TestGetDataSummary:

    def test_summary_values(self):
        df = pd.DataFrame({
            'loan_amnt': [1000.0, 3000.0],
            'Default': [0, 1],
            'fico_n': [700, 720],
        })
        summary = get_data_summary(df)
        assert summary['total_loans'] == 2
        assert summary['total_exposure'] == 4000.0
        assert summary['default_rate'] == 0.5
        assert summary['avg_fico'] == 710
        assert summary['avg_loan_amount'] == 2000.0

    def test_missing_columns_give_none(self):
        summary = get_data_summary(pd.DataFrame({'x': [1]}))
        assert summary['total_loans'] == 1
        assert summary['default_rate'] is None
        assert summary['avg_income'] is None


class TestLoadLoanData:

    def test_reads_a_csv(self, tmp_path):
        path = tmp_path / 'loans.csv'
        path.write_text('loan_amnt,fico_n' + chr(10) + '1000,700' + chr(10))
        df = load_loan_data(str(path))
        assert df.to_dict('records') == [{'loan_amnt': 1000, 'fico_n': 700}]

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_loan_data(str(tmp_path / 'nope.csv'))
