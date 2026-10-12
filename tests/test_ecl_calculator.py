"""
Unit Tests for ECL Calculator Module
=====================================

Author: Prabhu
"""

import pytest
import pandas as pd
import numpy as np
from src.ecl_calculator import (
    calculate_ead,
    calculate_lgd,
    calculate_ecl,
    aggregate_ecl_by_segment,
    calculate_portfolio_ecl_rate,
    create_pd_bands,
    get_segment_ecl_summary,
    estimate_ead,
    estimate_lgd,
    calculate_ecl_rate,
    calculate_ecl_by_bucket,
    get_portfolio_ecl_summary,
    compute_full_ecl,
    DEFAULT_LGD,
)


class TestCalculateEAD:
    """Tests for calculate_ead function."""

    def test_ead_equals_loan_amount(self):
        """Test EAD equals loan amount (conservative approach)."""
        df = pd.DataFrame({'loan_amnt': [10000, 25000, 50000]})
        result = calculate_ead(df.copy(), 'loan_amnt')

        np.testing.assert_array_equal(result['ead'].values, df['loan_amnt'].values)

    def test_ead_with_ccf(self):
        """Test EAD with credit conversion factor."""
        df = pd.DataFrame({'loan_amnt': [10000, 20000]})
        ccf = 0.75
        result = calculate_ead(df.copy(), 'loan_amnt', ccf=ccf)

        expected = [7500, 15000]
        np.testing.assert_array_equal(result['ead'].values, expected)

    def test_ead_no_negative_values(self):
        """Test EAD has no negative values."""
        df = pd.DataFrame({'loan_amnt': [10000, 0, 50000]})
        result = calculate_ead(df.copy(), 'loan_amnt')

        assert all(result['ead'] >= 0)


class TestCalculateLGD:
    """Tests for calculate_lgd function."""

    def test_fixed_lgd(self):
        """Test fixed LGD assignment."""
        df = pd.DataFrame({'loan_amnt': [10000, 20000, 30000]})
        fixed_lgd = 0.45
        result = calculate_lgd(df.copy(), fixed_lgd=fixed_lgd)

        assert all(result['lgd'] == fixed_lgd)

    def test_lgd_bounds(self):
        """Test LGD is within valid bounds [0, 1]."""
        df = pd.DataFrame({'loan_amnt': [10000] * 100})
        result = calculate_lgd(df.copy(), fixed_lgd=0.45)

        assert all(result['lgd'] >= 0)
        assert all(result['lgd'] <= 1)

    def test_lgd_by_segment(self):
        """Test different LGD by segment."""
        df = pd.DataFrame({
            'loan_amnt': [10000, 20000],
            'segment': ['secured', 'unsecured']
        })
        lgd_map = {'secured': 0.35, 'unsecured': 0.55}
        result = calculate_lgd(df.copy(), lgd_by_segment=lgd_map, segment_col='segment')

        assert result['lgd'].iloc[0] == 0.35
        assert result['lgd'].iloc[1] == 0.55


class TestCalculateECL:
    """Tests for calculate_ecl function."""

    def test_ecl_formula(self):
        """Test ECL = PD × LGD × EAD."""
        df = pd.DataFrame({
            'pd_hat': [0.10, 0.20],
            'lgd': [0.45, 0.45],
            'ead': [10000, 20000]
        })
        result = calculate_ecl(df.copy())

        expected = [0.10 * 0.45 * 10000, 0.20 * 0.45 * 20000]
        np.testing.assert_array_almost_equal(result['ecl_est'].values, expected)

    def test_ecl_no_negative(self):
        """Test ECL has no negative values."""
        df = pd.DataFrame({
            'pd_hat': [0.05, 0.15, 0.25],
            'lgd': [0.45, 0.45, 0.45],
            'ead': [10000, 20000, 30000]
        })
        result = calculate_ecl(df.copy())

        assert all(result['ecl_est'] >= 0)

    def test_ecl_zero_when_pd_zero(self):
        """Test ECL is zero when PD is zero."""
        df = pd.DataFrame({
            'pd_hat': [0.0],
            'lgd': [0.45],
            'ead': [10000]
        })
        result = calculate_ecl(df.copy())

        assert result['ecl_est'].iloc[0] == 0

    def test_ecl_proportional_to_pd(self):
        """Test ECL increases with PD."""
        df = pd.DataFrame({
            'pd_hat': [0.10, 0.20, 0.30],
            'lgd': [0.45, 0.45, 0.45],
            'ead': [10000, 10000, 10000]
        })
        result = calculate_ecl(df.copy())

        assert result['ecl_est'].iloc[0] < result['ecl_est'].iloc[1]
        assert result['ecl_est'].iloc[1] < result['ecl_est'].iloc[2]


class TestAggregateECLBySegment:
    """Tests for aggregate_ecl_by_segment function."""

    def test_aggregate_two_segments(self, sample_ecl_data):
        """Test aggregation by agricultural segment."""
        sample_ecl_data['ecl_est'] = sample_ecl_data['pd_hat'] * sample_ecl_data['lgd'] * sample_ecl_data['ead']

        result = aggregate_ecl_by_segment(sample_ecl_data, 'is_agri_portfolio', 'ecl_est')

        assert 0 in result.index or 1 in result.index
        assert result.sum() == sample_ecl_data['ecl_est'].sum()

    def test_aggregate_preserves_total(self, sample_ecl_data):
        """Test aggregation preserves total ECL."""
        sample_ecl_data['ecl_est'] = 1000  # Fixed for easy testing

        result = aggregate_ecl_by_segment(sample_ecl_data, 'is_agri_portfolio', 'ecl_est')

        assert result.sum() == sample_ecl_data['ecl_est'].sum()


class TestCreatePdBands:
    """Tests for create_pd_bands function.

    pd.cut's default bins are (left, right], so a PD of exactly 0.0 (the
    valid minimum -- e.g. a model score floored/clipped to zero) falls
    outside every bin and comes out as NaN instead of '0-5%' unless
    include_lowest=True is passed. This mirrors the same class of boundary
    bug already fixed for FICO buckets in create_fico_buckets.
    """

    def test_minimum_pd_is_banded(self):
        """A PD of exactly 0.0 must land in the '0-5%' band, not NaN."""
        df = pd.DataFrame({'pd_hat': [0.0, 0.03, 0.5, 0.99]})
        result = create_pd_bands(df.copy())

        assert result['pd_band'].isna().sum() == 0
        assert result.loc[result['pd_hat'] == 0.0, 'pd_band'].iloc[0] == '0-5%'

    def test_band_boundaries(self):
        """Spot-check band assignment across all boundary values."""
        df = pd.DataFrame({'pd_hat': [0.0, 0.05, 0.10, 0.20, 0.30, 1.0]})
        result = create_pd_bands(df.copy())

        expected = {
            0.0: '0-5%',
            0.05: '0-5%',
            0.10: '5-10%',
            0.20: '15-20%',
            0.30: '20-30%',
            1.0: '>30%',
        }
        for pd_val, band in expected.items():
            assert result.loc[result['pd_hat'] == pd_val, 'pd_band'].iloc[0] == band


class TestGetSegmentECLSummary:
    """Tests for get_segment_ecl_summary function.

    Regression: the groupby().agg() call passed default_col='Default'
    straight through unconditionally, unlike get_portfolio_ecl_summary's
    own default_col handling which checks `if default_col in df.columns`
    first. Any dataframe without a 'Default' column -- the normal case for
    a live portfolio scored before outcomes are known -- raised
    KeyError: "Column(s) ['Default'] do not exist" instead of returning a
    summary with actual_default_rate simply omitted.
    """

    def test_summary_without_default_column_does_not_raise(self):
        df = pd.DataFrame({
            'segment': ['A', 'A', 'B'],
            'ecl_est': [10, 20, 30],
            'ead_est': [100, 200, 300],
            'pd_hat': [0.1, 0.2, 0.3],
        })
        result = get_segment_ecl_summary(df, segment_col='segment')

        assert 'actual_default_rate' not in result.columns
        assert result.loc['A', 'total_ecl'] == 30
        assert result.loc['B', 'total_ecl'] == 30

    def test_summary_with_default_column_includes_default_rate(self):
        df = pd.DataFrame({
            'segment': ['A', 'A', 'B'],
            'ecl_est': [10, 20, 30],
            'ead_est': [100, 200, 300],
            'pd_hat': [0.1, 0.2, 0.3],
            'Default': [0, 1, 0],
        })
        result = get_segment_ecl_summary(df, segment_col='segment')

        assert 'actual_default_rate' in result.columns
        assert result.loc['A', 'actual_default_rate'] == 0.5
        assert result.loc['B', 'actual_default_rate'] == 0.0


class TestCalculatePortfolioECLRate:
    """Tests for calculate_portfolio_ecl_rate function."""

    def test_ecl_rate_calculation(self):
        """Test ECL rate = Total ECL / Total EAD."""
        df = pd.DataFrame({
            'ecl_est': [1000, 2000, 3000],
            'ead': [10000, 20000, 30000]
        })

        rate = calculate_portfolio_ecl_rate(df, 'ecl_est', 'ead')

        expected_rate = 6000 / 60000  # 10%
        assert abs(rate - expected_rate) < 0.0001

    def test_ecl_rate_bounds(self):
        """Test ECL rate is within reasonable bounds."""
        df = pd.DataFrame({
            'ecl_est': [500, 1500],
            'ead': [10000, 10000]
        })

        rate = calculate_portfolio_ecl_rate(df, 'ecl_est', 'ead')

        assert 0 <= rate <= 1


class TestEstimateEadAndLgd:

    def test_ead_estimate_copies_the_loan_amount(self):
        out = estimate_ead(pd.DataFrame({'loan_amnt': [1000.0, 2500.0]}))
        assert out['ead_est'].tolist() == [1000.0, 2500.0]

    def test_lgd_estimate_defaults_to_the_basel_assumption(self):
        out = estimate_lgd(pd.DataFrame({'x': [1, 2]}))
        assert out['lgd_est'].tolist() == [DEFAULT_LGD, DEFAULT_LGD]
        assert DEFAULT_LGD == 0.45

    def test_lgd_estimate_accepts_a_custom_value_and_column(self):
        out = estimate_lgd(pd.DataFrame({'x': [1]}), lgd_value=0.6, output_col='my_lgd')
        assert out['my_lgd'].tolist() == [0.6]

    def test_inputs_are_not_modified(self):
        df = pd.DataFrame({'loan_amnt': [1000.0]})
        estimate_ead(df)
        estimate_lgd(df)
        assert list(df.columns) == ['loan_amnt']


class TestCalculateLgdFallbacks:

    def test_unknown_segment_gives_nan(self):
        df = pd.DataFrame({'seg': ['a', 'zzz']})
        out = calculate_lgd(df, lgd_by_segment={'a': 0.3}, segment_col='seg')
        assert out['lgd'].iloc[0] == 0.3
        assert np.isnan(out['lgd'].iloc[1])

    def test_no_arguments_uses_the_default(self):
        out = calculate_lgd(pd.DataFrame({'x': [1]}))
        assert out['lgd'].tolist() == [DEFAULT_LGD]

    def test_segment_map_without_segment_col_falls_back_to_default(self):
        out = calculate_lgd(pd.DataFrame({'x': [1]}), lgd_by_segment={'a': 0.3})
        assert out['lgd'].tolist() == [DEFAULT_LGD]

    def test_fixed_lgd_wins_over_the_segment_map(self):
        df = pd.DataFrame({'seg': ['a']})
        out = calculate_lgd(df, fixed_lgd=0.2, lgd_by_segment={'a': 0.9}, segment_col='seg')
        assert out['lgd'].tolist() == [0.2]


class TestEclRateAndBuckets:

    def test_ecl_rate_is_ecl_over_ead(self):
        df = pd.DataFrame({'ecl_est': [10.0, 5.0], 'ead_est': [100.0, 1000.0]})
        assert calculate_ecl_rate(df)['ecl_rate'].tolist() == [0.1, 0.005]

    def test_bucket_summary_aggregates_and_shares_sum_to_100(self):
        df = pd.DataFrame({
            'bucket': ['A', 'A', 'B'],
            'ecl_est': [10.0, 20.0, 70.0],
            'ead_est': [100.0, 100.0, 200.0],
        })
        out = calculate_ecl_by_bucket(df, 'bucket')
        assert out.loc['A', 'loan_count'] == 2
        assert out.loc['A', 'total_ead'] == 200.0
        assert out.loc['A', 'ecl_rate'] == pytest.approx(15.0)
        assert out.loc['B', 'ecl_rate'] == pytest.approx(35.0)
        assert out['ecl_pct'].sum() == pytest.approx(100.0)

    def test_unused_categories_are_not_reported(self):
        buckets = pd.Categorical(['A', 'A'], categories=['A', 'B'])
        df = pd.DataFrame({'bucket': buckets, 'ecl_est': [1.0, 2.0], 'ead_est': [10.0, 10.0]})
        assert list(calculate_ecl_by_bucket(df, 'bucket').index) == ['A']


class TestPortfolioSummary:

    def test_summary_numbers(self):
        df = pd.DataFrame({
            'ecl_est': [10.0, 30.0],
            'ead_est': [100.0, 300.0],
            'pd_hat': [0.1, 0.3],
            'Default': [0, 1],
        })
        s = get_portfolio_ecl_summary(df)
        assert s['total_loans'] == 2
        assert s['total_ead'] == 400.0
        assert s['total_ecl'] == 40.0
        assert s['ecl_rate'] == pytest.approx(10.0)
        assert s['mean_pd'] == pytest.approx(0.2)
        assert s['median_ecl'] == 20.0
        assert s['actual_default_rate'] == 50.0

    def test_default_rate_is_none_without_the_column(self):
        df = pd.DataFrame({'ecl_est': [1.0], 'ead_est': [10.0], 'pd_hat': [0.1]})
        assert get_portfolio_ecl_summary(df)['actual_default_rate'] is None


class TestComputeFullEcl:

    def test_end_to_end_matches_pd_times_lgd_times_ead(self):
        df = pd.DataFrame({'loan_amnt': [10000.0, 20000.0], 'pd_hat': [0.1, 0.02]})
        out, summary = compute_full_ecl(df, lgd_value=0.5)
        assert out['ecl_est'].tolist() == pytest.approx([500.0, 200.0])
        assert out['pd_band'].astype(str).tolist() == ['5-10%', '0-5%']
        assert summary['total_ecl'] == pytest.approx(700.0)
        assert summary['ecl_rate'] == pytest.approx(700.0 / 30000.0 * 100)

    def test_input_frame_is_left_untouched(self):
        df = pd.DataFrame({'loan_amnt': [10000.0], 'pd_hat': [0.1]})
        compute_full_ecl(df)
        assert list(df.columns) == ['loan_amnt', 'pd_hat']
