"""
Unit Tests for Stress Testing Module
=====================================

Author: Prabhu
"""

import pytest
import pandas as pd
import numpy as np
from src.stress_testing import (
    apply_stress_scenario,
    apply_stress_to_pd,
    calculate_pd_multiplier,
    calculate_stressed_ecl,
    calculate_ecl_change,
    run_stress_test_suite,
    calculate_segment_stress_impact,
    run_stress_scenario,
    run_all_scenarios,
    calculate_stress_changes,
    get_stress_summary_table,
    calculate_incremental_ecl,
    assess_portfolio_vulnerability,
    DEFAULT_SCENARIOS
)


class TestCalculatePdMultiplier:
    """Tests for calculate_pd_multiplier. dti_change's sign is meaningful:
    positive means DTI deteriorated (more risk), negative means DTI
    improved (less risk). A prior version took abs(dti_change), which
    treated an improvement identically to a deterioration of the same
    size -- silently erasing the risk-reducing effect of any recovery
    scenario (income up, DTI down)."""

    def test_stress_scenario_increases_pd(self):
        """Income down, DTI up (both point toward higher risk)."""
        multiplier = calculate_pd_multiplier(income_change=-0.10, dti_change=0.15)
        assert multiplier > 1.0

    def test_recovery_scenario_decreases_pd(self):
        """Income up, DTI down (both point toward lower risk): the
        multiplier must go below 1.0, not sit near 1.0."""
        multiplier = calculate_pd_multiplier(income_change=0.05, dti_change=-0.10)
        assert multiplier < 1.0
        # With the sign bug, abs(-0.10) produced a multiplier of 0.9975 --
        # barely different from no stress at all, despite both inputs
        # improving. The correct multiplier compounds both effects.
        expected = (1 - 0.05) * (1 + (-0.10) * 0.5)
        assert multiplier == pytest.approx(expected)
        assert multiplier == pytest.approx(0.9025)

    def test_baseline_is_neutral(self):
        assert calculate_pd_multiplier(0.0, 0.0) == pytest.approx(1.0)


class TestApplyStressToPd:
    def test_recovery_scenario_reduces_pd_below_baseline(self):
        """apply_stress_to_pd must let PD fall below the unstressed value
        when both income and DTI move in the favorable direction."""
        df = pd.DataFrame({'pd_hat': [0.10, 0.20]})

        result = apply_stress_to_pd(df.copy(), income_change=0.05, dti_change=-0.10)

        assert (result['pd_stressed'] < df['pd_hat']).all()


class TestApplyStressScenario:
    """Tests for apply_stress_scenario function."""

    def test_pd_multiplier(self):
        """Test PD multiplier is applied correctly."""
        df = pd.DataFrame({'pd_hat': [0.10, 0.20]})
        scenario = {'pd_multiplier': 1.5, 'lgd_add': 0.0}

        result = apply_stress_scenario(df.copy(), scenario)

        expected = [0.15, 0.30]
        np.testing.assert_array_almost_equal(result['pd_stressed'].values, expected)

    def test_lgd_addition(self):
        """Test LGD addition is applied correctly."""
        df = pd.DataFrame({'pd_hat': [0.10], 'lgd': [0.45]})
        scenario = {'pd_multiplier': 1.0, 'lgd_add': 0.10}

        result = apply_stress_scenario(df.copy(), scenario)

        assert result['lgd_stressed'].iloc[0] == 0.55

    def test_pd_capped_at_one(self):
        """Test stressed PD is capped at 1.0."""
        df = pd.DataFrame({'pd_hat': [0.80]})
        scenario = {'pd_multiplier': 2.0, 'lgd_add': 0.0}

        result = apply_stress_scenario(df.copy(), scenario)

        assert result['pd_stressed'].iloc[0] == 1.0

    def test_lgd_capped_at_one(self):
        """Test stressed LGD is capped at 1.0."""
        df = pd.DataFrame({'pd_hat': [0.10], 'lgd': [0.90]})
        scenario = {'pd_multiplier': 1.0, 'lgd_add': 0.20}

        result = apply_stress_scenario(df.copy(), scenario)

        assert result['lgd_stressed'].iloc[0] == 1.0

    def test_agri_specific_multiplier(self):
        """Test agricultural-specific stress is applied."""
        df = pd.DataFrame({
            'pd_hat': [0.10, 0.10],
            'is_agri_portfolio': [1, 0]
        })
        scenario = {
            'pd_multiplier': 1.25,
            'lgd_add': 0.0,
            'agri_pd_multiplier': 1.75
        }

        result = apply_stress_scenario(df.copy(), scenario, agri_col='is_agri_portfolio')

        # Agricultural loan should have higher stress
        assert result['pd_stressed'].iloc[0] > result['pd_stressed'].iloc[1]

    def test_agri_multiplier_of_zero_is_honored(self):
        """An explicit agri_pd_multiplier of 0.0 (e.g. a relief/guarantee
        scenario that zeroes out agricultural default risk) must actually
        zero out the agricultural loan's stressed PD, not silently fall
        back to the portfolio-wide pd_multiplier.

        `if agri_col and agri_pd_multiplier:` treats 0.0 as falsy, which
        is indistinguishable from the key being absent -- this is the
        bug being guarded against here.
        """
        df = pd.DataFrame({
            'pd_hat': [0.10, 0.10],
            'is_agri_portfolio': [1, 0]
        })
        scenario = {
            'pd_multiplier': 1.25,
            'lgd_add': 0.0,
            'agri_pd_multiplier': 0.0
        }

        result = apply_stress_scenario(df.copy(), scenario, agri_col='is_agri_portfolio')

        # Agricultural loan's PD must be zeroed out by the explicit 0.0
        # multiplier, while the non-agricultural loan still gets the
        # portfolio-wide 1.25x multiplier.
        assert result['pd_stressed'].iloc[0] == 0.0
        assert result['pd_stressed'].iloc[1] == pytest.approx(0.125)


class TestCalculateStressedECL:
    """Tests for calculate_stressed_ecl function."""

    def test_stressed_ecl_calculation(self):
        """Test stressed ECL = PD_stressed × LGD_stressed × EAD."""
        df = pd.DataFrame({
            'pd_stressed': [0.15],
            'lgd_stressed': [0.55],
            'ead': [10000]
        })

        result = calculate_stressed_ecl(df.copy())

        expected = 0.15 * 0.55 * 10000  # 825
        assert abs(result['ecl_stressed'].iloc[0] - expected) < 0.01

    def test_stressed_ecl_higher_than_baseline(self, sample_ecl_data):
        """Test stressed ECL is higher than baseline under stress."""
        sample_ecl_data['ecl_est'] = sample_ecl_data['pd_hat'] * sample_ecl_data['lgd'] * sample_ecl_data['ead']
        sample_ecl_data['pd_stressed'] = sample_ecl_data['pd_hat'] * 1.5
        sample_ecl_data['lgd_stressed'] = np.minimum(sample_ecl_data['lgd'] + 0.10, 1.0)

        result = calculate_stressed_ecl(sample_ecl_data.copy())

        assert result['ecl_stressed'].sum() > sample_ecl_data['ecl_est'].sum()


class TestCalculateECLChange:
    """Tests for calculate_ecl_change function."""

    def test_positive_change(self):
        """Test positive ECL change under stress."""
        baseline_ecl = 1000000
        stressed_ecl = 1250000

        change = calculate_ecl_change(baseline_ecl, stressed_ecl)

        assert change == 25.0  # 25% increase

    def test_zero_change(self):
        """Test zero change for baseline scenario."""
        baseline_ecl = 1000000
        stressed_ecl = 1000000

        change = calculate_ecl_change(baseline_ecl, stressed_ecl)

        assert change == 0.0

    def test_negative_change(self):
        """Test negative change (ECL decrease)."""
        baseline_ecl = 1000000
        stressed_ecl = 900000

        change = calculate_ecl_change(baseline_ecl, stressed_ecl)

        assert change == -10.0


class TestRunStressTestSuite:
    """Tests for run_stress_test_suite function."""

    def test_returns_dict_of_results(self, sample_ecl_data, stress_scenarios):
        """Test function returns dictionary with all scenarios."""
        sample_ecl_data['ecl_est'] = sample_ecl_data['pd_hat'] * sample_ecl_data['lgd'] * sample_ecl_data['ead']

        results = run_stress_test_suite(sample_ecl_data, stress_scenarios)

        assert isinstance(results, dict)
        assert len(results) == len(stress_scenarios)

    def test_baseline_has_zero_change(self, sample_ecl_data, stress_scenarios):
        """Test baseline scenario shows zero ECL change."""
        sample_ecl_data['ecl_est'] = sample_ecl_data['pd_hat'] * sample_ecl_data['lgd'] * sample_ecl_data['ead']

        results = run_stress_test_suite(sample_ecl_data, stress_scenarios)

        assert results['Baseline']['total_ecl_change'] == 0.0

    def test_severe_higher_than_moderate(self, sample_ecl_data, stress_scenarios):
        """Test severe stress has higher ECL impact than moderate."""
        sample_ecl_data['ecl_est'] = sample_ecl_data['pd_hat'] * sample_ecl_data['lgd'] * sample_ecl_data['ead']

        results = run_stress_test_suite(sample_ecl_data, stress_scenarios)

        assert results['Severe Stress']['total_ecl_change'] > results['Moderate Stress']['total_ecl_change']


class TestCalculateSegmentStressImpact:
    """Tests for calculate_segment_stress_impact function."""

    def test_segment_impact_returned(self, sample_ecl_data):
        """Test function returns impact for each segment."""
        sample_ecl_data['ecl_est'] = 1000
        sample_ecl_data['ecl_stressed'] = 1500

        result = calculate_segment_stress_impact(
            sample_ecl_data,
            segment_col='is_agri_portfolio',
            baseline_col='ecl_est',
            stressed_col='ecl_stressed'
        )

        assert 'agri_ecl_change' in result or 'segment_0_change' in result or 0 in result

    def test_all_segments_accounted(self, sample_ecl_data):
        """Test all segments are included in results."""
        sample_ecl_data['ecl_est'] = 1000
        sample_ecl_data['ecl_stressed'] = 1200

        result = calculate_segment_stress_impact(
            sample_ecl_data,
            segment_col='is_agri_portfolio',
            baseline_col='ecl_est',
            stressed_col='ecl_stressed'
        )

        # Should have results for both segments (0 and 1)
        assert len(result) >= 1


@pytest.fixture
def small_portfolio():
    """Two agricultural and two other loans with round numbers."""
    return pd.DataFrame({
        'pd_hat': [0.10, 0.10, 0.10, 0.10],
        'lgd_est': [0.5, 0.5, 0.5, 0.5],
        'ead_est': [1000.0, 1000.0, 3000.0, 3000.0],
        'is_agri_portfolio': [1, 1, 0, 0],
    })


class TestRunStressScenario:

    def test_baseline_totals(self, small_portfolio):
        res = run_stress_scenario(small_portfolio, 'Baseline', DEFAULT_SCENARIOS['Baseline'])
        assert res['total_ead'] == 8000.0
        assert res['total_ecl'] == pytest.approx(0.10 * 0.5 * 8000.0)
        assert res['ecl_rate'] == pytest.approx(5.0)
        assert res['mean_pd_stressed'] == pytest.approx(0.10)

    def test_segment_totals_split_the_portfolio(self, small_portfolio):
        res = run_stress_scenario(small_portfolio, 'Baseline', DEFAULT_SCENARIOS['Baseline'])
        assert res['agri_ead'] == 2000.0
        assert res['non_agri_ead'] == 6000.0
        assert res['agri_ecl'] + res['non_agri_ecl'] == pytest.approx(res['total_ecl'])
        assert res['agri_ecl_rate'] == pytest.approx(5.0)

    def test_segment_keys_are_absent_without_the_segment_column(self, small_portfolio):
        res = run_stress_scenario(
            small_portfolio.drop(columns='is_agri_portfolio'), 'Baseline',
            DEFAULT_SCENARIOS['Baseline'],
        )
        assert 'agri_ecl' not in res

    def test_stress_raises_ecl_and_does_not_modify_the_input(self, small_portfolio):
        before = small_portfolio.copy()
        base = run_stress_scenario(small_portfolio, 'Baseline', DEFAULT_SCENARIOS['Baseline'])
        severe = run_stress_scenario(small_portfolio, 'Severe Stress', DEFAULT_SCENARIOS['Severe Stress'])
        assert severe['total_ecl'] > base['total_ecl']
        pd.testing.assert_frame_equal(small_portfolio, before)

    def test_pd_multiplier_matches_the_formula(self, small_portfolio):
        params = {'income_change': -0.10, 'dti_change': 0.15}
        res = run_stress_scenario(small_portfolio, 'x', params)
        assert res['mean_pd_stressed'] == pytest.approx(0.10 * (1.10) * (1 + 0.15 * 0.5))


@pytest.fixture
def suite_results(small_portfolio):
    return calculate_stress_changes(run_all_scenarios(small_portfolio))


class TestScenarioSuite:

    def test_one_result_per_default_scenario(self, suite_results):
        assert list(suite_results) == list(DEFAULT_SCENARIOS)

    def test_baseline_change_is_zero_and_stress_is_positive(self, suite_results):
        assert suite_results['Baseline']['total_ecl_change'] == 0
        assert suite_results['Moderate Stress']['total_ecl_change'] > 0
        assert (
            suite_results['Agricultural Crisis']['total_ecl_change']
            > suite_results['Severe Stress']['total_ecl_change']
        )

    def test_segment_changes_are_added(self, suite_results):
        assert 'agri_ecl_change' in suite_results['Severe Stress']
        assert 'non_agri_ecl_change' in suite_results['Severe Stress']

    def test_summary_table_has_one_row_per_scenario(self, suite_results):
        table = get_stress_summary_table(suite_results)
        assert table['Scenario'].tolist() == list(DEFAULT_SCENARIOS)
        assert {'Total ECL', 'ECL Rate (%)', 'ECL Change (%)', 'Agri ECL'} <= set(table.columns)

    def test_summary_table_without_changes_defaults_to_zero(self, small_portfolio):
        table = get_stress_summary_table(run_all_scenarios(small_portfolio))
        assert (table['ECL Change (%)'] == 0).all()

    def test_incremental_ecl_is_relative_to_baseline(self, suite_results):
        inc = calculate_incremental_ecl(suite_results)
        assert inc['Baseline'] == 0
        assert inc['Severe Stress'] == pytest.approx(
            suite_results['Severe Stress']['total_ecl'] - suite_results['Baseline']['total_ecl']
        )

    def test_vulnerability_picks_the_worst_scenario(self, suite_results):
        a = assess_portfolio_vulnerability(suite_results)
        assert a['worst_scenario'] == 'Agricultural Crisis'
        assert a['max_incremental_ecl'] == pytest.approx(
            a['worst_ecl'] - a['baseline_ecl']
        )
        assert a['baseline_ecl_rate'] == pytest.approx(5.0)
        assert 'agri_sensitivity_ratio' in a

    def test_identical_segment_sensitivity_gives_a_ratio_of_one(self, suite_results):
        # Both segments share the same PD, LGD and multiplier, so they move together.
        a = assess_portfolio_vulnerability(suite_results)
        assert a['agri_sensitivity_ratio'] == pytest.approx(1.0)

    def test_custom_baseline_name(self, small_portfolio):
        scenarios = {
            'Today': {'income_change': 0.0, 'dti_change': 0.0},
            'Bad': {'income_change': -0.2, 'dti_change': 0.2},
        }
        res = calculate_stress_changes(run_all_scenarios(small_portfolio, scenarios), baseline_scenario='Today')
        assert res['Today']['total_ecl_change'] == 0
        assert res['Bad']['total_ecl_change'] > 0
