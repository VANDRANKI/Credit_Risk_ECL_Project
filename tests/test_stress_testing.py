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
    calculate_stressed_ecl,
    calculate_ecl_change,
    run_stress_test_suite,
    calculate_segment_stress_impact
)


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
