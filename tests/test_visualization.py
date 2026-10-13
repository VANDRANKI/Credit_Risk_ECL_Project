"""
Unit Tests for Visualization Module
===================================

Charts are drawn with the non-interactive Agg backend (set by a fixture). The tests read the
bar heights, labels and file output back from the returned figures, so they
check that the plotted numbers are the ones computed from the input data.

Author: Prabhu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import visualization as viz


@pytest.fixture(autouse=True)
def close_figures():
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def _bar_heights(fig):
    return [p.get_height() for p in fig.axes[0].patches]


class TestDefaultRateByFico:
    def test_bars_are_default_rate_percent_per_bucket(self):
        df = pd.DataFrame({
            "fico_bucket": pd.Categorical(["low", "low", "low", "low", "high", "high"],
                                          categories=["low", "high"]),
            "Default": [1, 0, 0, 0, 0, 0],
        })
        fig = viz.plot_default_rate_by_fico(df)
        assert _bar_heights(fig) == pytest.approx([25.0, 0.0])
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert labels == ["low", "high"]

    def test_unused_categories_are_not_plotted(self):
        df = pd.DataFrame({
            "fico_bucket": pd.Categorical(["a", "a"], categories=["a", "never_seen"]),
            "Default": [1, 0],
        })
        fig = viz.plot_default_rate_by_fico(df)
        assert _bar_heights(fig) == pytest.approx([50.0])

    def test_custom_columns_and_save_path(self, tmp_path):
        df = pd.DataFrame({"band": ["x", "y"], "bad": [1, 0]})
        out = tmp_path / "fico.png"
        viz.plot_default_rate_by_fico(df, fico_col="band", default_col="bad", save_path=str(out))
        assert out.exists() and out.stat().st_size > 0


class TestEclBySegment:
    def test_segment_totals_are_in_millions(self):
        df = pd.DataFrame({
            "is_agri_portfolio": [1, 1, 0],
            "ecl_est": [1_000_000.0, 500_000.0, 4_000_000.0],
        })
        fig = viz.plot_ecl_by_segment(df)
        assert _bar_heights(fig) == pytest.approx([1.5, 4.0])
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert labels == ["Agricultural", "Non-Agricultural"]

    def test_segment_with_no_loans_plots_zero(self):
        df = pd.DataFrame({"is_agri_portfolio": [0], "ecl_est": [2_000_000.0]})
        fig = viz.plot_ecl_by_segment(df)
        assert _bar_heights(fig) == pytest.approx([0.0, 2.0])


class TestStressCharts:
    RESULTS = {
        "Baseline": {"total_ecl_change": 0.0, "agri_ecl_change": 0.0, "non_agri_ecl_change": 0.0},
        "Mild": {"total_ecl_change": 12.5, "agri_ecl_change": 20.0, "non_agri_ecl_change": 10.0},
        "Severe": {"total_ecl_change": 48.0, "agri_ecl_change": 70.0, "non_agri_ecl_change": 40.0},
    }

    def test_stress_results_plot_one_bar_per_scenario(self):
        fig = viz.plot_stress_test_results(self.RESULTS)
        assert _bar_heights(fig) == pytest.approx([0.0, 12.5, 48.0])

    def test_missing_change_key_defaults_to_zero(self):
        fig = viz.plot_stress_test_results({"Baseline": {}, "Mild": {"total_ecl_change": 5.0}})
        assert _bar_heights(fig) == pytest.approx([0.0, 5.0])

    def test_more_scenarios_than_colors_still_plots(self):
        results = {f"S{i}": {"total_ecl_change": float(i)} for i in range(6)}
        fig = viz.plot_stress_test_results(results)
        assert len(_bar_heights(fig)) == 6

    def test_vulnerability_chart_excludes_baseline(self):
        fig = viz.plot_agricultural_vulnerability(self.RESULTS)
        heights = _bar_heights(fig)
        assert heights == pytest.approx([20.0, 70.0, 10.0, 40.0])
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert labels == ["Mild", "Severe"]


class TestModelCharts:
    def test_roc_curve_reports_auc_in_legend(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        fig = viz.plot_roc_curve(y_true, scores, model_name="Toy")
        texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert texts[0] == "Toy (AUC = 0.7500)"

    def test_feature_importance_shows_top_n_sorted_ascending(self):
        imp = pd.DataFrame({
            "feature": ["a", "b", "c", "d"],
            "importance": [0.4, 0.3, 0.2, 0.1],
        })
        fig = viz.plot_feature_importance(imp, top_n=3)
        widths = [p.get_width() for p in fig.axes[0].patches]
        assert widths == pytest.approx([0.2, 0.3, 0.4])
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["c", "b", "a"]

    def test_pd_distribution_marks_the_mean(self):
        df = pd.DataFrame({"pd_hat": [0.1, 0.2, 0.3, 0.4]})
        fig = viz.plot_pd_distribution(df)
        legend = fig.axes[0].get_legend().get_texts()[0].get_text()
        assert legend == "Mean: 0.2500"

    def test_chart_formatting_hides_top_and_right_spines(self):
        fig, ax = plt.subplots()
        viz.apply_chart_formatting(ax, "T", "X", "Y")
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.get_title() == "T"
