"""
Unit Tests for Modeling Module
===============================

Author: Prabhu
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from src.modeling import (
    train_logistic_regression,
    train_xgboost,
    evaluate_model,
    get_feature_importance,
    select_best_model,
    time_based_split
)


class TestTrainLogisticRegression:
    """Tests for train_logistic_regression function."""

    def test_returns_fitted_model(self, sample_features):
        """Test function returns a fitted model."""
        X, y = sample_features
        model = train_logistic_regression(X, y)

        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'coef_')

    def test_model_predictions_valid(self, sample_features):
        """Test model produces valid probability predictions."""
        X, y = sample_features
        model = train_logistic_regression(X, y)

        predictions = model.predict_proba(X)[:, 1]
        assert all(0 <= p <= 1 for p in predictions)

    def test_with_class_weights(self, sample_features):
        """Test training with class weights."""
        X, y = sample_features
        model = train_logistic_regression(X, y, class_weight='balanced')

        assert model is not None


class TestTrainXGBoost:
    """Tests for train_xgboost function."""

    def test_returns_fitted_model(self, sample_features):
        """Test function returns a fitted model."""
        X, y = sample_features
        model = train_xgboost(X, y, n_estimators=10)  # Small for speed

        assert hasattr(model, 'predict_proba')

    def test_model_predictions_valid(self, sample_features):
        """Test model produces valid probability predictions."""
        X, y = sample_features
        model = train_xgboost(X, y, n_estimators=10)

        predictions = model.predict_proba(X)[:, 1]
        assert all(0 <= p <= 1 for p in predictions)


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_metrics_dict(self, sample_features):
        """Test function returns dictionary with metrics."""
        X, y = sample_features
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)

        assert 'auc' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_auc_in_valid_range(self, sample_features):
        """Test AUC is between 0 and 1."""
        X, y = sample_features
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)
        assert 0 <= metrics['auc'] <= 1

    def test_accuracy_in_valid_range(self, sample_features):
        """Test accuracy is between 0 and 1."""
        X, y = sample_features
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)
        assert 0 <= metrics['accuracy'] <= 1


class TestGetFeatureImportance:
    """Tests for get_feature_importance function."""

    def test_logistic_importance(self, sample_features):
        """Test feature importance for logistic regression."""
        X, y = sample_features
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        importance_df = get_feature_importance(model, X.columns.tolist())

        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(X.columns)

    def test_xgboost_importance(self, sample_features):
        """Test feature importance for XGBoost."""
        X, y = sample_features
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance_df = get_feature_importance(model, X.columns.tolist())

        assert len(importance_df) == len(X.columns)
        assert importance_df['importance'].sum() > 0

    def test_importance_sorted(self, sample_features):
        """Test importance is sorted descending."""
        X, y = sample_features
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance_df = get_feature_importance(model, X.columns.tolist())

        # Check sorted descending
        assert importance_df['importance'].is_monotonic_decreasing


class TestSelectBestModel:
    """Tests for select_best_model function."""

    def test_selects_higher_auc(self):
        """Test selects model with higher AUC."""
        models = {
            'model_a': {'model': None, 'auc': 0.65},
            'model_b': {'model': None, 'auc': 0.72}
        }

        best_name, best_info = select_best_model(models, metric='auc')

        assert best_name == 'model_b'
        assert best_info['auc'] == 0.72

    def test_handles_single_model(self):
        """Test handles single model case."""
        models = {
            'only_model': {'model': None, 'auc': 0.68}
        }

        best_name, best_info = select_best_model(models, metric='auc')

        assert best_name == 'only_model'


class TestTimeBasedSplit:
    """Tests for time_based_split function."""

    def test_normal_multi_year_split(self):
        """Sanity check: a balanced multi-year dataset splits without error
        and roughly respects the requested train ratio."""
        df = pd.DataFrame({
            'issue_year': [2016] * 40 + [2017] * 30 + [2018] * 30
        })

        train_mask, test_mask = time_based_split(df, year_column='issue_year', train_ratio=0.7)

        assert train_mask.sum() + test_mask.sum() == len(df)
        assert train_mask.sum() > 0
        assert test_mask.sum() > 0

    def test_single_year_dataset_does_not_crash(self):
        """A dataset covering a single issue year (e.g. one loan vintage)
        means every row shares one year value, so its cumulative count
        (100% of the data) always exceeds `len(df) * train_ratio` for any
        train_ratio < 1. `year_counts[year_counts <= split_threshold]` is
        then empty, and indexing it with `.index[-1]` raised an
        IndexError instead of falling back to a valid split.
        """
        df = pd.DataFrame({'issue_year': [2020] * 50})

        train_mask, test_mask = time_based_split(df, year_column='issue_year', train_ratio=0.7)

        assert train_mask.sum() + test_mask.sum() == len(df)
        assert train_mask.all()
        assert not test_mask.any()

    def test_dominant_early_year_does_not_crash(self):
        """Same root cause as the single-year case, but reached with more
        than one distinct year present: if the earliest year's share of
        records alone already exceeds train_ratio, no year's cumulative
        count qualifies and the same IndexError was raised."""
        df = pd.DataFrame({
            'issue_year': [2016] * 80 + [2017] * 15 + [2018] * 5
        })

        train_mask, test_mask = time_based_split(df, year_column='issue_year', train_ratio=0.5)

        assert train_mask.sum() + test_mask.sum() == len(df)
        assert train_mask.sum() > 0
