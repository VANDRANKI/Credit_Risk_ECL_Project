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
    select_best_model
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
