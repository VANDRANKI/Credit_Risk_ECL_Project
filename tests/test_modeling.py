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
    time_based_split,
    scale_features,
    predict_pd,
    save_model,
    load_model,
    compare_models
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


class TestSelectBestModelDirection:
    """Regression: selection always took the largest value, so asking for the
    Brier score (lower is better) returned the worst-calibrated model."""

    def test_brier_score_picks_the_lowest(self):
        models = {
            'calibrated': {'model': None, 'brier_score': 0.08},
            'overconfident': {'model': None, 'brier_score': 0.19},
        }

        best_name, _ = select_best_model(models, metric='brier_score')

        assert best_name == 'calibrated'

    def test_gini_still_picks_the_highest(self):
        models = {
            'a': {'model': None, 'gini': 0.30},
            'b': {'model': None, 'gini': 0.45},
        }

        assert select_best_model(models, metric='gini')[0] == 'b'

    def test_model_missing_the_metric_ranks_last(self):
        models = {
            'no_metric': {'model': None},
            'weak': {'model': None, 'gini': -0.05},
        }

        assert select_best_model(models, metric='gini')[0] == 'weak'

    def test_model_missing_brier_does_not_win_as_zero(self):
        models = {
            'no_metric': {'model': None},
            'measured': {'model': None, 'brier_score': 0.2},
        }

        assert select_best_model(models, metric='brier_score')[0] == 'measured'

    def test_works_on_real_evaluate_model_output(self, sample_features):
        X, y = sample_features
        good = train_logistic_regression(X, y)
        bad = train_logistic_regression(X * 0, y)
        models = {}
        for name, m, Xm in (('good', good, X), ('bad', bad, X * 0)):
            models[name] = {'model': m, **evaluate_model(m, Xm, y)}

        best_name, _ = select_best_model(models, metric='brier_score')

        assert models[best_name]['brier_score'] == min(
            v['brier_score'] for v in models.values()
        )


class TestScaleFeatures:
    """Tests for scale_features function."""

    def _frames(self):
        X_train = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [10.0, 10.0, 20.0, 20.0],
                                'grade': ['A', 'B', 'A', 'B']})
        X_test = pd.DataFrame({'a': [5.0, 6.0], 'b': [30.0, 10.0], 'grade': ['A', 'B']})
        return X_train, X_test

    def test_train_columns_are_standardised(self):
        X_train, X_test = self._frames()
        scaled_train, _, _ = scale_features(X_train, X_test, ['a', 'b'])

        assert scaled_train['a'].mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled_train['a'].std(ddof=0) == pytest.approx(1.0)

    def test_test_set_uses_training_statistics(self):
        """Test data must be scaled with the training mean and std, not its own."""
        X_train, X_test = self._frames()
        _, scaled_test, scaler = scale_features(X_train, X_test, ['a', 'b'])

        mean, std = X_train['a'].mean(), X_train['a'].std(ddof=0)
        assert scaled_test['a'].tolist() == pytest.approx(
            [(5.0 - mean) / std, (6.0 - mean) / std]
        )
        assert scaler.mean_[0] == pytest.approx(mean)

    def test_other_columns_and_inputs_are_untouched(self):
        X_train, X_test = self._frames()
        original_train = X_train.copy()
        scaled_train, scaled_test, _ = scale_features(X_train, X_test, ['a'])

        assert scaled_train['grade'].tolist() == ['A', 'B', 'A', 'B']
        assert scaled_test['b'].tolist() == [30.0, 10.0]
        pd.testing.assert_frame_equal(X_train, original_train)


class TestPredictPD:
    """Tests for predict_pd function."""

    def test_returns_positive_class_probability(self, sample_features):
        X, y = sample_features
        model = train_logistic_regression(X, y)

        pd_hat = predict_pd(model, X)

        assert pd_hat.shape == (len(X),)
        np.testing.assert_allclose(pd_hat, model.predict_proba(X)[:, 1])


class TestSaveAndLoadModel:
    """Tests for save_model and load_model functions."""

    def test_round_trip_keeps_predictions(self, sample_features, tmp_path):
        X, y = sample_features
        model = train_logistic_regression(X, y)
        path = str(tmp_path / 'model.joblib')

        save_model(model, path)
        loaded = load_model(path)

        np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))

    def test_loading_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / 'missing.joblib'))


class TestCompareModels:
    """Tests for compare_models function."""

    def test_one_row_per_model_indexed_by_name(self, sample_features):
        X, y = sample_features
        models = {
            'logit': train_logistic_regression(X, y),
            'xgb': train_xgboost(X, y),
        }

        table = compare_models(models, X, y)

        assert list(table.index) == ['logit', 'xgb']
        assert table.index.name == 'model'
        assert {'auc', 'gini', 'brier_score'} <= set(table.columns)

    def test_row_matches_evaluate_model(self, sample_features):
        X, y = sample_features
        model = train_logistic_regression(X, y)

        table = compare_models({'logit': model}, X, y)

        assert table.loc['logit', 'auc'] == pytest.approx(evaluate_model(model, X, y)['auc'])
