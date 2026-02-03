"""
Modeling Module
===============

Functions for training and evaluating PD (Probability of Default) models.

Author: Prabhu
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    brier_score_loss, accuracy_score, precision_score, recall_score
)
import joblib

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def time_based_split(
    df: pd.DataFrame,
    year_column: str = 'issue_year',
    train_ratio: float = 0.70
) -> Tuple[pd.Series, pd.Series]:
    """
    Create time-based train/test split masks.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with year column.
    year_column : str
        Name of the year column.
    train_ratio : float
        Proportion of data for training.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Boolean masks for train and test sets.
    """
    year_counts = df.groupby(year_column).size().cumsum()
    split_threshold = len(df) * train_ratio
    split_year = year_counts[year_counts <= split_threshold].index[-1]

    train_mask = df[year_column] <= split_year
    test_mask = df[year_column] > split_year

    print(f"Split year: {split_year}")
    print(f"Training set: {train_mask.sum():,} samples")
    print(f"Test set: {test_mask.sum():,} samples")

    return train_mask, test_mask


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    numeric_features : List[str]
        List of numeric feature names to scale.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]
        Scaled train/test dataframes and fitted scaler.
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    class_weight: Optional[str] = 'balanced'
) -> LogisticRegression:
    """
    Train logistic regression model for PD prediction.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target.
    random_state : int
        Random seed for reproducibility.
    class_weight : str, optional
        Class weight strategy.

    Returns
    -------
    LogisticRegression
        Trained logistic regression model.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight=class_weight,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)

    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 200
) -> Any:
    """
    Train XGBoost model for PD prediction.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target.
    random_state : int
        Random seed for reproducibility.
    n_estimators : int
        Number of estimators.

    Returns
    -------
    XGBClassifier or RandomForestClassifier
        Trained model (RandomForest if XGBoost not available).
    """
    if XGB_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='auc',
            use_label_encoder=False
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Parameters
    ----------
    model : Any
        Trained model with predict_proba method.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target.

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'gini': 2 * roc_auc_score(y_test, y_pred_proba) - 1,
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0)
    }

    return metrics


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    model_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Get feature importance from trained model.

    Parameters
    ----------
    model : Any
        Trained model.
    feature_names : List[str]
        List of feature names.
    model_type : str, optional
        Type of model ('tree' or 'linear'). Auto-detected if not provided.

    Returns
    -------
    pd.DataFrame
        Feature importance dataframe sorted by importance.
    """
    # Auto-detect model type
    if model_type is None:
        if hasattr(model, 'feature_importances_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            model_type = 'tree'

    if model_type == 'linear':
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


def select_best_model(
    models: Dict[str, Dict],
    metric: str = 'auc'
) -> Tuple[str, Dict]:
    """
    Select the best model based on a metric.

    Parameters
    ----------
    models : Dict[str, Dict]
        Dictionary of model name to dict with 'model' and metrics.
    metric : str
        Metric to use for selection.

    Returns
    -------
    Tuple[str, Dict]
        Name and info dict of the best model.
    """
    best_name = max(models.keys(), key=lambda x: models[x].get(metric, 0))
    return best_name, models[best_name]


def predict_pd(
    model: Any,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Generate PD predictions for loans.

    Parameters
    ----------
    model : Any
        Trained model with predict_proba method.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Array of predicted probabilities.
    """
    return model.predict_proba(X)[:, 1]


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to file.

    Parameters
    ----------
    model : Any
        Model to save.
    filepath : str
        Output file path.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load model from file.

    Parameters
    ----------
    filepath : str
        Path to model file.

    Returns
    -------
    Any
        Loaded model.
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


def compare_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models on test set.

    Parameters
    ----------
    models : Dict[str, Any]
        Dictionary of model name to model object.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target.

    Returns
    -------
    pd.DataFrame
        Comparison of model metrics.
    """
    results = []

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['model'] = name
        results.append(metrics)

    return pd.DataFrame(results).set_index('model')
