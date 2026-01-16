"""
Evaluation utilities for sumo bout prediction models.

Includes:
- Classification metrics (accuracy, AUC, log loss)
- Calibration analysis
- SHAP feature importance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    """
    y_pred = (y_pred_proba > threshold).astype(int)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                        n_bins: int = 10) -> Dict[str, any]:
    """
    Compute calibration metrics and binned accuracy.

    Returns:
    - ece: Expected Calibration Error
    - bin_data: DataFrame with predicted vs actual probabilities per bin
    """
    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_data = []
    total_ece = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred = y_pred_proba[mask].mean()
            bin_actual = y_true[mask].mean()
            bin_count = mask.sum()

            bin_data.append({
                'bin': i,
                'pred_prob_low': bins[i],
                'pred_prob_high': bins[i + 1],
                'mean_predicted': bin_pred,
                'mean_actual': bin_actual,
                'count': bin_count,
                'calibration_error': abs(bin_pred - bin_actual)
            })

            total_ece += abs(bin_pred - bin_actual) * bin_count

    ece = total_ece / len(y_true)

    return {
        'ece': ece,
        'bin_data': pd.DataFrame(bin_data)
    }


def compute_accuracy_by_confidence(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   confidence_bins: List[Tuple[float, float]] = None) -> pd.DataFrame:
    """
    Compute accuracy stratified by prediction confidence.

    Confidence = abs(pred - 0.5) * 2 (0 = uncertain, 1 = very confident)
    """
    if confidence_bins is None:
        confidence_bins = [
            (0.0, 0.1),   # 50-55% or 45-50%
            (0.1, 0.2),   # 55-60% or 40-45%
            (0.2, 0.4),   # 60-70% or 30-40%
            (0.4, 0.5),   # 70-75% or 25-30%
            (0.5, 1.0),   # 75%+ or 25%-
        ]

    y_pred = (y_pred_proba > 0.5).astype(int)
    confidence = np.abs(y_pred_proba - 0.5) * 2

    results = []
    for low, high in confidence_bins:
        mask = (confidence >= low) & (confidence < high)
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            results.append({
                'confidence_low': low,
                'confidence_high': high,
                'confidence_label': f'{50 + low * 25:.0f}-{50 + high * 25:.0f}%',
                'accuracy': acc,
                'count': mask.sum(),
                'pct_of_total': mask.sum() / len(y_true) * 100
            })

    return pd.DataFrame(results)


def compute_accuracy_by_feature(df: pd.DataFrame, y_true_col: str,
                                y_pred_col: str, feature_col: str,
                                n_bins: int = 5) -> pd.DataFrame:
    """
    Compute accuracy stratified by a feature value.
    """
    df = df.copy()

    # Bin numeric features
    if df[feature_col].dtype in ['float64', 'float32', 'int64', 'int32']:
        df['bin'] = pd.qcut(df[feature_col], n_bins, duplicates='drop')
    else:
        df['bin'] = df[feature_col]

    results = []
    for bin_val in df['bin'].unique():
        mask = df['bin'] == bin_val
        if mask.sum() > 0:
            subset = df[mask]
            acc = accuracy_score(subset[y_true_col], subset[y_pred_col])
            results.append({
                'bin': str(bin_val),
                'accuracy': acc,
                'count': mask.sum()
            })

    return pd.DataFrame(results).sort_values('bin')


def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                           n_bins: int = 10, ax=None):
    """Plot calibration curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(prob_pred, prob_true, 's-', label='Model')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend()

    return ax


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, ax=None):
    """Plot ROC curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()

    return ax


def plot_prediction_distribution(y_pred_proba: np.ndarray, y_true: np.ndarray = None,
                                 ax=None):
    """Plot distribution of predicted probabilities."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if y_true is not None:
        ax.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5,
                label='Actual: West won', density=True)
        ax.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5,
                label='Actual: East won', density=True)
        ax.legend()
    else:
        ax.hist(y_pred_proba, bins=50, alpha=0.7, density=True)

    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted P(East wins)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Predictions')

    return ax


def shap_analysis(model, X: pd.DataFrame, max_display: int = 20):
    """
    Compute SHAP values for feature importance analysis.

    Requires: pip install shap
    """
    import shap

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, use the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary dataframe
    importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    return shap_values, importance


def compare_to_baseline(y_true: np.ndarray, y_pred_proba: np.ndarray,
                        baselines: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compare model performance to baseline predictors.

    baselines: Dict of name -> predicted probabilities
    """
    results = []

    # Model metrics
    model_metrics = classification_metrics(y_true, y_pred_proba)
    model_metrics['model'] = 'LightGBM'
    results.append(model_metrics)

    # Baseline metrics
    for name, baseline_pred in baselines.items():
        baseline_metrics = classification_metrics(y_true, baseline_pred)
        baseline_metrics['model'] = name
        results.append(baseline_metrics)

    return pd.DataFrame(results).set_index('model')
