"""
Model training utilities for sumo bout prediction.

Models:
1. Winner Prediction - Binary classification
2. Kimarite Prediction - Multiclass classification
3. Bout Duration Prediction - Regression (if data available)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix
)
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path


# =============================================================================
# Feature Configuration
# =============================================================================

# Features to use for modeling (excludes IDs, targets, categorical strings)
NUMERIC_FEATURES = [
    # Career stats
    'east_career_wins', 'east_career_losses', 'east_career_total_bouts',
    'east_career_win_rate', 'east_career_length_days', 'east_career_total_tournaments',
    'west_career_wins', 'west_career_losses', 'west_career_total_bouts',
    'west_career_win_rate', 'west_career_length_days', 'west_career_total_tournaments',

    # Current basho
    'east_basho_wins', 'east_basho_losses',
    'east_current_win_streak', 'east_current_loss_streak',
    'west_basho_wins', 'west_basho_losses',
    'west_current_win_streak', 'west_current_loss_streak',

    # Recent form
    'east_win_rate_last_5_bouts', 'east_win_rate_last_10_bouts',
    'east_win_rate_last_15_bouts', 'east_win_rate_last_20_bouts',
    'west_win_rate_last_5_bouts', 'west_win_rate_last_10_bouts',
    'west_win_rate_last_15_bouts', 'west_win_rate_last_20_bouts',

    # Basho win rates
    'east_win_rate_last_1_basho', 'east_win_rate_last_2_basho', 'east_win_rate_last_3_basho',
    'west_win_rate_last_1_basho', 'west_win_rate_last_2_basho', 'west_win_rate_last_3_basho',

    # Streaks
    'east_kachikoshi_streak', 'east_makekoshi_streak',
    'west_kachikoshi_streak', 'west_makekoshi_streak',

    # Style - wins
    'east_pct_wins_by_push', 'east_pct_wins_by_grapple', 'east_pct_wins_by_evasion',
    'west_pct_wins_by_push', 'west_pct_wins_by_grapple', 'west_pct_wins_by_evasion',
    'east_pct_wins_by_modal_kimarite', 'east_kimarite_entropy',
    'west_pct_wins_by_modal_kimarite', 'west_kimarite_entropy',

    # Style - losses
    'east_pct_losses_by_push', 'east_pct_losses_by_grapple', 'east_pct_losses_by_evasion',
    'west_pct_losses_by_push', 'west_pct_losses_by_grapple', 'west_pct_losses_by_evasion',

    # H2H
    'east_h2h_total_bouts', 'east_h2h_wins', 'east_h2h_losses',
    'east_h2h_win_rate', 'east_h2h_never_met', 'east_h2h_current_streak',
    'east_h2h_last_result',

    # Pressure - east
    'east_needs_one_win_for_kachikoshi', 'east_needs_two_wins_for_kachikoshi',
    'east_already_kachikoshi', 'east_already_makekoshi',
    'east_day_times_needs_one_for_kachikoshi',
    'east_is_ozeki', 'east_is_yokozuna',
    'east_yokozuna_losing_record_so_far', 'east_yokozuna_multiple_losses_early',

    # Pressure - west
    'west_needs_one_win_for_kachikoshi', 'west_needs_two_wins_for_kachikoshi',
    'west_already_kachikoshi', 'west_already_makekoshi',
    'west_day_times_needs_one_for_kachikoshi',
    'west_is_ozeki', 'west_is_yokozuna',
    'west_yokozuna_losing_record_so_far', 'west_yokozuna_multiple_losses_early',

    # Ratings
    'east_elo', 'west_elo', 'elo_diff',
    'east_glicko_rating', 'west_glicko_rating', 'glicko_rating_diff',
    'east_glicko_rd', 'west_glicko_rd',
    'east_elo_minus_expected', 'west_elo_minus_expected',
    'east_glicko_minus_expected', 'west_glicko_minus_expected',

    # Rank
    'east_rank_numeric', 'west_rank_numeric', 'rank_diff',

    # Pairwise
    'career_win_rate_diff',

    # Context
    'year', 'tournament_number', 'tournament_month', 'day_of_tournament', 'is_tokyo',
]

# Top kimarite for multiclass prediction
TOP_KIMARITE = [
    'yorikiri', 'oshidashi', 'hatakikomi', 'uwatenage', 'oshitaoshi',
    'shitatenage', 'tsukiotoshi', 'hikiotoshi', 'kotenage', 'sukuinage',
    'tsukidashi', 'okuridashi', 'yoritaoshi', 'katasukashi', 'sotogake',
    'uwatedashinage', 'makiotoshi', 'tsukitaoshi', 'kimedashi', 'uchigake',
]


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_data(features_df: pd.DataFrame,
                 target: str = 'east_won',
                 feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for modeling.

    Returns X (features) and y (target).
    """
    df = features_df.copy()

    # Filter to valid targets
    df = df[df[target].notna()]

    # Get feature columns
    if feature_cols is None:
        feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Handle missing values - fill with median for numeric
    for col in X.columns:
        if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            X[col] = X[col].fillna(X[col].median())

    return X, y


def time_based_split(features_df: pd.DataFrame,
                     train_end_basho: str = '202212',
                     val_end_basho: str = '202312') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time for proper validation.

    train: everything before train_end_basho
    val: train_end_basho to val_end_basho
    test: everything after val_end_basho
    """
    df = features_df.copy()

    train = df[df['bashoId'] < train_end_basho]
    val = df[(df['bashoId'] >= train_end_basho) & (df['bashoId'] < val_end_basho)]
    test = df[df['bashoId'] >= val_end_basho]

    print(f"Train: {len(train):,} bouts (before {train_end_basho})")
    print(f"Val: {len(val):,} bouts ({train_end_basho} to {val_end_basho})")
    print(f"Test: {len(test):,} bouts (after {val_end_basho})")

    return train, val, test


# =============================================================================
# Model Training
# =============================================================================

def train_winner_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       params: Optional[Dict] = None) -> lgb.Booster:
    """
    Train binary classification model for winner prediction.
    """
    default_params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    return model


def train_kimarite_model(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         params: Optional[Dict] = None) -> Tuple[lgb.Booster, LabelEncoder]:
    """
    Train multiclass model for kimarite prediction.
    """
    # Encode kimarite labels
    le = LabelEncoder()

    # Filter to top kimarite
    y_train_filtered = y_train.apply(lambda x: x if x in TOP_KIMARITE else 'other')
    y_val_filtered = y_val.apply(lambda x: x if x in TOP_KIMARITE else 'other')

    y_train_encoded = le.fit_transform(y_train_filtered)
    y_val_encoded = le.transform(y_val_filtered)

    default_params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train_encoded)
    val_data = lgb.Dataset(X_val, label=y_val_encoded, reference=train_data)

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    return model, le


def train_duration_model(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         params: Optional[Dict] = None) -> lgb.Booster:
    """
    Train regression model for bout duration prediction.
    """
    default_params = {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    return model


# =============================================================================
# Prediction
# =============================================================================

def predict_winner(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predict P(east wins)."""
    return model.predict(X)


def predict_kimarite(model: lgb.Booster, le: LabelEncoder,
                     X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict kimarite probabilities.

    Returns:
    - predicted class (most likely kimarite)
    - probability matrix (n_samples x n_classes)
    """
    probs = model.predict(X)
    pred_classes = le.inverse_transform(probs.argmax(axis=1))
    return pred_classes, probs


def get_kimarite_category_probs(probs: np.ndarray,
                                le: LabelEncoder) -> Dict[str, np.ndarray]:
    """
    Sum kimarite probabilities into categories (push/grapple/evasion).
    """
    from .feature_engineering import KIMARITE_PUSH, KIMARITE_GRAPPLE, KIMARITE_EVASION

    classes = le.classes_

    push_idx = [i for i, c in enumerate(classes) if c in KIMARITE_PUSH]
    grapple_idx = [i for i, c in enumerate(classes) if c in KIMARITE_GRAPPLE]
    evasion_idx = [i for i, c in enumerate(classes) if c in KIMARITE_EVASION]

    return {
        'push': probs[:, push_idx].sum(axis=1) if push_idx else np.zeros(len(probs)),
        'grapple': probs[:, grapple_idx].sum(axis=1) if grapple_idx else np.zeros(len(probs)),
        'evasion': probs[:, evasion_idx].sum(axis=1) if evasion_idx else np.zeros(len(probs)),
    }


# =============================================================================
# Model Persistence
# =============================================================================

def save_model(model: lgb.Booster, path: str, name: str):
    """Save LightGBM model."""
    filepath = Path(path) / f"{name}.lgb"
    model.save_model(str(filepath))
    print(f"Saved model to {filepath}")


def load_model(path: str, name: str) -> lgb.Booster:
    """Load LightGBM model."""
    filepath = Path(path) / f"{name}.lgb"
    return lgb.Booster(model_file=str(filepath))


def save_label_encoder(le: LabelEncoder, path: str, name: str):
    """Save label encoder."""
    filepath = Path(path) / f"{name}_encoder.joblib"
    joblib.dump(le, filepath)
    print(f"Saved encoder to {filepath}")


def load_label_encoder(path: str, name: str) -> LabelEncoder:
    """Load label encoder."""
    filepath = Path(path) / f"{name}_encoder.joblib"
    return joblib.load(filepath)
