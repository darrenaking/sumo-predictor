"""
Feature engineering for sumo bout prediction.

CRITICAL: All features must be computed using ONLY information available BEFORE the bout.
No leakage from the bout itself or future bouts.

This module uses vectorized pandas/numpy operations for performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

# =============================================================================
# Constants
# =============================================================================

# Kimarite category mappings (coarse)
KIMARITE_PUSH = {
    "oshidashi", "tsukidashi", "oshitaoshi", "tsukiotoshi",
    "tsukitaoshi", "okuridashi", "abisetaoshi"
}

KIMARITE_GRAPPLE = {
    "yorikiri", "uwatenage", "shitatenage", "sukuinage", "kotenage",
    "kubinage", "yoritaoshi", "uwatedashinage", "shitatedashinage",
    "kakenage", "kirikaeshi", "tsukaminage", "tsuridashi", "tsuriotoshi",
    "utchari", "sotogake", "uchigake", "kimedashi", "kimekiri",
    "katasukashi", "okurinage", "okuritaoshi", "okurihineri",
    "okuritsuridashi", "amiuchi", "sabaori", "waridashi", "makiotoshi",
    "uwatehineri", "shitatehineri"
}

KIMARITE_EVASION = {
    "hatakikomi", "hikiotoshi", "hikkake", "ketaguri", "kekaeshi",
    "ashitori", "tsumadori", "chongake", "kawazugake", "komatasukui",
    "tottari", "izori", "shumokuzori", "tasukizori", "nichonage"
}

# Finer-grained kimarite buckets
KIMARITE_BUCKETS = {
    'force_out_standing': {'yorikiri', 'oshidashi'},
    'force_out_falling': {'yoritaoshi', 'oshitaoshi'},
    'throws_overarm': {'uwatenage', 'uwatedashinage'},
    'throws_underarm': {'shitatenage', 'shitatedashinage'},
    'slap_pull_down': {'hatakikomi', 'hikiotoshi'},
    'leg_trips': {'sotogake', 'uchigake', 'kekaeshi', 'ketaguri'},
    'twist_downs': {'katasukashi', 'shitatehineri', 'uwatehineri', 'kirikaeshi'},
    'rear_techniques': {'okuridashi', 'okuritaoshi'},
    'thrust_techniques': {'tsukiotoshi', 'tsukidashi', 'tsukitaoshi'},
    'lift_techniques': {'tsuridashi', 'amiuchi'},
}

# Grip inference from kimarite (for favored grip)
LEFT_YOTSU_KIMARITE = {'shitatenage', 'shitatedashinage', 'shitatehineri'}  # Left hand inside
RIGHT_YOTSU_KIMARITE = {'uwatenage', 'uwatedashinage', 'uwatehineri'}  # Right hand outside/over
MOROZASHI_KIMARITE = {'yorikiri', 'yoritaoshi'}  # Both hands inside (often)

# Time windows for style features
STYLE_WINDOWS = [1, 3, 6]  # basho lookback windows

# Zero-importance features to exclude (determined from model analysis)
# These features had zero gain in the trained LightGBM model
ZERO_IMPORTANCE_FEATURES = {
    'is_tokyo',
    'west_glicko_vol',
    'east_glicko_vol',
    'east_basho_year',
    'west_pct_wins_by_evasion_last_1_basho',
    'west_basho_month',
    'west_day',
    'east_pct_wins_by_lift_techniques_last_3_basho',
    'east_pct_wins_by_lift_techniques_last_1_basho',
    'east_h2h_win_rate',
    'east_h2h_wins',
    'west_pct_wins_by_leg_trips_last_1_basho',
    'west_pct_wins_by_leg_trips_last_3_basho',
    'west_pct_wins_by_slap_pull_down_last_1_basho',
    'east_already_makekoshi',
    'east_needs_one_win_for_kachikoshi',
    'east_h2h_last_result',
    'west_day_times_needs_one_for_kachikoshi',
    'west_is_day_15',
    'west_already_makekoshi',
    'west_already_kachikoshi',
    'west_needs_one_win_for_kachikoshi',
    'east_yokozuna_losing_record_so_far',
    'east_is_yokozuna',
    'east_is_ozeki',
    'east_already_kachikoshi',
    'east_day_times_needs_one_for_kachikoshi',
    'east_is_day_15',
    'west_is_ozeki',
    'east_expected_glicko',
    'west_yokozuna_losing_record_so_far',
    'west_expected_glicko',
    'year',
}

# Rank bases for parsing
RANK_BASES = {
    "Y": 0,      # Yokozuna
    "O": 10,     # Ozeki
    "S": 30,     # Sekiwake
    "K": 40,     # Komusubi
    "M": 50,     # Maegashira
    "J": 100,    # Juryo
}


# =============================================================================
# Utility Functions
# =============================================================================

def categorize_kimarite(kimarite: str) -> str:
    """Categorize a kimarite into push/grapple/evasion."""
    if not kimarite or pd.isna(kimarite):
        return "unknown"
    k = str(kimarite).lower().strip()
    if k in KIMARITE_PUSH:
        return "push"
    elif k in KIMARITE_GRAPPLE:
        return "grapple"
    elif k in KIMARITE_EVASION:
        return "evasion"
    else:
        return "unknown"


def categorize_kimarite_bucket(kimarite: str) -> str:
    """Categorize a kimarite into fine-grained bucket."""
    if not kimarite or pd.isna(kimarite):
        return "unknown"
    k = str(kimarite).lower().strip()
    for bucket_name, bucket_set in KIMARITE_BUCKETS.items():
        if k in bucket_set:
            return bucket_name
    return "other"


def infer_grip_preference(kimarite: str) -> str:
    """Infer grip preference from winning kimarite."""
    if not kimarite or pd.isna(kimarite):
        return None
    k = str(kimarite).lower().strip()
    if k in LEFT_YOTSU_KIMARITE:
        return "left_yotsu"
    elif k in RIGHT_YOTSU_KIMARITE:
        return "right_yotsu"
    elif k in MOROZASHI_KIMARITE:
        return "morozashi"
    return None


def parse_banzuke_rank(rank_str: str) -> int:
    """
    Convert banzuke rank string to numeric value.
    Lower number = higher rank.
    """
    import re
    if not rank_str or pd.isna(rank_str):
        return 999
    rank_str = str(rank_str).strip().upper()
    match = re.match(r"([YOSKM]|J)(\d+)?([EW])?", rank_str)
    if not match:
        return 999
    rank_letter = match.group(1)
    rank_num = int(match.group(2)) if match.group(2) else 1
    direction = match.group(3) if match.group(3) else "E"
    base = RANK_BASES.get(rank_letter, 999)
    numeric = base + (rank_num - 1) * 2
    if direction == "W":
        numeric += 1
    return numeric


def parse_banzuke_rank_vectorized(rank_series: pd.Series) -> pd.Series:
    """Vectorized version of rank parsing."""
    import re

    def parse_single(rank_str):
        if not rank_str or pd.isna(rank_str):
            return 999
        rank_str = str(rank_str).strip().upper()
        match = re.match(r"([YOSKM]|J)(\d+)?([EW])?", rank_str)
        if not match:
            return 999
        rank_letter = match.group(1)
        rank_num = int(match.group(2)) if match.group(2) else 1
        direction = match.group(3) if match.group(3) else "E"
        base = RANK_BASES.get(rank_letter, 999)
        numeric = base + (rank_num - 1) * 2
        if direction == "W":
            numeric += 1
        return numeric

    return rank_series.apply(parse_single)


# =============================================================================
# Vectorized Feature Engineering
# =============================================================================

class VectorizedFeatureEngine:
    """
    Vectorized feature engineering for sumo bout prediction.

    Uses pandas groupby/transform operations instead of row-by-row iteration
    where possible. Falls back to chronological processing for features that
    require strict temporal ordering (rolling stats before each bout).
    """

    def __init__(self, matches_df: pd.DataFrame,
                 rikishi_df: Optional[pd.DataFrame] = None,
                 rank_averages_df: Optional[pd.DataFrame] = None):
        """
        Initialize with match and rikishi data.

        Args:
            matches_df: DataFrame with match data (must include ratings from notebook 02)
            rikishi_df: DataFrame with rikishi profiles (optional)
            rank_averages_df: DataFrame with rank average ratings (optional)
        """
        self.matches = matches_df.copy()
        self.rikishi = rikishi_df
        self.rank_averages = rank_averages_df

        # Build lookups
        self.rikishi_lookup = {}
        if rikishi_df is not None:
            for _, row in rikishi_df.iterrows():
                self.rikishi_lookup[row.get('id')] = row.to_dict()

        self.rank_avg_lookup = {}
        if rank_averages_df is not None:
            for _, row in rank_averages_df.iterrows():
                self.rank_avg_lookup[row['rank_numeric']] = {
                    'avg_elo': row.get('avg_elo_for_rank', 1500),
                    'avg_glicko': row.get('avg_glicko_for_rank', 1500)
                }

    def _prepare_base_data(self) -> pd.DataFrame:
        """Prepare base dataframe with cleaned columns."""
        df = self.matches.copy()

        # Sort chronologically
        df = df.sort_values(['bashoId', 'day', 'matchNo'] if 'matchNo' in df.columns
                           else ['bashoId', 'day']).reset_index(drop=True)

        # Ensure bout_id exists
        if 'bout_id' not in df.columns:
            df['bout_id'] = df.index

        # Clean kimarite
        df['kimarite_clean'] = df['kimarite'].fillna('unknown').str.lower().str.strip()
        df['kimarite_known'] = df['kimarite_clean'] != 'unknown'

        # Add kimarite categories
        df['kimarite_category'] = df['kimarite_clean'].apply(categorize_kimarite)
        df['kimarite_bucket'] = df['kimarite_clean'].apply(categorize_kimarite_bucket)
        df['kimarite_grip'] = df['kimarite_clean'].apply(infer_grip_preference)

        # Parse ranks (vectorized)
        df['east_rank_numeric'] = parse_banzuke_rank_vectorized(
            df['eastRank'] if 'eastRank' in df.columns else df.get('east_rank', pd.Series([''] * len(df)))
        )
        df['west_rank_numeric'] = parse_banzuke_rank_vectorized(
            df['westRank'] if 'westRank' in df.columns else df.get('west_rank', pd.Series([''] * len(df)))
        )

        # Winner indicators
        df['east_won'] = (df['winnerId'] == df['eastId']).astype(int)
        df['west_won'] = (df['winnerId'] == df['westId']).astype(int)

        # Parse basho date info
        df['basho_year'] = df['bashoId'].astype(str).str[:4].astype(int)
        df['basho_month'] = df['bashoId'].astype(str).str[4:6].astype(int)

        # Tournament number (1-6)
        month_to_num = {1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6}
        df['tournament_number'] = df['basho_month'].map(month_to_num).fillna(0).astype(int)

        # Venue
        month_to_venue = {1: 'Tokyo', 3: 'Osaka', 5: 'Tokyo', 7: 'Nagoya', 9: 'Tokyo', 11: 'Fukuoka'}
        df['venue'] = df['basho_month'].map(month_to_venue).fillna('Unknown')
        df['is_tokyo'] = (df['venue'] == 'Tokyo').astype(int)

        return df

    def _build_wrestler_bout_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build long-form wrestler bout history for vectorized aggregations.

        Creates a dataframe where each row is a wrestler's participation in a bout,
        enabling groupby operations for rolling statistics.
        """
        # Create east perspective
        east_df = df[['bout_id', 'bashoId', 'basho_year', 'basho_month', 'day',
                      'eastId', 'westId', 'winnerId', 'kimarite_clean', 'kimarite_known',
                      'kimarite_category', 'kimarite_bucket', 'kimarite_grip']].copy()
        east_df = east_df.rename(columns={
            'eastId': 'wrestler_id',
            'westId': 'opponent_id'
        })
        east_df['won'] = (df['winnerId'] == df['eastId']).astype(int)
        east_df['position'] = 'east'

        # Create west perspective
        west_df = df[['bout_id', 'bashoId', 'basho_year', 'basho_month', 'day',
                      'westId', 'eastId', 'winnerId', 'kimarite_clean', 'kimarite_known',
                      'kimarite_category', 'kimarite_bucket', 'kimarite_grip']].copy()
        west_df = west_df.rename(columns={
            'westId': 'wrestler_id',
            'eastId': 'opponent_id'
        })
        west_df['won'] = (df['winnerId'] == df['westId']).astype(int)
        west_df['position'] = 'west'

        # Combine
        history = pd.concat([east_df, west_df], ignore_index=True)
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).reset_index(drop=True)

        return history

    def _compute_career_stats(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute career statistics using expanding windows."""
        # Sort by wrestler and time
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        # Expanding sum of wins/losses (shifted to exclude current bout)
        history['career_wins'] = history.groupby('wrestler_id')['won'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0).astype(int)

        history['career_losses'] = history.groupby('wrestler_id')['won'].transform(
            lambda x: (1 - x).shift(1).expanding().sum()
        ).fillna(0).astype(int)

        history['career_total_bouts'] = history['career_wins'] + history['career_losses']
        history['career_win_rate'] = np.where(
            history['career_total_bouts'] > 0,
            history['career_wins'] / history['career_total_bouts'],
            0.5
        )

        # Career length (approximation in days)
        history['bout_date_approx'] = history['basho_year'] * 365 + history['basho_month'] * 30 + history['day']
        history['debut_date'] = history.groupby('wrestler_id')['bout_date_approx'].transform('min')
        history['career_length_days'] = history['bout_date_approx'] - history['debut_date']

        # Tournament count
        history['basho_num'] = history.groupby('wrestler_id')['bashoId'].transform(
            lambda x: pd.factorize(x)[0]
        )
        history['career_total_tournaments'] = history.groupby('wrestler_id')['basho_num'].transform(
            lambda x: x.shift(1).expanding().max()
        ).fillna(0).astype(int) + 1

        return history

    def _compute_basho_stats(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute current tournament statistics."""
        # Create basho grouper
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        # Wins/losses in current basho before current bout
        history['basho_wins'] = history.groupby(['wrestler_id', 'bashoId'])['won'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0).astype(int)

        history['basho_losses'] = history.groupby(['wrestler_id', 'bashoId'])['won'].transform(
            lambda x: (1 - x).shift(1).expanding().sum()
        ).fillna(0).astype(int)

        return history

    def _compute_streak_stats(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute win/loss streaks."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        def compute_streak(series, target_value):
            """Compute streak of consecutive target_value occurrences."""
            streaks = []
            current_streak = 0
            for val in series:
                if pd.isna(val):
                    streaks.append(current_streak)
                elif val == target_value:
                    current_streak += 1
                    streaks.append(current_streak)
                else:
                    current_streak = 0
                    streaks.append(current_streak)
            # Shift by 1 to get streak BEFORE current bout
            return pd.Series([0] + streaks[:-1])

        history['current_win_streak'] = history.groupby('wrestler_id')['won'].transform(
            lambda x: compute_streak(x.values, 1)
        ).astype(int)

        history['current_loss_streak'] = history.groupby('wrestler_id')['won'].transform(
            lambda x: compute_streak(x.values, 0)
        ).astype(int)

        return history

    def _compute_recent_form(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute win rates over recent bouts."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        for n_bouts in [5, 10, 15, 20]:
            history[f'win_rate_last_{n_bouts}_bouts'] = history.groupby('wrestler_id')['won'].transform(
                lambda x: x.shift(1).rolling(window=n_bouts, min_periods=n_bouts).mean()
            )

        return history

    def _compute_basho_form(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute win rates over recent completed tournaments."""
        # First, compute each wrestler's record per basho
        basho_records = history.groupby(['wrestler_id', 'bashoId']).agg({
            'won': ['sum', 'count']
        }).reset_index()
        basho_records.columns = ['wrestler_id', 'bashoId', 'basho_wins_total', 'basho_bouts_total']
        basho_records['basho_win_rate'] = basho_records['basho_wins_total'] / basho_records['basho_bouts_total']
        basho_records['kachikoshi'] = (basho_records['basho_wins_total'] >= 8).astype(int)
        basho_records['makekoshi'] = (
            (basho_records['basho_wins_total'] < 8) &
            (basho_records['basho_bouts_total'] >= 8)
        ).astype(int)

        basho_records = basho_records.sort_values(['wrestler_id', 'bashoId'])

        # Rolling averages over completed basho
        for n_basho in [1, 2, 3]:
            basho_records[f'win_rate_last_{n_basho}_basho'] = basho_records.groupby('wrestler_id')['basho_win_rate'].transform(
                lambda x: x.shift(1).rolling(window=n_basho, min_periods=n_basho).mean()
            )

        # Kachikoshi/Makekoshi streaks
        def compute_kk_streak(series):
            streaks = []
            current = 0
            for val in series:
                if pd.isna(val):
                    streaks.append(0)
                elif val == 1:
                    current += 1
                    streaks.append(current)
                else:
                    current = 0
                    streaks.append(0)
            return pd.Series([0] + streaks[:-1])  # Shift

        basho_records['kachikoshi_streak'] = basho_records.groupby('wrestler_id')['kachikoshi'].transform(
            lambda x: compute_kk_streak(x.values)
        ).astype(int)

        basho_records['makekoshi_streak'] = basho_records.groupby('wrestler_id')['makekoshi'].transform(
            lambda x: compute_kk_streak(x.values)
        ).astype(int)

        # Merge back to history
        history = history.merge(
            basho_records[['wrestler_id', 'bashoId', 'win_rate_last_1_basho',
                          'win_rate_last_2_basho', 'win_rate_last_3_basho',
                          'kachikoshi_streak', 'makekoshi_streak']],
            on=['wrestler_id', 'bashoId'],
            how='left'
        )

        return history

    def _compute_style_features_windowed(self, history: pd.DataFrame) -> pd.DataFrame:
        """
        Compute style features at multiple time windows.

        For push/grapple/evasion and finer buckets, compute:
        - last 1 basho
        - last 3 basho
        - last 6 basho
        - career
        """
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        # Filter to known kimarite only for style computation
        history_known = history[history['kimarite_known']].copy()

        # Create indicators for each category
        for cat in ['push', 'grapple', 'evasion']:
            history_known[f'is_{cat}_win'] = (
                (history_known['kimarite_category'] == cat) &
                (history_known['won'] == 1)
            ).astype(int)
            history_known[f'is_{cat}_loss'] = (
                (history_known['kimarite_category'] == cat) &
                (history_known['won'] == 0)
            ).astype(int)

        # Create indicators for fine buckets
        for bucket in KIMARITE_BUCKETS.keys():
            history_known[f'is_{bucket}_win'] = (
                (history_known['kimarite_bucket'] == bucket) &
                (history_known['won'] == 1)
            ).astype(int)

        # Group by wrestler and basho to get basho-level style summaries
        basho_style = history_known.groupby(['wrestler_id', 'bashoId']).agg({
            'won': 'sum',  # Total wins in basho
            **{f'is_{cat}_win': 'sum' for cat in ['push', 'grapple', 'evasion']},
            **{f'is_{cat}_loss': 'sum' for cat in ['push', 'grapple', 'evasion']},
            **{f'is_{bucket}_win': 'sum' for bucket in KIMARITE_BUCKETS.keys()},
        }).reset_index()

        basho_style = basho_style.sort_values(['wrestler_id', 'bashoId'])

        # Compute percentages over different windows
        for window in STYLE_WINDOWS + ['career']:
            if window == 'career':
                # Expanding sum (all prior basho)
                for cat in ['push', 'grapple', 'evasion']:
                    basho_style[f'pct_wins_by_{cat}_{window}'] = basho_style.groupby('wrestler_id').apply(
                        lambda g: g[f'is_{cat}_win'].shift(1).expanding().sum() /
                                  g['won'].shift(1).expanding().sum().replace(0, np.nan)
                    ).reset_index(level=0, drop=True)

                    total_losses_career = basho_style.groupby('wrestler_id').apply(
                        lambda g: (g[f'is_{cat}_loss'].shift(1).expanding().sum() +
                                  g[f'is_{cat}_win'].shift(1).expanding().sum())
                    ).reset_index(level=0, drop=True)
                    # Skip loss percentages for brevity - can add if needed

                for bucket in KIMARITE_BUCKETS.keys():
                    basho_style[f'pct_wins_by_{bucket}_{window}'] = basho_style.groupby('wrestler_id').apply(
                        lambda g: g[f'is_{bucket}_win'].shift(1).expanding().sum() /
                                  g['won'].shift(1).expanding().sum().replace(0, np.nan)
                    ).reset_index(level=0, drop=True)
            else:
                # Rolling window over last N basho
                for cat in ['push', 'grapple', 'evasion']:
                    basho_style[f'pct_wins_by_{cat}_last_{window}_basho'] = basho_style.groupby('wrestler_id').apply(
                        lambda g: g[f'is_{cat}_win'].shift(1).rolling(window=window, min_periods=1).sum() /
                                  g['won'].shift(1).rolling(window=window, min_periods=1).sum().replace(0, np.nan)
                    ).reset_index(level=0, drop=True)

                for bucket in KIMARITE_BUCKETS.keys():
                    basho_style[f'pct_wins_by_{bucket}_last_{window}_basho'] = basho_style.groupby('wrestler_id').apply(
                        lambda g: g[f'is_{bucket}_win'].shift(1).rolling(window=window, min_periods=1).sum() /
                                  g['won'].shift(1).rolling(window=window, min_periods=1).sum().replace(0, np.nan)
                    ).reset_index(level=0, drop=True)

        # Compute style drift
        for cat in ['push', 'grapple', 'evasion']:
            basho_style[f'style_drift_{cat}'] = (
                basho_style[f'pct_wins_by_{cat}_last_3_basho'] -
                basho_style[f'pct_wins_by_{cat}_career']
            )

        # Select columns to merge back
        style_cols = ['wrestler_id', 'bashoId']
        for cat in ['push', 'grapple', 'evasion']:
            style_cols.extend([
                f'pct_wins_by_{cat}_last_1_basho',
                f'pct_wins_by_{cat}_last_3_basho',
                f'pct_wins_by_{cat}_last_6_basho',
                f'pct_wins_by_{cat}_career',
                f'style_drift_{cat}'
            ])
        for bucket in KIMARITE_BUCKETS.keys():
            style_cols.extend([
                f'pct_wins_by_{bucket}_last_1_basho',
                f'pct_wins_by_{bucket}_last_3_basho',
                f'pct_wins_by_{bucket}_last_6_basho',
                f'pct_wins_by_{bucket}_career'
            ])

        history = history.merge(
            basho_style[[c for c in style_cols if c in basho_style.columns]],
            on=['wrestler_id', 'bashoId'],
            how='left'
        )

        return history

    def _compute_favored_grip(self, history: pd.DataFrame) -> pd.DataFrame:
        """Compute favored grip from kimarite history."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        # Filter to wins with grip-indicating kimarite
        grip_bouts = history[
            (history['won'] == 1) &
            (history['kimarite_grip'].notna())
        ].copy()

        if len(grip_bouts) == 0:
            history['favored_grip'] = None
            history['favored_grip_win_rate'] = np.nan
            return history

        # Count grip types per wrestler (cumulative before each bout)
        grip_counts = grip_bouts.groupby(['wrestler_id', 'bashoId']).agg({
            'kimarite_grip': lambda x: x.value_counts().to_dict()
        }).reset_index()
        grip_counts.columns = ['wrestler_id', 'bashoId', 'grip_counts']

        # Compute cumulative grip preference
        def get_favored_grip(counts_dict):
            if not counts_dict:
                return None
            return max(counts_dict, key=counts_dict.get)

        # This is simplified - for full implementation, need cumulative counts
        # For now, mark as placeholder
        history['favored_grip'] = None
        history['favored_grip_win_rate'] = np.nan

        return history

    def _compute_h2h_features(self, df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Compute head-to-head features between wrestlers."""
        # Create H2H pairs
        h2h_bouts = history[['bout_id', 'wrestler_id', 'opponent_id', 'won',
                             'bashoId', 'day', 'kimarite_clean']].copy()

        # Sort by time
        h2h_bouts = h2h_bouts.sort_values(['wrestler_id', 'opponent_id', 'bashoId', 'day'])

        # Create pair key (always smaller id first for consistency)
        h2h_bouts['pair_key'] = h2h_bouts.apply(
            lambda r: (min(r['wrestler_id'], r['opponent_id']),
                      max(r['wrestler_id'], r['opponent_id'])), axis=1
        )

        # Compute cumulative H2H stats
        h2h_bouts['h2h_total_bouts'] = h2h_bouts.groupby(['wrestler_id', 'opponent_id']).cumcount()
        h2h_bouts['h2h_wins'] = h2h_bouts.groupby(['wrestler_id', 'opponent_id'])['won'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0).astype(int)
        h2h_bouts['h2h_losses'] = h2h_bouts['h2h_total_bouts'] - h2h_bouts['h2h_wins']
        h2h_bouts['h2h_win_rate'] = np.where(
            h2h_bouts['h2h_total_bouts'] > 0,
            h2h_bouts['h2h_wins'] / h2h_bouts['h2h_total_bouts'],
            np.nan
        )
        h2h_bouts['h2h_never_met'] = (h2h_bouts['h2h_total_bouts'] == 0).astype(int)

        # Compute H2H streak (simplified)
        def compute_h2h_streak(group):
            streaks = [0]
            current = 0
            for won in group['won'].values[:-1]:  # Exclude current
                if won == 1:
                    current = current + 1 if current >= 0 else 1
                else:
                    current = current - 1 if current <= 0 else -1
                streaks.append(current)
            return pd.Series(streaks, index=group.index)

        h2h_bouts['h2h_current_streak'] = h2h_bouts.groupby(['wrestler_id', 'opponent_id']).apply(
            compute_h2h_streak
        ).reset_index(level=[0, 1], drop=True)

        # Last result and kimarite
        h2h_bouts['h2h_last_result'] = h2h_bouts.groupby(['wrestler_id', 'opponent_id'])['won'].shift(1)
        h2h_bouts['h2h_last_bout_kimarite'] = h2h_bouts.groupby(['wrestler_id', 'opponent_id'])['kimarite_clean'].shift(1)

        # Select H2H features
        h2h_features = h2h_bouts[['bout_id', 'wrestler_id', 'h2h_total_bouts', 'h2h_wins',
                                  'h2h_losses', 'h2h_win_rate', 'h2h_never_met',
                                  'h2h_current_streak', 'h2h_last_result',
                                  'h2h_last_bout_kimarite']].copy()

        return h2h_features

    def _compute_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute pressure situation features."""
        df = df.copy()

        for prefix in ['east', 'west']:
            wins_col = f'{prefix}_basho_wins'
            losses_col = f'{prefix}_basho_losses'
            rank_col = f'{prefix}_rank_numeric'

            # Need basho stats merged first
            if wins_col not in df.columns:
                continue

            df[f'{prefix}_needs_one_win_for_kachikoshi'] = (df[wins_col] == 7).astype(int)
            df[f'{prefix}_needs_two_wins_for_kachikoshi'] = (df[wins_col] == 6).astype(int)
            df[f'{prefix}_already_kachikoshi'] = (df[wins_col] >= 8).astype(int)
            df[f'{prefix}_already_makekoshi'] = (df[losses_col] >= 8).astype(int)
            df[f'{prefix}_is_day_15'] = (df['day'] == 15).astype(int)
            df[f'{prefix}_day_times_needs_one_for_kachikoshi'] = np.where(
                df[wins_col] == 7, df['day'], 0
            )

            # Rank-based features
            if rank_col in df.columns:
                df[f'{prefix}_is_ozeki'] = ((df[rank_col] >= 10) & (df[rank_col] < 30)).astype(int)
                df[f'{prefix}_is_yokozuna'] = (df[rank_col] < 10).astype(int)
                df[f'{prefix}_yokozuna_losing_record_so_far'] = (
                    (df[rank_col] < 10) & (df[losses_col] > df[wins_col])
                ).astype(int)
                df[f'{prefix}_yokozuna_multiple_losses_early'] = (
                    (df[rank_col] < 10) & (df[losses_col] >= 2) & (df['day'] <= 7)
                ).astype(int)

        return df

    def _compute_origin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute origin features from rikishi data."""
        df = df.copy()

        for prefix, id_col in [('east', 'eastId'), ('west', 'westId')]:
            if self.rikishi is None:
                df[f'{prefix}_is_japanese'] = np.nan
                df[f'{prefix}_country_of_origin'] = None
                df[f'{prefix}_region_of_origin_japan'] = None
                continue

            # Get shusshin (birthplace) from rikishi data
            shusshin_map = {}
            for _, row in self.rikishi.iterrows():
                wrestler_id = row.get('id')
                shusshin = row.get('shusshin', row.get('birthplace', ''))
                if wrestler_id and shusshin:
                    shusshin_map[wrestler_id] = str(shusshin)

            df[f'{prefix}_shusshin'] = df[id_col].map(shusshin_map)

            # Infer country (simplified - check for known foreign origins)
            foreign_keywords = {
                'Mongolia': ['mongolia', 'ulaanbaatar'],
                'Georgia': ['georgia', 'tbilisi'],
                'Bulgaria': ['bulgaria', 'sofia'],
                'Brazil': ['brazil', 'são paulo', 'rio'],
                'USA': ['usa', 'hawaii', 'california', 'texas'],
                'Russia': ['russia', 'moscow'],
                'China': ['china', 'beijing'],
                'Estonia': ['estonia', 'tallinn'],
                'Egypt': ['egypt', 'cairo'],
            }

            def infer_country(shusshin):
                if pd.isna(shusshin):
                    return None
                s = str(shusshin).lower()
                for country, keywords in foreign_keywords.items():
                    for kw in keywords:
                        if kw in s:
                            return country
                return 'Japan'  # Default to Japan

            df[f'{prefix}_country_of_origin'] = df[f'{prefix}_shusshin'].apply(infer_country)
            df[f'{prefix}_is_japanese'] = (df[f'{prefix}_country_of_origin'] == 'Japan').astype(int)

            # Region for Japanese wrestlers (prefecture)
            df[f'{prefix}_region_of_origin_japan'] = np.where(
                df[f'{prefix}_country_of_origin'] == 'Japan',
                df[f'{prefix}_shusshin'],
                None
            )

            # Drop intermediate column
            df = df.drop(columns=[f'{prefix}_shusshin'])

        return df

    def _compute_kensho_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute kensho (prize money) features if available."""
        df = df.copy()

        # Check if kensho data exists
        if 'kensho' in df.columns or 'kensho_count' in df.columns:
            kensho_col = 'kensho' if 'kensho' in df.columns else 'kensho_count'
            df['kensho_count'] = df[kensho_col].fillna(0).astype(int)
            df['is_high_profile_bout'] = (df['kensho_count'] > 10).astype(int)
        else:
            # Not available - skip these features
            df['kensho_count'] = np.nan
            df['is_high_profile_bout'] = np.nan

        return df

    def _add_rating_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rating-related features (ELO, Glicko)."""
        df = df.copy()

        # Rating vs expected for rank
        for prefix in ['east', 'west']:
            rank_col = f'{prefix}_rank_numeric'
            elo_col = f'{prefix}_elo'
            glicko_col = f'{prefix}_glicko_rating'

            if rank_col in df.columns and elo_col in df.columns:
                # Map rank to expected rating
                df[f'{prefix}_expected_elo'] = df[rank_col].map(
                    lambda r: self.rank_avg_lookup.get(r, {}).get('avg_elo', 1500)
                )
                df[f'{prefix}_expected_glicko'] = df[rank_col].map(
                    lambda r: self.rank_avg_lookup.get(r, {}).get('avg_glicko', 1500)
                )

                df[f'{prefix}_elo_minus_expected'] = df[elo_col] - df[f'{prefix}_expected_elo']
                df[f'{prefix}_glicko_minus_expected'] = df[glicko_col] - df[f'{prefix}_expected_glicko']

        return df

    def engineer_features(self) -> pd.DataFrame:
        """
        Main entry point for feature engineering.

        Returns DataFrame with all features computed.
        """
        print("Starting vectorized feature engineering...")

        # Step 1: Prepare base data
        print("  Preparing base data...")
        df = self._prepare_base_data()

        # Step 2: Build wrestler bout history
        print("  Building wrestler bout history...")
        history = self._build_wrestler_bout_history(df)

        # Step 3: Compute career stats
        print("  Computing career stats...")
        history = self._compute_career_stats(history)

        # Step 4: Compute basho stats
        print("  Computing basho stats...")
        history = self._compute_basho_stats(history)

        # Step 5: Compute streaks
        print("  Computing streak stats...")
        history = self._compute_streak_stats(history)

        # Step 6: Compute recent form
        print("  Computing recent form...")
        history = self._compute_recent_form(history)

        # Step 7: Compute basho form (requires separate aggregation)
        print("  Computing basho form...")
        history = self._compute_basho_form(history)

        # Step 8: Compute windowed style features
        print("  Computing style features (windowed)...")
        history = self._compute_style_features_windowed(history)

        # Step 9: Compute favored grip
        print("  Computing favored grip features...")
        history = self._compute_favored_grip(history)

        # Step 10: Compute H2H features
        print("  Computing head-to-head features...")
        h2h_features = self._compute_h2h_features(df, history)

        # Separate east and west history
        east_history = history[history['position'] == 'east'].copy()
        west_history = history[history['position'] == 'west'].copy()

        # Rename columns for merging
        east_cols = {c: f'east_{c}' for c in east_history.columns
                     if c not in ['bout_id', 'wrestler_id', 'opponent_id', 'bashoId',
                                  'position', 'kimarite_clean', 'kimarite_known',
                                  'kimarite_category', 'kimarite_bucket', 'kimarite_grip']}
        west_cols = {c: f'west_{c}' for c in west_history.columns
                     if c not in ['bout_id', 'wrestler_id', 'opponent_id', 'bashoId',
                                  'position', 'kimarite_clean', 'kimarite_known',
                                  'kimarite_category', 'kimarite_bucket', 'kimarite_grip']}

        east_history = east_history.rename(columns=east_cols)
        west_history = west_history.rename(columns=west_cols)

        # Merge back to main dataframe
        print("  Merging features to bout dataframe...")
        df = df.merge(
            east_history[['bout_id'] + list(east_cols.values())],
            on='bout_id',
            how='left'
        )
        df = df.merge(
            west_history[['bout_id'] + list(west_cols.values())],
            on='bout_id',
            how='left'
        )

        # Merge H2H features for east wrestler
        east_h2h = h2h_features.copy()
        east_h2h = east_h2h.rename(columns={
            c: f'east_{c}' for c in east_h2h.columns if c not in ['bout_id', 'wrestler_id']
        })
        df = df.merge(
            east_h2h.drop(columns=['wrestler_id']),
            on='bout_id',
            how='left'
        )

        # Step 11: Compute pressure features
        print("  Computing pressure features...")
        df = self._compute_pressure_features(df)

        # Step 12: Add origin features
        print("  Computing origin features...")
        df = self._compute_origin_features(df)

        # Step 13: Add kensho features
        print("  Computing kensho features...")
        df = self._compute_kensho_features(df)

        # Step 14: Add rating features
        print("  Adding rating features...")
        df = self._add_rating_features(df)

        # Step 15: Compute pairwise differentials
        print("  Computing pairwise differentials...")
        df['rank_diff'] = df['west_rank_numeric'] - df['east_rank_numeric']
        df['career_win_rate_diff'] = df['east_career_win_rate'] - df['west_career_win_rate']

        if 'east_elo' in df.columns and 'west_elo' in df.columns:
            df['elo_diff'] = df['east_elo'] - df['west_elo']
        if 'east_glicko_rating' in df.columns and 'west_glicko_rating' in df.columns:
            df['glicko_rating_diff'] = df['east_glicko_rating'] - df['west_glicko_rating']

        # Clean up
        print("  Cleaning up...")
        df = df.drop(columns=['kimarite_clean', 'kimarite_known', 'kimarite_bucket',
                              'kimarite_grip', 'bout_date_approx'], errors='ignore')

        print(f"Feature engineering complete. Generated {len(df):,} rows with {len(df.columns)} columns.")

        return df


def create_symmetric_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create symmetric bout representation for data augmentation.

    For each bout, creates two training examples by swapping east/west.
    Both versions share the same bout_id for proper cross-validation splits.

    Args:
        df: Feature dataframe with east_ and west_ prefixed columns

    Returns:
        DataFrame with doubled rows (original + flipped)
    """
    print("Creating symmetric dataset...")

    # Original data
    original = df.copy()
    original['symmetric_version'] = 'original'

    # Find columns to swap
    east_cols = [c for c in df.columns if c.startswith('east_')]
    west_cols = [c for c in df.columns if c.startswith('west_')]

    # Create mapping for swapping
    swap_map = {}
    for ec in east_cols:
        wc = 'west_' + ec[5:]  # Remove 'east_' prefix, add 'west_'
        if wc in west_cols:
            swap_map[ec] = wc
            swap_map[wc] = ec

    # Create flipped version
    flipped = df.copy()
    flipped['symmetric_version'] = 'flipped'

    # Swap east/west columns
    for ec, wc in list(swap_map.items()):
        if ec.startswith('east_'):  # Only process once per pair
            flipped[ec], flipped[wc] = df[wc].copy(), df[ec].copy()

    # Swap IDs
    flipped['eastId'], flipped['westId'] = df['westId'].copy(), df['eastId'].copy()

    # Flip target
    if 'east_won' in flipped.columns:
        flipped['east_won'] = 1 - df['east_won']

    # Flip differentials
    diff_cols = [c for c in df.columns if '_diff' in c]
    for col in diff_cols:
        flipped[col] = -df[col]

    # Combine
    combined = pd.concat([original, flipped], ignore_index=True)

    # Sort to keep bout pairs together
    combined = combined.sort_values(['bout_id', 'symmetric_version']).reset_index(drop=True)

    print(f"Created symmetric dataset: {len(original):,} -> {len(combined):,} rows")

    return combined


def get_train_val_test_split(df: pd.DataFrame,
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/val/test split.

    Splits by bout_id to keep symmetric pairs together.
    Uses temporal ordering to prevent leakage.

    Args:
        df: Feature dataframe
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for test

    Returns:
        (train_df, val_df, test_df)
    """
    # Get unique bout_ids in temporal order
    bout_order = df.groupby('bout_id')['bashoId'].first().sort_values()
    unique_bouts = bout_order.index.tolist()

    n = len(unique_bouts)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_bouts = set(unique_bouts[:n_train])
    val_bouts = set(unique_bouts[n_train:n_train + n_val])
    test_bouts = set(unique_bouts[n_train + n_val:])

    train_df = df[df['bout_id'].isin(train_bouts)].copy()
    val_df = df[df['bout_id'].isin(val_bouts)].copy()
    test_df = df[df['bout_id'].isin(test_bouts)].copy()

    print(f"Train: {len(train_df):,} rows ({len(train_bouts):,} bouts)")
    print(f"Val: {len(val_df):,} rows ({len(val_bouts):,} bouts)")
    print(f"Test: {len(test_df):,} rows ({len(test_bouts):,} bouts)")

    return train_df, val_df, test_df


# =============================================================================
# Legacy Interface (for backward compatibility)
# =============================================================================

def engineer_features(matches_df: pd.DataFrame,
                      rikishi_df: Optional[pd.DataFrame] = None,
                      rank_averages_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Engineer all features for the matches dataset.

    This is the main entry point, using the vectorized engine.
    """
    engine = VectorizedFeatureEngine(matches_df, rikishi_df, rank_averages_df)
    return engine.engineer_features()


# Legacy class for compatibility
WrestlerStatsTracker = VectorizedFeatureEngine  # Alias


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_features(df: pd.DataFrame) -> Dict[str, any]:
    """
    Run validation checks on engineered features.

    Returns dict with validation results and any issues found.
    """
    issues = []

    # Check for unexpected NaN in critical columns
    critical_cols = ['bout_id', 'eastId', 'westId', 'bashoId', 'day']
    for col in critical_cols:
        if col in df.columns and df[col].isna().any():
            issues.append(f"Unexpected NaN in critical column: {col}")

    # Check win rate bounds
    win_rate_cols = [c for c in df.columns if 'win_rate' in c]
    for col in win_rate_cols:
        if col in df.columns:
            valid_mask = df[col].notna()
            if valid_mask.any():
                if (df.loc[valid_mask, col] < 0).any() or (df.loc[valid_mask, col] > 1).any():
                    issues.append(f"Win rate out of bounds [0,1]: {col}")

    # Check percentage bounds
    pct_cols = [c for c in df.columns if c.startswith('pct_')]
    for col in pct_cols:
        if col in df.columns:
            valid_mask = df[col].notna()
            if valid_mask.any():
                if (df.loc[valid_mask, col] < 0).any() or (df.loc[valid_mask, col] > 1).any():
                    issues.append(f"Percentage out of bounds [0,1]: {col}")

    # Check rank bounds
    rank_cols = [c for c in df.columns if 'rank_numeric' in c]
    for col in rank_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                issues.append(f"Negative rank value: {col}")

    # Summary stats
    n_rows = len(df)
    n_cols = len(df.columns)
    n_numeric = len(df.select_dtypes(include=[np.number]).columns)
    n_missing = df.isna().sum().sum()

    return {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'n_numeric_cols': n_numeric,
        'total_missing_values': n_missing,
        'pct_missing': n_missing / (n_rows * n_cols) * 100 if n_rows * n_cols > 0 else 0,
        'issues': issues,
        'valid': len(issues) == 0
    }
