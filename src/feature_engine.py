"""
Vectorized feature engineering for sumo bout prediction.

Extracted from notebooks/03_feature_engineering.ipynb for use in scripts.
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import Dict, Optional


# =============================================================================
# Kimarite Categories (Coarse)
# =============================================================================

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

# =============================================================================
# Finer-Grained Kimarite Buckets
# =============================================================================

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

# Grip inference from kimarite
LEFT_YOTSU_KIMARITE = {'shitatenage', 'shitatedashinage', 'shitatehineri'}
RIGHT_YOTSU_KIMARITE = {'uwatenage', 'uwatedashinage', 'uwatehineri'}
MOROZASHI_KIMARITE = {'yorikiri', 'yoritaoshi'}

# Time windows for style features
STYLE_WINDOWS = [1, 3, 6]

# Rank bases
RANK_BASES = {"Y": 0, "O": 10, "S": 30, "K": 40, "M": 50, "J": 100}

# Zero-importance features to remove
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


def categorize_kimarite(kimarite: str) -> str:
    """Categorize kimarite into push/grapple/evasion."""
    if not kimarite or pd.isna(kimarite):
        return "unknown"
    k = str(kimarite).lower().strip()
    if k in KIMARITE_PUSH:
        return "push"
    elif k in KIMARITE_GRAPPLE:
        return "grapple"
    elif k in KIMARITE_EVASION:
        return "evasion"
    return "unknown"


def categorize_kimarite_bucket(kimarite: str) -> str:
    """Categorize kimarite into fine-grained bucket."""
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
    """Convert rank to numeric. Lower = higher rank."""
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


class VectorizedFeatureEngine:
    """
    Vectorized feature engineering for sumo bout prediction.
    Uses pandas groupby/transform for rolling statistics.
    """

    def __init__(self, matches_df, rikishi_df=None, rank_averages_df=None):
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

    def _prepare_base_data(self):
        """Prepare base dataframe with cleaned columns."""
        df = self.matches.copy()

        # Sort chronologically
        sort_cols = ['bashoId', 'day', 'matchNo'] if 'matchNo' in df.columns else ['bashoId', 'day']
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Ensure bout_id exists
        if 'bout_id' not in df.columns:
            df['bout_id'] = df.index

        # Clean kimarite
        df['kimarite_clean'] = df['kimarite'].fillna('unknown').str.lower().str.strip()
        df['kimarite_known'] = df['kimarite_clean'] != 'unknown'

        # Kimarite categories
        df['kimarite_category'] = df['kimarite_clean'].apply(categorize_kimarite)
        df['kimarite_bucket'] = df['kimarite_clean'].apply(categorize_kimarite_bucket)
        df['kimarite_grip'] = df['kimarite_clean'].apply(infer_grip_preference)

        # Parse ranks
        rank_col_east = 'eastRank' if 'eastRank' in df.columns else 'east_rank'
        rank_col_west = 'westRank' if 'westRank' in df.columns else 'west_rank'
        df['east_rank_numeric'] = df[rank_col_east].apply(parse_banzuke_rank) if rank_col_east in df.columns else 999
        df['west_rank_numeric'] = df[rank_col_west].apply(parse_banzuke_rank) if rank_col_west in df.columns else 999

        # Winner indicators
        df['east_won'] = (df['winnerId'] == df['eastId']).astype(int)
        df['west_won'] = (df['winnerId'] == df['westId']).astype(int)

        # Basho date info
        df['basho_year'] = df['bashoId'].astype(str).str[:4].astype(int)
        df['basho_month'] = df['bashoId'].astype(str).str[4:6].astype(int)

        # Tournament number and venue
        month_to_num = {1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6}
        month_to_venue = {1: 'Tokyo', 3: 'Osaka', 5: 'Tokyo', 7: 'Nagoya', 9: 'Tokyo', 11: 'Fukuoka'}
        df['tournament_number'] = df['basho_month'].map(month_to_num).fillna(0).astype(int)
        df['venue'] = df['basho_month'].map(month_to_venue).fillna('Unknown')
        df['is_tokyo'] = (df['venue'] == 'Tokyo').astype(int)

        return df

    def _build_wrestler_history(self, df):
        """Build long-form wrestler bout history."""
        # East perspective
        cols = ['bout_id', 'bashoId', 'basho_year', 'basho_month', 'day',
                'eastId', 'westId', 'winnerId', 'kimarite_clean', 'kimarite_known',
                'kimarite_category', 'kimarite_bucket', 'kimarite_grip']

        east_df = df[cols].copy()
        east_df = east_df.rename(columns={'eastId': 'wrestler_id', 'westId': 'opponent_id'})
        east_df['won'] = (df['winnerId'] == df['eastId']).astype(int)
        east_df['position'] = 'east'

        west_df = df[cols].copy()
        west_df = west_df.rename(columns={'westId': 'wrestler_id', 'eastId': 'opponent_id'})
        west_df['won'] = (df['winnerId'] == df['westId']).astype(int)
        west_df['position'] = 'west'

        history = pd.concat([east_df, west_df], ignore_index=True)
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).reset_index(drop=True)

        return history

    def _compute_career_stats(self, history):
        """Compute career statistics using expanding windows."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        # Expanding wins/losses (shifted to exclude current)
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

        # Career length
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

    def _compute_basho_stats(self, history):
        """Compute current tournament statistics."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        history['basho_wins'] = history.groupby(['wrestler_id', 'bashoId'])['won'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0).astype(int)

        history['basho_losses'] = history.groupby(['wrestler_id', 'bashoId'])['won'].transform(
            lambda x: (1 - x).shift(1).expanding().sum()
        ).fillna(0).astype(int)

        return history

    def _compute_streak_stats(self, history):
        """Compute win/loss streaks."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        def compute_streak(series, target):
            """Compute streak, returning series with proper index alignment."""
            streaks = []
            current = 0
            for val in series.values:
                if pd.isna(val):
                    streaks.append(current)
                elif val == target:
                    current += 1
                    streaks.append(current)
                else:
                    current = 0
                    streaks.append(current)
            # Shift by 1 to get streak BEFORE current bout
            result = [0] + streaks[:-1]
            return pd.Series(result, index=series.index)

        history['current_win_streak'] = history.groupby('wrestler_id')['won'].transform(
            lambda x: compute_streak(x, 1)
        ).fillna(0).astype(int)

        history['current_loss_streak'] = history.groupby('wrestler_id')['won'].transform(
            lambda x: compute_streak(x, 0)
        ).fillna(0).astype(int)

        return history

    def _compute_recent_form(self, history):
        """Compute win rates over recent bouts."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()

        for n_bouts in [5, 10, 15, 20]:
            history[f'win_rate_last_{n_bouts}_bouts'] = history.groupby('wrestler_id')['won'].transform(
                lambda x: x.shift(1).rolling(window=n_bouts, min_periods=n_bouts).mean()
            )

        return history

    def _compute_basho_form(self, history):
        """Compute win rates over recent completed tournaments."""
        basho_records = history.groupby(['wrestler_id', 'bashoId']).agg({
            'won': ['sum', 'count']
        }).reset_index()
        basho_records.columns = ['wrestler_id', 'bashoId', 'basho_wins_total', 'basho_bouts_total']
        basho_records['basho_win_rate'] = basho_records['basho_wins_total'] / basho_records['basho_bouts_total']
        basho_records['kachikoshi'] = (basho_records['basho_wins_total'] >= 8).astype(int)
        basho_records['makekoshi'] = (
            (basho_records['basho_wins_total'] < 8) & (basho_records['basho_bouts_total'] >= 8)
        ).astype(int)

        basho_records = basho_records.sort_values(['wrestler_id', 'bashoId'])

        for n_basho in [1, 2, 3]:
            basho_records[f'win_rate_last_{n_basho}_basho'] = basho_records.groupby('wrestler_id')['basho_win_rate'].transform(
                lambda x: x.shift(1).rolling(window=n_basho, min_periods=n_basho).mean()
            )

        def compute_kk_streak(series):
            streaks = []
            current = 0
            for val in series.values:
                if pd.isna(val):
                    streaks.append(0)
                elif val == 1:
                    current += 1
                    streaks.append(current)
                else:
                    current = 0
                    streaks.append(0)
            result = [0] + streaks[:-1]
            return pd.Series(result, index=series.index)

        basho_records['kachikoshi_streak'] = basho_records.groupby('wrestler_id')['kachikoshi'].transform(
            lambda x: compute_kk_streak(x)
        ).fillna(0).astype(int)

        basho_records['makekoshi_streak'] = basho_records.groupby('wrestler_id')['makekoshi'].transform(
            lambda x: compute_kk_streak(x)
        ).fillna(0).astype(int)

        history = history.merge(
            basho_records[['wrestler_id', 'bashoId', 'win_rate_last_1_basho',
                          'win_rate_last_2_basho', 'win_rate_last_3_basho',
                          'kachikoshi_streak', 'makekoshi_streak']],
            on=['wrestler_id', 'bashoId'],
            how='left'
        )

        return history

    def _compute_style_features(self, history):
        """Compute style features at multiple time windows."""
        history = history.sort_values(['wrestler_id', 'bashoId', 'day']).copy()
        history_known = history[history['kimarite_known']].copy()

        # Create indicators
        for cat in ['push', 'grapple', 'evasion']:
            history_known[f'is_{cat}_win'] = (
                (history_known['kimarite_category'] == cat) & (history_known['won'] == 1)
            ).astype(int)

        for bucket in KIMARITE_BUCKETS.keys():
            history_known[f'is_{bucket}_win'] = (
                (history_known['kimarite_bucket'] == bucket) & (history_known['won'] == 1)
            ).astype(int)

        # Basho-level aggregation
        agg_dict = {'won': 'sum'}
        for cat in ['push', 'grapple', 'evasion']:
            agg_dict[f'is_{cat}_win'] = 'sum'
        for bucket in KIMARITE_BUCKETS.keys():
            agg_dict[f'is_{bucket}_win'] = 'sum'

        basho_style = history_known.groupby(['wrestler_id', 'bashoId']).agg(agg_dict).reset_index()
        basho_style = basho_style.sort_values(['wrestler_id', 'bashoId'])

        # Compute percentages at different windows
        all_categories = ['push', 'grapple', 'evasion'] + list(KIMARITE_BUCKETS.keys())

        for cat in all_categories:
            col = f'is_{cat}_win'
            # Career (expanding)
            basho_style[f'pct_wins_by_{cat}_career'] = basho_style.groupby('wrestler_id').apply(
                lambda g: g[col].shift(1).expanding().sum() / g['won'].shift(1).expanding().sum().replace(0, np.nan)
            ).reset_index(level=0, drop=True)

            # Rolling windows
            for window in STYLE_WINDOWS:
                basho_style[f'pct_wins_by_{cat}_last_{window}_basho'] = basho_style.groupby('wrestler_id').apply(
                    lambda g: g[col].shift(1).rolling(window=window, min_periods=1).sum() /
                              g['won'].shift(1).rolling(window=window, min_periods=1).sum().replace(0, np.nan)
                ).reset_index(level=0, drop=True)

        # Style drift
        for cat in ['push', 'grapple', 'evasion']:
            basho_style[f'style_drift_{cat}'] = (
                basho_style[f'pct_wins_by_{cat}_last_3_basho'] - basho_style[f'pct_wins_by_{cat}_career']
            )

        # Select columns to merge
        style_cols = ['wrestler_id', 'bashoId']
        for cat in ['push', 'grapple', 'evasion']:
            style_cols.extend([
                f'pct_wins_by_{cat}_last_1_basho', f'pct_wins_by_{cat}_last_3_basho',
                f'pct_wins_by_{cat}_last_6_basho', f'pct_wins_by_{cat}_career', f'style_drift_{cat}'
            ])
        for bucket in KIMARITE_BUCKETS.keys():
            style_cols.extend([
                f'pct_wins_by_{bucket}_last_1_basho', f'pct_wins_by_{bucket}_last_3_basho',
                f'pct_wins_by_{bucket}_last_6_basho', f'pct_wins_by_{bucket}_career'
            ])

        history = history.merge(
            basho_style[[c for c in style_cols if c in basho_style.columns]],
            on=['wrestler_id', 'bashoId'],
            how='left'
        )

        return history

    def _compute_h2h_features(self, df, history):
        """Compute head-to-head features."""
        h2h = history[['bout_id', 'wrestler_id', 'opponent_id', 'won', 'bashoId', 'day', 'kimarite_clean']].copy()
        h2h = h2h.sort_values(['wrestler_id', 'opponent_id', 'bashoId', 'day'])

        h2h['h2h_total_bouts'] = h2h.groupby(['wrestler_id', 'opponent_id']).cumcount()
        h2h['h2h_wins'] = h2h.groupby(['wrestler_id', 'opponent_id'])['won'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0).astype(int)
        h2h['h2h_losses'] = h2h['h2h_total_bouts'] - h2h['h2h_wins']
        h2h['h2h_win_rate'] = np.where(
            h2h['h2h_total_bouts'] > 0,
            h2h['h2h_wins'] / h2h['h2h_total_bouts'],
            np.nan
        )
        h2h['h2h_never_met'] = (h2h['h2h_total_bouts'] == 0).astype(int)
        h2h['h2h_last_result'] = h2h.groupby(['wrestler_id', 'opponent_id'])['won'].shift(1)

        return h2h[['bout_id', 'wrestler_id', 'h2h_total_bouts', 'h2h_wins', 'h2h_losses',
                    'h2h_win_rate', 'h2h_never_met', 'h2h_last_result']].copy()

    def _compute_origin_features(self, df):
        """Compute origin features."""
        df = df.copy()

        foreign_keywords = {
            'Mongolia': ['mongolia', 'ulaanbaatar'],
            'Georgia': ['georgia', 'tbilisi'],
            'Bulgaria': ['bulgaria', 'sofia'],
            'Brazil': ['brazil', 'são paulo', 'rio'],
            'USA': ['usa', 'hawaii', 'california', 'texas'],
            'Russia': ['russia', 'moscow'],
            'China': ['china', 'beijing'],
            'Estonia': ['estonia'],
            'Egypt': ['egypt'],
        }

        for prefix, id_col in [('east', 'eastId'), ('west', 'westId')]:
            if self.rikishi is None:
                df[f'{prefix}_is_japanese'] = np.nan
                df[f'{prefix}_country_of_origin'] = None
                continue

            shusshin_map = {}
            for _, row in self.rikishi.iterrows():
                wrestler_id = row.get('id')
                shusshin = row.get('shusshin', row.get('birthplace', ''))
                if wrestler_id and shusshin:
                    shusshin_map[wrestler_id] = str(shusshin)

            df[f'{prefix}_shusshin'] = df[id_col].map(shusshin_map)

            def infer_country(shusshin):
                if pd.isna(shusshin):
                    return None
                s = str(shusshin).lower()
                for country, keywords in foreign_keywords.items():
                    for kw in keywords:
                        if kw in s:
                            return country
                return 'Japan'

            df[f'{prefix}_country_of_origin'] = df[f'{prefix}_shusshin'].apply(infer_country)
            df[f'{prefix}_is_japanese'] = (df[f'{prefix}_country_of_origin'] == 'Japan').astype(int)
            df = df.drop(columns=[f'{prefix}_shusshin'])

        return df

    def compute_all_features(self):
        """Main entry point for feature engineering."""
        print("Starting vectorized feature engineering...")

        print("  Preparing base data...")
        df = self._prepare_base_data()

        print("  Building wrestler bout history...")
        history = self._build_wrestler_history(df)

        print("  Computing career stats...")
        history = self._compute_career_stats(history)

        print("  Computing basho stats...")
        history = self._compute_basho_stats(history)

        print("  Computing streak stats...")
        history = self._compute_streak_stats(history)

        print("  Computing recent form...")
        history = self._compute_recent_form(history)

        print("  Computing basho form...")
        history = self._compute_basho_form(history)

        print("  Computing style features (windowed)...")
        history = self._compute_style_features(history)

        print("  Computing head-to-head features...")
        h2h_features = self._compute_h2h_features(df, history)

        # Separate east and west
        print("  Merging features to bout dataframe...")
        east_history = history[history['position'] == 'east'].copy()
        west_history = history[history['position'] == 'west'].copy()

        exclude_cols = ['bout_id', 'wrestler_id', 'opponent_id', 'bashoId', 'position',
                       'kimarite_clean', 'kimarite_known', 'kimarite_category',
                       'kimarite_bucket', 'kimarite_grip', 'winnerId']

        east_cols = {c: f'east_{c}' for c in east_history.columns if c not in exclude_cols}
        west_cols = {c: f'west_{c}' for c in west_history.columns if c not in exclude_cols}

        east_history = east_history.rename(columns=east_cols)
        west_history = west_history.rename(columns=west_cols)

        df = df.merge(east_history[['bout_id'] + list(east_cols.values())], on='bout_id', how='left')
        df = df.merge(west_history[['bout_id'] + list(west_cols.values())], on='bout_id', how='left')

        # H2H for east
        east_h2h = h2h_features[h2h_features['wrestler_id'].isin(df['eastId'].unique())].copy()
        east_h2h = east_h2h.rename(columns={c: f'east_{c}' for c in east_h2h.columns if c not in ['bout_id', 'wrestler_id']})
        df = df.merge(east_h2h.drop(columns=['wrestler_id']), on='bout_id', how='left')

        # Pressure features
        print("  Computing pressure features...")
        for prefix in ['east', 'west']:
            wins_col = f'{prefix}_basho_wins'
            losses_col = f'{prefix}_basho_losses'
            rank_col = f'{prefix}_rank_numeric'

            if wins_col in df.columns:
                df[f'{prefix}_needs_one_win_for_kachikoshi'] = (df[wins_col] == 7).astype(int)
                df[f'{prefix}_needs_two_wins_for_kachikoshi'] = (df[wins_col] == 6).astype(int)
                df[f'{prefix}_already_kachikoshi'] = (df[wins_col] >= 8).astype(int)
                df[f'{prefix}_already_makekoshi'] = (df[losses_col] >= 8).astype(int)
                df[f'{prefix}_is_day_15'] = (df['day'] == 15).astype(int)
                df[f'{prefix}_day_times_needs_one_for_kachikoshi'] = np.where(df[wins_col] == 7, df['day'], 0)

                if rank_col in df.columns:
                    df[f'{prefix}_is_ozeki'] = ((df[rank_col] >= 10) & (df[rank_col] < 30)).astype(int)
                    df[f'{prefix}_is_yokozuna'] = (df[rank_col] < 10).astype(int)
                    df[f'{prefix}_yokozuna_losing_record_so_far'] = (
                        (df[rank_col] < 10) & (df[losses_col] > df[wins_col])
                    ).astype(int)

        # Origin features
        print("  Computing origin features...")
        df = self._compute_origin_features(df)

        # Rating features
        print("  Adding rating features...")
        for prefix in ['east', 'west']:
            rank_col = f'{prefix}_rank_numeric'
            elo_col = f'{prefix}_elo'
            glicko_col = f'{prefix}_glicko_rating'

            if rank_col in df.columns and elo_col in df.columns:
                df[f'{prefix}_expected_elo'] = df[rank_col].map(
                    lambda r: self.rank_avg_lookup.get(r, {}).get('avg_elo', 1500)
                )
                df[f'{prefix}_elo_minus_expected'] = df[elo_col] - df[f'{prefix}_expected_elo']

            if rank_col in df.columns and glicko_col in df.columns:
                df[f'{prefix}_expected_glicko'] = df[rank_col].map(
                    lambda r: self.rank_avg_lookup.get(r, {}).get('avg_glicko', 1500)
                )
                df[f'{prefix}_glicko_minus_expected'] = df[glicko_col] - df[f'{prefix}_expected_glicko']

        # Pairwise differentials
        print("  Computing pairwise differentials...")
        df['rank_diff'] = df['west_rank_numeric'] - df['east_rank_numeric']
        df['career_win_rate_diff'] = df['east_career_win_rate'] - df['west_career_win_rate']

        if 'east_elo' in df.columns and 'west_elo' in df.columns:
            df['elo_diff'] = df['east_elo'] - df['west_elo']
        if 'east_glicko_rating' in df.columns and 'west_glicko_rating' in df.columns:
            df['glicko_rating_diff'] = df['east_glicko_rating'] - df['west_glicko_rating']

        # Clean up duplicates from merges
        print("  Cleaning up...")
        for col in list(df.columns):
            if col.endswith('_x'):
                base_col = col[:-2]
                y_col = base_col + '_y'
                if y_col in df.columns:
                    df[base_col] = df[col].fillna(df[y_col])
                    df = df.drop(columns=[col, y_col])

        # Drop zero-importance features
        cols_to_drop = [c for c in df.columns if c in ZERO_IMPORTANCE_FEATURES]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Drop utility columns
        drop_cols = ['kimarite_clean', 'kimarite_known', 'kimarite_bucket', 'kimarite_grip', 'bout_date_approx']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        print(f"Feature engineering complete. Generated {len(df):,} rows with {len(df.columns)} columns.")
        return df
