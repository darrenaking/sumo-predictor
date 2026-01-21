#!/usr/bin/env python3
"""
Update features.parquet with latest basho days.

This script:
1. Fetches new match data from sumo-api.com
2. Appends to existing matches.parquet
3. Recomputes ratings from scratch
4. Regenerates features for new days

Usage:
    python scripts/update_features.py --basho 202601 --through-day 11
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List
import math

from src.data_collection import (
    fetch_basho_torikumi,
    parse_banzuke_rank,
)


# Paths
KAGGLE_01 = PROJECT_ROOT / "kaggle-output" / "01"
KAGGLE_02 = PROJECT_ROOT / "kaggle-output" / "02"
KAGGLE_03 = PROJECT_ROOT / "kaggle-output" / "03-v6"

# Kimarite categories
KIMARITE_CATEGORIES = {
    "push": [
        "oshidashi", "tsukidashi", "oshitaoshi", "tsukiotoshi",
        "tsukitaoshi", "okuridashi", "abisetaoshi"
    ],
    "grapple": [
        "yorikiri", "uwatenage", "shitatenage", "sukuinage", "kotenage",
        "kubinage", "yoritaoshi", "uwatedashinage", "shitatedashinage",
        "kakenage", "kirikaeshi", "tsukaminage", "tsuridashi", "tsuriotoshi",
        "utchari", "sotogake", "uchigake", "kimedashi", "kimekiri",
        "katasukashi", "okurinage", "okuritaoshi", "okurihineri",
        "okuritsuridashi", "amiuchi", "sabaori", "waridashi", "makiotoshi",
        "uwatehineri", "shitatehineri"
    ],
    "evasion": [
        "hatakikomi", "hikiotoshi", "hikkake", "ketaguri", "kekaeshi",
        "ashitori", "tsumadori", "chongake", "kawazugake", "komatasukui",
        "tottari", "izori", "shumokuzori", "tasukizori", "nichonage"
    ]
}


def categorize_kimarite(kimarite: str) -> str:
    """Categorize a kimarite into push/grapple/evasion."""
    if not kimarite or pd.isna(kimarite):
        return "unknown"
    kimarite_lower = str(kimarite).lower().strip()
    for category, techniques in KIMARITE_CATEGORIES.items():
        if kimarite_lower in techniques:
            return category
    return "grapple"


def fetch_basho_matches(basho_id: str, through_day: int, division: str = "Makuuchi") -> pd.DataFrame:
    """Fetch all matches for a basho through the given day."""
    all_bouts = []

    for day in range(1, through_day + 1):
        print(f"  Fetching {basho_id} day {day}...")
        data = fetch_basho_torikumi(basho_id, division, day)

        if data and 'torikumi' in data:
            for bout in data['torikumi']:
                bout_record = {
                    'bashoId': basho_id,
                    'division': division,
                    'day': day,
                    'matchNo': bout.get('matchNo', 0),
                    'eastId': bout.get('eastId'),
                    'eastShikona': bout.get('eastShikona'),
                    'eastRank': bout.get('eastRank'),
                    'westId': bout.get('westId'),
                    'westShikona': bout.get('westShikona'),
                    'westRank': bout.get('westRank'),
                    'kimarite': bout.get('kimarite', ''),
                    'winnerId': bout.get('winnerId') if bout.get('winnerId') else None,
                    'winnerEn': bout.get('winnerEn', ''),
                    'winnerJp': bout.get('winnerJp', ''),
                }
                all_bouts.append(bout_record)

    if not all_bouts:
        return pd.DataFrame()

    df = pd.DataFrame(all_bouts)

    # Create bout_id for deduplication
    def create_bout_id(row):
        ids = sorted([str(row.get('eastId', '')), str(row.get('westId', ''))])
        return f"{row.get('bashoId', '')}_{row.get('day', '')}_{ids[0]}_{ids[1]}"

    df['bout_id'] = df.apply(create_bout_id, axis=1)
    df = df.drop_duplicates(subset=['bout_id'])

    # Parse basho date
    df['basho_year'] = df['bashoId'].astype(str).str[:4].astype(int)
    df['basho_month'] = df['bashoId'].astype(str).str[4:6].astype(int)

    # Parse ranks to numeric
    df['east_rank_numeric'] = df['eastRank'].apply(parse_banzuke_rank)
    df['west_rank_numeric'] = df['westRank'].apply(parse_banzuke_rank)

    # Categorize kimarite
    df['kimarite_category'] = df['kimarite'].apply(categorize_kimarite)

    # Create winner columns
    df['east_won'] = (df['winnerId'] == df['eastId']).astype(int)
    df['west_won'] = (df['winnerId'] == df['westId']).astype(int)

    return df


# ELO System
@dataclass
class EloRating:
    rating: float = 1500.0
    games_played: int = 0


class EloSystem:
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[int, EloRating] = defaultdict(
            lambda: EloRating(rating=self.initial_rating)
        )

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_id: int, loser_id: int) -> Tuple[float, float]:
        winner = self.ratings[winner_id]
        loser = self.ratings[loser_id]
        expected_winner = self.expected_score(winner.rating, loser.rating)
        expected_loser = 1.0 - expected_winner
        winner.rating += self.k_factor * (1.0 - expected_winner)
        loser.rating += self.k_factor * (0.0 - expected_loser)
        winner.games_played += 1
        loser.games_played += 1
        return winner.rating, loser.rating

    def get_rating(self, wrestler_id: int) -> float:
        return self.ratings[wrestler_id].rating


# Glicko-2 System
@dataclass
class Glicko2Rating:
    mu: float = 0.0
    phi: float = 350.0 / 173.7178
    sigma: float = 0.06

    @property
    def rating(self) -> float:
        return self.mu * 173.7178 + 1500.0

    @property
    def rd(self) -> float:
        return self.phi * 173.7178

    @classmethod
    def from_standard(cls, rating: float = 1500.0, rd: float = 350.0,
                     sigma: float = 0.06) -> 'Glicko2Rating':
        return cls(
            mu=(rating - 1500.0) / 173.7178,
            phi=rd / 173.7178,
            sigma=sigma
        )


class Glicko2System:
    def __init__(self, tau: float = 0.5, initial_rating: float = 1500.0,
                 initial_rd: float = 350.0, initial_volatility: float = 0.06):
        self.tau = tau
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_volatility = initial_volatility
        self.ratings: Dict[int, Glicko2Rating] = defaultdict(
            lambda: Glicko2Rating.from_standard(
                self.initial_rating, self.initial_rd, self.initial_volatility
            )
        )

    def _g(self, phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)

    def _E(self, mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def update_single_bout(self, winner_id: int, loser_id: int) -> None:
        winner = self.ratings[winner_id]
        loser = self.ratings[loser_id]
        g_w = self._g(winner.phi)
        g_l = self._g(loser.phi)
        E_w = self._E(winner.mu, loser.mu, loser.phi)
        E_l = self._E(loser.mu, winner.mu, winner.phi)
        winner.mu += 0.5 * g_l * (1.0 - E_w)
        loser.mu += 0.5 * g_w * (0.0 - E_l)
        winner.phi = max(winner.phi * 0.98, 30.0 / 173.7178)
        loser.phi = max(loser.phi * 0.98, 30.0 / 173.7178)

    def get_rating(self, wrestler_id: int) -> Tuple[float, float, float]:
        r = self.ratings[wrestler_id]
        return r.rating, r.rd, r.sigma


def compute_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ELO and Glicko-2 ratings for all bouts."""
    df = df.copy()
    df = df.sort_values(['bashoId', 'day']).reset_index(drop=True)

    elo = EloSystem(k_factor=32.0)
    glicko = Glicko2System(tau=0.5)

    n = len(df)
    east_elo = np.zeros(n)
    west_elo = np.zeros(n)
    east_glicko_rating = np.zeros(n)
    east_glicko_rd = np.zeros(n)
    east_glicko_vol = np.zeros(n)
    west_glicko_rating = np.zeros(n)
    west_glicko_rd = np.zeros(n)
    west_glicko_vol = np.zeros(n)

    for idx, row in df.iterrows():
        if idx % 50000 == 0:
            print(f"  Computing ratings: {idx:,} / {n:,}")

        east_id = row['eastId']
        west_id = row['westId']
        winner_id = row.get('winnerId')

        east_elo[idx] = elo.get_rating(east_id)
        west_elo[idx] = elo.get_rating(west_id)

        g_e = glicko.get_rating(east_id)
        g_w = glicko.get_rating(west_id)

        east_glicko_rating[idx] = g_e[0]
        east_glicko_rd[idx] = g_e[1]
        east_glicko_vol[idx] = g_e[2]
        west_glicko_rating[idx] = g_w[0]
        west_glicko_rd[idx] = g_w[1]
        west_glicko_vol[idx] = g_w[2]

        if winner_id and (winner_id == east_id or winner_id == west_id):
            if winner_id == east_id:
                elo.update(east_id, west_id)
                glicko.update_single_bout(east_id, west_id)
            else:
                elo.update(west_id, east_id)
                glicko.update_single_bout(west_id, east_id)

    df['east_elo'] = east_elo
    df['west_elo'] = west_elo
    df['east_glicko_rating'] = east_glicko_rating
    df['east_glicko_rd'] = east_glicko_rd
    df['east_glicko_vol'] = east_glicko_vol
    df['west_glicko_rating'] = west_glicko_rating
    df['west_glicko_rd'] = west_glicko_rd
    df['west_glicko_vol'] = west_glicko_vol
    df['elo_diff'] = df['east_elo'] - df['west_elo']
    df['glicko_rating_diff'] = df['east_glicko_rating'] - df['west_glicko_rating']
    df['glicko_rd_diff'] = df['east_glicko_rd'] - df['west_glicko_rd']

    return df


def main():
    parser = argparse.ArgumentParser(description='Update features with latest basho data')
    parser.add_argument('--basho', required=True, help='Basho ID (YYYYMM)')
    parser.add_argument('--through-day', type=int, required=True, help='Update through this day')
    args = parser.parse_args()

    basho_id = args.basho
    through_day = args.through_day

    print(f"Updating features for {basho_id} through day {through_day}")

    # Step 1: Load existing matches
    print("\n1. Loading existing matches...")
    existing_matches = pd.read_parquet(KAGGLE_01 / "matches.parquet")
    print(f"   Existing matches: {len(existing_matches):,}")

    # Check what days we have for this basho
    basho_existing = existing_matches[existing_matches['bashoId'] == basho_id]
    existing_days = sorted(basho_existing['day'].unique()) if len(basho_existing) > 0 else []
    print(f"   Existing days for {basho_id}: {existing_days}")

    # Step 2: Fetch new data from API
    print(f"\n2. Fetching {basho_id} days 1-{through_day} from API...")
    new_basho_matches = fetch_basho_matches(basho_id, through_day)
    print(f"   Fetched {len(new_basho_matches)} bouts")

    # Step 3: Remove old basho data and add new
    print("\n3. Merging data...")
    other_matches = existing_matches[existing_matches['bashoId'] != basho_id]

    # Ensure columns match
    for col in existing_matches.columns:
        if col not in new_basho_matches.columns:
            new_basho_matches[col] = None

    # Keep only columns that exist in original
    new_basho_matches = new_basho_matches[[c for c in existing_matches.columns if c in new_basho_matches.columns]]

    updated_matches = pd.concat([other_matches, new_basho_matches], ignore_index=True)
    updated_matches = updated_matches.sort_values(['bashoId', 'day']).reset_index(drop=True)
    print(f"   Total matches after merge: {len(updated_matches):,}")

    # Save updated matches
    updated_matches.to_parquet(KAGGLE_01 / "matches.parquet", index=False)
    print(f"   Saved to {KAGGLE_01 / 'matches.parquet'}")

    # Step 4: Recompute ratings
    print("\n4. Recomputing ratings...")
    matches_with_ratings = compute_ratings(updated_matches)
    matches_with_ratings.to_parquet(KAGGLE_02 / "matches_with_ratings.parquet", index=False)
    print(f"   Saved to {KAGGLE_02 / 'matches_with_ratings.parquet'}")

    # Verify
    basho_check = matches_with_ratings[matches_with_ratings['bashoId'] == basho_id]
    print(f"   Days for {basho_id}: {sorted(basho_check['day'].unique())}")

    # Step 5: Regenerate features
    print("\n5. Regenerating features...")

    # Load rikishi and rank averages if available
    rikishi_df = None
    rank_averages_df = None

    rikishi_path = PROJECT_ROOT / "data" / "rikishi.parquet"
    if rikishi_path.exists():
        rikishi_df = pd.read_parquet(rikishi_path)
        print(f"   Loaded {len(rikishi_df)} rikishi profiles")

    rank_avg_path = KAGGLE_02 / "rank_averages.parquet"
    if rank_avg_path.exists():
        rank_averages_df = pd.read_parquet(rank_avg_path)
        print(f"   Loaded rank averages")

    # Import the vectorized feature engine
    from src.feature_engine import VectorizedFeatureEngine

    engine = VectorizedFeatureEngine(matches_with_ratings, rikishi_df, rank_averages_df)
    features_df = engine.compute_all_features()

    print(f"   Generated {len(features_df):,} feature rows with {len(features_df.columns)} columns")

    # Save features
    features_df.to_parquet(KAGGLE_03 / "features.parquet", index=False)
    print(f"   Saved to {KAGGLE_03 / 'features.parquet'}")

    # Verify
    basho_features = features_df[features_df['bashoId'] == basho_id]
    print(f"   Feature days for {basho_id}: {sorted(basho_features['day'].unique())}")

    print("\nDone!")


if __name__ == '__main__':
    main()
