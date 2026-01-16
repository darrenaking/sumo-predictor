"""
Rating system implementations: ELO and Glicko-2

These ratings are computed from historical bout data and become features
for the prediction model.
"""

import math
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# ELO Rating System
# =============================================================================

@dataclass
class EloRating:
    """ELO rating for a single wrestler."""
    rating: float = 1500.0
    games_played: int = 0


class EloSystem:
    """
    Standard ELO rating system.

    Parameters:
    - k_factor: How much ratings change per game (default 32)
    - initial_rating: Starting rating for new players (default 1500)
    """

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[int, EloRating] = defaultdict(
            lambda: EloRating(rating=self.initial_rating)
        )

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, winner_id: int, loser_id: int) -> Tuple[float, float]:
        """
        Update ratings after a match.

        Returns the new ratings for (winner, loser).
        """
        winner = self.ratings[winner_id]
        loser = self.ratings[loser_id]

        # Expected scores
        expected_winner = self.expected_score(winner.rating, loser.rating)
        expected_loser = 1.0 - expected_winner

        # Update ratings
        winner.rating += self.k_factor * (1.0 - expected_winner)
        loser.rating += self.k_factor * (0.0 - expected_loser)

        # Update games played
        winner.games_played += 1
        loser.games_played += 1

        return winner.rating, loser.rating

    def get_rating(self, wrestler_id: int) -> float:
        """Get current rating for a wrestler."""
        return self.ratings[wrestler_id].rating

    def get_ratings_at_time(self, wrestler_id: int) -> Tuple[float, int]:
        """Get rating and games played for a wrestler."""
        r = self.ratings[wrestler_id]
        return r.rating, r.games_played


# =============================================================================
# Glicko-2 Rating System
# =============================================================================

@dataclass
class Glicko2Rating:
    """
    Glicko-2 rating with uncertainty measures.

    - mu: Rating on Glicko-2 scale (converted from/to standard scale)
    - phi: Rating deviation (uncertainty)
    - sigma: Volatility (consistency of performance)
    """
    mu: float = 0.0  # Internal scale, 0 = 1500 on standard scale
    phi: float = 350.0 / 173.7178  # Initial RD
    sigma: float = 0.06  # Initial volatility

    @property
    def rating(self) -> float:
        """Convert to standard rating scale (centered at 1500)."""
        return self.mu * 173.7178 + 1500.0

    @property
    def rd(self) -> float:
        """Rating deviation on standard scale."""
        return self.phi * 173.7178

    @classmethod
    def from_standard(cls, rating: float = 1500.0, rd: float = 350.0,
                     sigma: float = 0.06) -> 'Glicko2Rating':
        """Create from standard rating scale."""
        return cls(
            mu=(rating - 1500.0) / 173.7178,
            phi=rd / 173.7178,
            sigma=sigma
        )


class Glicko2System:
    """
    Glicko-2 rating system implementation.

    Key features vs ELO:
    - Tracks rating deviation (uncertainty)
    - Tracks volatility (consistency)
    - RD increases during inactivity

    Parameters:
    - tau: System constant controlling volatility change (0.3-1.2 typical)
    - initial_rating: Starting rating (default 1500)
    - initial_rd: Starting rating deviation (default 350)
    - initial_volatility: Starting volatility (default 0.06)
    - rd_increase_per_inactive_period: How much RD increases per missed basho
    """

    def __init__(self, tau: float = 0.5, initial_rating: float = 1500.0,
                 initial_rd: float = 350.0, initial_volatility: float = 0.06,
                 rd_increase_per_inactive_period: float = 30.0):
        self.tau = tau
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_volatility = initial_volatility
        self.rd_increase_per_inactive_period = rd_increase_per_inactive_period

        self.ratings: Dict[int, Glicko2Rating] = defaultdict(
            lambda: Glicko2Rating.from_standard(
                self.initial_rating, self.initial_rd, self.initial_volatility
            )
        )

        # Track last active basho for each wrestler
        self.last_active_basho: Dict[int, str] = {}

    def _g(self, phi: float) -> float:
        """g function from Glicko-2 paper."""
        return 1.0 / math.sqrt(1.0 + 3.0 * phi ** 2 / math.pi ** 2)

    def _E(self, mu: float, mu_j: float, phi_j: float) -> float:
        """Expected score function."""
        return 1.0 / (1.0 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def _compute_variance(self, mu: float, opponents: list) -> float:
        """Compute the variance (v) from Glicko-2 Step 3."""
        v_inv = 0.0
        for mu_j, phi_j, _ in opponents:
            g = self._g(phi_j)
            E = self._E(mu, mu_j, phi_j)
            v_inv += g ** 2 * E * (1 - E)
        return 1.0 / v_inv if v_inv > 0 else float('inf')

    def _compute_delta(self, mu: float, v: float, opponents: list) -> float:
        """Compute the delta (improvement) from Glicko-2 Step 4."""
        delta = 0.0
        for mu_j, phi_j, score in opponents:
            g = self._g(phi_j)
            E = self._E(mu, mu_j, phi_j)
            delta += g * (score - E)
        return v * delta

    def _compute_new_volatility(self, sigma: float, phi: float,
                                 v: float, delta: float) -> float:
        """Compute new volatility using iterative algorithm (Step 5)."""
        a = math.log(sigma ** 2)

        def f(x):
            ex = math.exp(x)
            num1 = ex * (delta ** 2 - phi ** 2 - v - ex)
            denom1 = 2 * (phi ** 2 + v + ex) ** 2
            return num1 / denom1 - (x - a) / self.tau ** 2

        # Find initial bounds
        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        # Iterative algorithm
        epsilon = 0.000001
        fA = f(A)
        fB = f(B)

        while abs(B - A) > epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)

            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2

            B = C
            fB = fC

        return math.exp(A / 2)

    def update_for_basho(self, wrestler_id: int, basho_id: str,
                         opponents_results: list) -> Tuple[float, float, float]:
        """
        Update rating for a wrestler after a tournament.

        opponents_results: List of (opponent_id, score) where score is 1 for win, 0 for loss

        Returns: (new_rating, new_rd, new_volatility) on standard scale
        """
        player = self.ratings[wrestler_id]

        # Handle inactivity - increase RD
        if wrestler_id in self.last_active_basho:
            inactive_periods = self._count_inactive_periods(
                self.last_active_basho[wrestler_id], basho_id
            )
            if inactive_periods > 0:
                # Increase phi (RD) for inactivity
                phi_increase = (self.rd_increase_per_inactive_period * inactive_periods) / 173.7178
                player.phi = min(math.sqrt(player.phi ** 2 + phi_increase ** 2),
                                self.initial_rd / 173.7178)

        self.last_active_basho[wrestler_id] = basho_id

        if not opponents_results:
            # No matches this basho - just return current (with increased RD)
            return player.rating, player.rd, player.sigma

        # Build opponent list
        opponents = []
        for opp_id, score in opponents_results:
            opp = self.ratings[opp_id]
            opponents.append((opp.mu, opp.phi, score))

        # Step 3: Compute variance
        v = self._compute_variance(player.mu, opponents)

        # Step 4: Compute delta
        delta = self._compute_delta(player.mu, v, opponents)

        # Step 5: New volatility
        new_sigma = self._compute_new_volatility(player.sigma, player.phi, v, delta)

        # Step 6: Update phi (pre-rating period)
        phi_star = math.sqrt(player.phi ** 2 + new_sigma ** 2)

        # Step 7: Update mu and phi
        new_phi = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)

        new_mu = player.mu
        for mu_j, phi_j, score in opponents:
            new_mu += new_phi ** 2 * self._g(phi_j) * (score - self._E(player.mu, mu_j, phi_j))

        # Update player
        player.mu = new_mu
        player.phi = new_phi
        player.sigma = new_sigma

        return player.rating, player.rd, player.sigma

    def update_single_bout(self, winner_id: int, loser_id: int) -> None:
        """
        Simple update for a single bout (simplified, not full Glicko-2).

        For proper Glicko-2, use update_for_basho with all results at once.
        This is a simplified version for bout-by-bout processing.
        """
        winner = self.ratings[winner_id]
        loser = self.ratings[loser_id]

        # Simplified update using ELO-like calculation but preserving RD/volatility
        g_w = self._g(winner.phi)
        g_l = self._g(loser.phi)

        E_w = self._E(winner.mu, loser.mu, loser.phi)
        E_l = self._E(loser.mu, winner.mu, winner.phi)

        # Update ratings
        winner.mu += 0.5 * g_l * (1.0 - E_w)
        loser.mu += 0.5 * g_w * (0.0 - E_l)

        # Reduce RD slightly after each bout
        winner.phi = max(winner.phi * 0.98, 30.0 / 173.7178)
        loser.phi = max(loser.phi * 0.98, 30.0 / 173.7178)

    def _count_inactive_periods(self, last_basho: str, current_basho: str) -> int:
        """Count number of basho missed between two basho IDs."""
        # basho_id format: YYYYMM
        last_year = int(last_basho[:4])
        last_month = int(last_basho[4:6])
        curr_year = int(current_basho[:4])
        curr_month = int(current_basho[4:6])

        basho_months = [1, 3, 5, 7, 9, 11]

        # Count basho between
        count = 0
        year, month = last_year, last_month

        while True:
            # Move to next basho
            idx = basho_months.index(month) if month in basho_months else 0
            idx += 1
            if idx >= len(basho_months):
                idx = 0
                year += 1
            month = basho_months[idx]

            if year > curr_year or (year == curr_year and month >= curr_month):
                break

            count += 1

            if count > 100:  # Safety limit
                break

        return max(0, count - 1)  # -1 because we don't count the current basho

    def get_rating(self, wrestler_id: int) -> Tuple[float, float, float]:
        """Get (rating, rd, volatility) for a wrestler."""
        r = self.ratings[wrestler_id]
        return r.rating, r.rd, r.sigma


# =============================================================================
# Utility Functions for Computing Ratings
# =============================================================================

def compute_elo_ratings(matches_df: pd.DataFrame,
                        k_factor: float = 32.0) -> pd.DataFrame:
    """
    Compute ELO ratings for all bouts chronologically.

    Adds columns: east_elo_before, west_elo_before, east_elo_after, west_elo_after

    IMPORTANT: Ratings are computed using only information available BEFORE each bout.
    """
    # Sort matches chronologically
    df = matches_df.copy()
    df = df.sort_values(['bashoId', 'day']).reset_index(drop=True)

    elo = EloSystem(k_factor=k_factor)

    east_elo_before = []
    west_elo_before = []
    east_elo_after = []
    west_elo_after = []

    for idx, row in df.iterrows():
        east_id = row['eastId']
        west_id = row['westId']
        winner_id = row.get('winnerId')

        # Get ratings BEFORE the bout
        east_before = elo.get_rating(east_id)
        west_before = elo.get_rating(west_id)

        east_elo_before.append(east_before)
        west_elo_before.append(west_before)

        # Update ratings if we know the winner
        if winner_id and (winner_id == east_id or winner_id == west_id):
            if winner_id == east_id:
                elo.update(east_id, west_id)
            else:
                elo.update(west_id, east_id)

        # Get ratings AFTER the bout
        east_elo_after.append(elo.get_rating(east_id))
        west_elo_after.append(elo.get_rating(west_id))

    df['east_elo_before'] = east_elo_before
    df['west_elo_before'] = west_elo_before
    df['east_elo_after'] = east_elo_after
    df['west_elo_after'] = west_elo_after

    return df


def compute_glicko2_ratings(matches_df: pd.DataFrame,
                            tau: float = 0.5) -> pd.DataFrame:
    """
    Compute Glicko-2 ratings for all bouts chronologically.

    Adds columns: east_glicko_rating_before, east_glicko_rd_before, east_glicko_vol_before,
                  west_glicko_rating_before, west_glicko_rd_before, west_glicko_vol_before,
                  and corresponding _after columns
    """
    # Sort matches chronologically
    df = matches_df.copy()
    df = df.sort_values(['bashoId', 'day']).reset_index(drop=True)

    glicko = Glicko2System(tau=tau)

    # Initialize result columns
    columns = ['east_glicko_rating', 'east_glicko_rd', 'east_glicko_vol',
               'west_glicko_rating', 'west_glicko_rd', 'west_glicko_vol']

    results_before = {col: [] for col in columns}
    results_after = {col: [] for col in columns}

    for idx, row in df.iterrows():
        east_id = row['eastId']
        west_id = row['westId']
        winner_id = row.get('winnerId')

        # Get ratings BEFORE
        e_rating, e_rd, e_vol = glicko.get_rating(east_id)
        w_rating, w_rd, w_vol = glicko.get_rating(west_id)

        results_before['east_glicko_rating'].append(e_rating)
        results_before['east_glicko_rd'].append(e_rd)
        results_before['east_glicko_vol'].append(e_vol)
        results_before['west_glicko_rating'].append(w_rating)
        results_before['west_glicko_rd'].append(w_rd)
        results_before['west_glicko_vol'].append(w_vol)

        # Update if winner known
        if winner_id and (winner_id == east_id or winner_id == west_id):
            if winner_id == east_id:
                glicko.update_single_bout(east_id, west_id)
            else:
                glicko.update_single_bout(west_id, east_id)

        # Get ratings AFTER
        e_rating, e_rd, e_vol = glicko.get_rating(east_id)
        w_rating, w_rd, w_vol = glicko.get_rating(west_id)

        results_after['east_glicko_rating'].append(e_rating)
        results_after['east_glicko_rd'].append(e_rd)
        results_after['east_glicko_vol'].append(e_vol)
        results_after['west_glicko_rating'].append(w_rating)
        results_after['west_glicko_rd'].append(w_rd)
        results_after['west_glicko_vol'].append(w_vol)

    # Add columns to dataframe
    for col in columns:
        df[f'{col}_before'] = results_before[col]
        df[f'{col}_after'] = results_after[col]

    return df


def compute_all_ratings(matches_df: pd.DataFrame,
                        elo_k: float = 32.0,
                        glicko_tau: float = 0.5) -> pd.DataFrame:
    """
    Compute both ELO and Glicko-2 ratings for all matches.
    """
    print("Computing ELO ratings...")
    df = compute_elo_ratings(matches_df, k_factor=elo_k)

    print("Computing Glicko-2 ratings...")
    df = compute_glicko2_ratings(df, tau=glicko_tau)

    return df
