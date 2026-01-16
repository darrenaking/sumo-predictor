"""
Feature engineering for sumo bout prediction.

CRITICAL: All features must be computed using ONLY information available BEFORE the bout.
No leakage from the bout itself or future bouts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import math


# =============================================================================
# Kimarite Categories
# =============================================================================

KIMARITE_PUSH = [
    "oshidashi", "tsukidashi", "oshitaoshi", "tsukiotoshi",
    "tsukitaoshi", "okuridashi", "abisetaoshi"
]

KIMARITE_GRAPPLE = [
    "yorikiri", "uwatenage", "shitatenage", "sukuinage", "kotenage",
    "kubinage", "yoritaoshi", "uwatedashinage", "shitatedashinage",
    "kakenage", "kirikaeshi", "tsukaminage", "tsuridashi", "tsuriotoshi",
    "utchari", "sotogake", "uchigake", "kimedashi", "kimekiri",
    "katasukashi", "okurinage", "okuritaoshi", "okurihineri",
    "okuritsuridashi", "amiuchi", "sabaori", "waridashi", "makiotoshi",
    "uwatehineri", "shitatehineri"
]

KIMARITE_EVASION = [
    "hatakikomi", "hikiotoshi", "hikkake", "ketaguri", "kekaeshi",
    "ashitori", "tsumadori", "chongake", "kawazugake", "komatasukui",
    "tottari", "izori", "shumokuzori", "tasukizori", "nichonage"
]


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
        return "grapple"  # Default


# =============================================================================
# Banzuke Rank Parsing
# =============================================================================

def parse_banzuke_rank(rank_str: str) -> int:
    """
    Convert banzuke rank string to numeric value.
    Lower number = higher rank.
    """
    import re

    if not rank_str or pd.isna(rank_str):
        return 999

    rank_str = str(rank_str).strip().upper()

    rank_bases = {
        "Y": 0,      # Yokozuna
        "O": 10,     # Ozeki
        "S": 30,     # Sekiwake
        "K": 40,     # Komusubi
        "M": 50,     # Maegashira
        "J": 100,    # Juryo
    }

    match = re.match(r"([YOSKM]|J)(\d+)?([EW])?", rank_str)

    if not match:
        return 999

    rank_letter = match.group(1)
    rank_num = int(match.group(2)) if match.group(2) else 1
    direction = match.group(3) if match.group(3) else "E"

    base = rank_bases.get(rank_letter, 999)
    numeric = base + (rank_num - 1) * 2

    if direction == "W":
        numeric += 1

    return numeric


# =============================================================================
# Rolling Statistics Calculator
# =============================================================================

class WrestlerStatsTracker:
    """
    Tracks rolling statistics for all wrestlers.

    Maintains state that can be queried at any point in time to get
    stats using only information available up to that point.
    """

    def __init__(self):
        # Career stats
        self.career_wins: Dict[int, int] = defaultdict(int)
        self.career_losses: Dict[int, int] = defaultdict(int)
        self.career_bouts: Dict[int, List[Dict]] = defaultdict(list)
        self.debut_date: Dict[int, str] = {}

        # Tournament stats (reset each basho)
        self.current_basho: str = ""
        self.basho_wins: Dict[int, int] = defaultdict(int)
        self.basho_losses: Dict[int, int] = defaultdict(int)

        # Streaks
        self.current_win_streak: Dict[int, int] = defaultdict(int)
        self.current_loss_streak: Dict[int, int] = defaultdict(int)

        # Completed basho records
        self.completed_basho_records: Dict[int, List[Tuple[str, int, int]]] = defaultdict(list)

        # Head to head
        self.h2h_wins: Dict[Tuple[int, int], int] = defaultdict(int)
        self.h2h_bouts: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

        # Kimarite tracking
        self.win_kimarite: Dict[int, List[str]] = defaultdict(list)
        self.loss_kimarite: Dict[int, List[str]] = defaultdict(list)

        # Duration tracking (if available)
        self.bout_durations: Dict[int, List[float]] = defaultdict(list)

        # Absence tracking
        self.last_basho_participated: Dict[int, str] = {}
        self.basho_absences: Dict[int, List[str]] = defaultdict(list)  # Full kyujo
        self.basho_withdrawals: Dict[int, List[str]] = defaultdict(list)  # Mid-tournament

        # Torinaoshi tracking
        self.torinaoshi_count: Dict[int, int] = defaultdict(int)
        self.torinaoshi_wins: Dict[int, int] = defaultdict(int)

    def new_basho(self, basho_id: str):
        """Called when a new tournament starts."""
        # Save completed basho records
        if self.current_basho:
            for wrestler_id in set(self.basho_wins.keys()) | set(self.basho_losses.keys()):
                wins = self.basho_wins.get(wrestler_id, 0)
                losses = self.basho_losses.get(wrestler_id, 0)
                if wins + losses > 0:
                    self.completed_basho_records[wrestler_id].append(
                        (self.current_basho, wins, losses)
                    )

        # Reset tournament stats
        self.current_basho = basho_id
        self.basho_wins.clear()
        self.basho_losses.clear()

    def record_bout(self, bout: Dict):
        """
        Record a bout result. Call this chronologically.

        bout should have: eastId, westId, winnerId, bashoId, day, kimarite, duration (optional)
        """
        east_id = bout['eastId']
        west_id = bout['westId']
        winner_id = bout.get('winnerId')
        basho_id = bout.get('bashoId', '')
        kimarite = bout.get('kimarite', '')
        duration = bout.get('duration')
        is_torinaoshi = bout.get('is_torinaoshi', False)

        # Check for new basho
        if basho_id and basho_id != self.current_basho:
            self.new_basho(basho_id)

        # Update debut dates
        if east_id not in self.debut_date:
            self.debut_date[east_id] = basho_id
        if west_id not in self.debut_date:
            self.debut_date[west_id] = basho_id

        # Update last participation
        self.last_basho_participated[east_id] = basho_id
        self.last_basho_participated[west_id] = basho_id

        # Record bout in history
        bout_record = {
            'basho': basho_id,
            'opponent': west_id,
            'won': winner_id == east_id if winner_id else None,
            'kimarite': kimarite,
            'duration': duration
        }
        self.career_bouts[east_id].append(bout_record)
        self.career_bouts[west_id].append({
            **bout_record,
            'opponent': east_id,
            'won': winner_id == west_id if winner_id else None
        })

        # Update H2H
        h2h_key = tuple(sorted([east_id, west_id]))
        self.h2h_bouts[h2h_key].append({
            'basho': basho_id,
            'winner': winner_id,
            'kimarite': kimarite,
            'duration': duration
        })

        # Update stats if winner known
        if winner_id:
            loser_id = west_id if winner_id == east_id else east_id

            # Career wins/losses
            self.career_wins[winner_id] += 1
            self.career_losses[loser_id] += 1

            # Basho wins/losses
            self.basho_wins[winner_id] += 1
            self.basho_losses[loser_id] += 1

            # Streaks
            self.current_win_streak[winner_id] += 1
            self.current_loss_streak[winner_id] = 0
            self.current_loss_streak[loser_id] += 1
            self.current_win_streak[loser_id] = 0

            # H2H wins
            if winner_id == east_id:
                self.h2h_wins[(east_id, west_id)] += 1
            else:
                self.h2h_wins[(west_id, east_id)] += 1

            # Kimarite tracking
            if kimarite:
                self.win_kimarite[winner_id].append(kimarite.lower())
                self.loss_kimarite[loser_id].append(kimarite.lower())

            # Torinaoshi tracking
            if is_torinaoshi:
                self.torinaoshi_count[east_id] += 1
                self.torinaoshi_count[west_id] += 1
                self.torinaoshi_wins[winner_id] += 1

        # Duration tracking
        if duration is not None:
            self.bout_durations[east_id].append(duration)
            self.bout_durations[west_id].append(duration)

    def get_career_stats(self, wrestler_id: int, bout_date: str = None) -> Dict:
        """Get career stats for a wrestler (as of a given date)."""
        total_wins = self.career_wins.get(wrestler_id, 0)
        total_losses = self.career_losses.get(wrestler_id, 0)
        total_bouts = total_wins + total_losses

        career_length = 0
        debut = self.debut_date.get(wrestler_id)
        if debut and bout_date:
            # Approximate days from basho IDs
            try:
                debut_year = int(debut[:4])
                debut_month = int(debut[4:6])
                bout_year = int(bout_date[:4])
                bout_month = int(bout_date[4:6])
                career_length = (bout_year - debut_year) * 365 + (bout_month - debut_month) * 30
            except (ValueError, TypeError):
                pass

        return {
            'career_wins': total_wins,
            'career_losses': total_losses,
            'career_total_bouts': total_bouts,
            'career_win_rate': total_wins / total_bouts if total_bouts > 0 else 0.5,
            'career_length_days': max(0, career_length),
            'career_total_tournaments': len(self.completed_basho_records.get(wrestler_id, []))
        }

    def get_recent_form(self, wrestler_id: int, n_bouts: int = 10) -> float:
        """Get win rate over last n bouts."""
        bouts = self.career_bouts.get(wrestler_id, [])

        if len(bouts) < n_bouts:
            return None

        recent = bouts[-n_bouts:]
        wins = sum(1 for b in recent if b.get('won') is True)
        return wins / n_bouts

    def get_basho_win_rates(self, wrestler_id: int, n_basho: int = 3) -> Optional[float]:
        """Get average win rate over last n completed tournaments."""
        records = self.completed_basho_records.get(wrestler_id, [])

        if len(records) < n_basho:
            return None

        recent = records[-n_basho:]
        total_wins = sum(r[1] for r in recent)
        total_bouts = sum(r[1] + r[2] for r in recent)

        return total_wins / total_bouts if total_bouts > 0 else None

    def get_kachikoshi_streak(self, wrestler_id: int) -> int:
        """Count consecutive tournaments with 8+ wins entering current basho."""
        records = self.completed_basho_records.get(wrestler_id, [])
        streak = 0
        for _, wins, losses in reversed(records):
            if wins >= 8:
                streak += 1
            else:
                break
        return streak

    def get_makekoshi_streak(self, wrestler_id: int) -> int:
        """Count consecutive tournaments with 7 or fewer wins entering current basho."""
        records = self.completed_basho_records.get(wrestler_id, [])
        streak = 0
        for _, wins, losses in reversed(records):
            if wins < 8 and (wins + losses) >= 8:  # Full tournament with losing record
                streak += 1
            else:
                break
        return streak

    def get_style_profile(self, wrestler_id: int) -> Dict:
        """Get kimarite style profile for a wrestler."""
        wins = self.win_kimarite.get(wrestler_id, [])
        losses = self.loss_kimarite.get(wrestler_id, [])

        def count_categories(kimarite_list):
            push = sum(1 for k in kimarite_list if k in KIMARITE_PUSH)
            grapple = sum(1 for k in kimarite_list if k in KIMARITE_GRAPPLE)
            evasion = sum(1 for k in kimarite_list if k in KIMARITE_EVASION)
            total = max(len(kimarite_list), 1)
            return push / total, grapple / total, evasion / total

        win_push, win_grapple, win_evasion = count_categories(wins)
        loss_push, loss_grapple, loss_evasion = count_categories(losses)

        # Modal kimarite
        from collections import Counter
        if wins:
            win_counts = Counter(wins)
            modal = win_counts.most_common(3)
            modal_kimarite = modal[0][0] if modal else None
            second_modal = modal[1][0] if len(modal) > 1 else None
            third_modal = modal[2][0] if len(modal) > 2 else None
            pct_modal = modal[0][1] / len(wins) if modal else 0

            # Entropy
            probs = [c / len(wins) for k, c in win_counts.items()]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
        else:
            modal_kimarite = None
            second_modal = None
            third_modal = None
            pct_modal = 0
            entropy = 0

        return {
            'pct_wins_by_push': win_push,
            'pct_wins_by_grapple': win_grapple,
            'pct_wins_by_evasion': win_evasion,
            'pct_losses_by_push': loss_push,
            'pct_losses_by_grapple': loss_grapple,
            'pct_losses_by_evasion': loss_evasion,
            'modal_kimarite': modal_kimarite,
            'second_modal_kimarite': second_modal,
            'third_modal_kimarite': third_modal,
            'pct_wins_by_modal_kimarite': pct_modal,
            'kimarite_entropy': entropy
        }

    def get_h2h_stats(self, wrestler_a: int, wrestler_b: int) -> Dict:
        """Get head-to-head statistics between two wrestlers."""
        h2h_key = tuple(sorted([wrestler_a, wrestler_b]))
        bouts = self.h2h_bouts.get(h2h_key, [])

        if not bouts:
            return {
                'h2h_total_bouts': 0,
                'h2h_wins': 0,
                'h2h_losses': 0,
                'h2h_win_rate': None,
                'h2h_never_met': True,
                'h2h_current_streak': 0,
                'h2h_last_result': None,
                'h2h_last_bout_kimarite': None
            }

        wins_a = sum(1 for b in bouts if b['winner'] == wrestler_a)
        losses_a = len(bouts) - wins_a

        # Current streak (positive = winning, negative = losing)
        streak = 0
        for b in reversed(bouts):
            if b['winner'] == wrestler_a:
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break

        last_bout = bouts[-1]
        last_result = 1 if last_bout['winner'] == wrestler_a else 0

        return {
            'h2h_total_bouts': len(bouts),
            'h2h_wins': wins_a,
            'h2h_losses': losses_a,
            'h2h_win_rate': wins_a / len(bouts),
            'h2h_never_met': False,
            'h2h_current_streak': streak,
            'h2h_last_result': last_result,
            'h2h_last_bout_kimarite': last_bout.get('kimarite')
        }

    def get_current_basho_stats(self, wrestler_id: int) -> Dict:
        """Get current tournament stats (before the current bout)."""
        return {
            'basho_wins': self.basho_wins.get(wrestler_id, 0),
            'basho_losses': self.basho_losses.get(wrestler_id, 0),
            'current_win_streak': self.current_win_streak.get(wrestler_id, 0),
            'current_loss_streak': self.current_loss_streak.get(wrestler_id, 0)
        }


# =============================================================================
# Feature Computation
# =============================================================================

def compute_physical_features(row: Dict, prefix: str = 'east') -> Dict:
    """Compute physical attribute features for a wrestler."""
    height = row.get(f'{prefix}Height', row.get(f'{prefix}_height'))
    weight = row.get(f'{prefix}Weight', row.get(f'{prefix}_weight'))
    birth_date = row.get(f'{prefix}Birthdate', row.get(f'{prefix}_birthdate'))
    bout_date = row.get('bashoId', '')

    features = {
        f'{prefix}_height_cm': height,
        f'{prefix}_weight_kg': weight,
    }

    # BMI
    if height and weight and height > 0:
        height_m = height / 100
        features[f'{prefix}_bmi'] = weight / (height_m ** 2)
    else:
        features[f'{prefix}_bmi'] = None

    # Age
    if birth_date and bout_date:
        try:
            # Approximate age from basho date
            bout_year = int(str(bout_date)[:4])
            bout_month = int(str(bout_date)[4:6])

            if isinstance(birth_date, str):
                birth_year = int(birth_date[:4])
                birth_month = int(birth_date[5:7]) if len(birth_date) > 5 else 1
            else:
                birth_year = birth_date.year
                birth_month = birth_date.month

            age = bout_year - birth_year + (bout_month - birth_month) / 12
            features[f'{prefix}_age_years'] = age
        except (ValueError, TypeError, AttributeError):
            features[f'{prefix}_age_years'] = None
    else:
        features[f'{prefix}_age_years'] = None

    return features


def compute_pressure_features(row: Dict, basho_stats: Dict, prefix: str = 'east') -> Dict:
    """Compute pressure situation features."""
    wins = basho_stats.get('basho_wins', 0)
    losses = basho_stats.get('basho_losses', 0)
    day = row.get('day', 1)
    rank = row.get(f'{prefix}Rank', row.get(f'{prefix}_rank', ''))

    # Parse rank
    rank_str = str(rank).upper() if rank else ''
    is_ozeki = rank_str.startswith('O')
    is_yokozuna = rank_str.startswith('Y')

    features = {
        f'{prefix}_needs_one_win_for_kachikoshi': int(wins == 7),
        f'{prefix}_needs_two_wins_for_kachikoshi': int(wins == 6),
        f'{prefix}_already_kachikoshi': int(wins >= 8),
        f'{prefix}_already_makekoshi': int(losses >= 8),
        f'{prefix}_is_day_15': int(day == 15),
        f'{prefix}_day_times_needs_one_for_kachikoshi': day if wins == 7 else 0,
        f'{prefix}_is_ozeki': int(is_ozeki),
        f'{prefix}_is_yokozuna': int(is_yokozuna),
        f'{prefix}_yokozuna_losing_record_so_far': int(is_yokozuna and losses > wins),
        f'{prefix}_yokozuna_multiple_losses_early': int(is_yokozuna and losses >= 2 and day <= 7),
    }

    return features


def compute_pairwise_features(east_features: Dict, west_features: Dict) -> Dict:
    """Compute pairwise differential features."""

    def safe_diff(a, b):
        if a is None or b is None:
            return None
        return a - b

    return {
        'weight_diff_kg': safe_diff(east_features.get('east_weight_kg'),
                                     west_features.get('west_weight_kg')),
        'height_diff_cm': safe_diff(east_features.get('east_height_cm'),
                                     west_features.get('west_height_cm')),
        'age_diff_years': safe_diff(east_features.get('east_age_years'),
                                     west_features.get('west_age_years')),
        'rank_diff': safe_diff(west_features.get('west_rank_numeric'),
                               east_features.get('east_rank_numeric')),  # Positive = east higher ranked
        'career_win_rate_diff': safe_diff(east_features.get('east_career_win_rate'),
                                          west_features.get('west_career_win_rate')),
    }


def compute_context_features(row: Dict) -> Dict:
    """Compute bout context features."""
    basho_id = str(row.get('bashoId', ''))

    # Parse basho
    try:
        year = int(basho_id[:4])
        month = int(basho_id[4:6])
    except (ValueError, IndexError):
        year = 0
        month = 0

    # Tournament number (1-6)
    month_to_num = {1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6}
    tournament_number = month_to_num.get(month, 0)

    # Venue (approximate from month)
    month_to_venue = {
        1: 'Tokyo', 3: 'Osaka', 5: 'Tokyo',
        7: 'Nagoya', 9: 'Tokyo', 11: 'Fukuoka'
    }
    venue = month_to_venue.get(month, 'Unknown')

    return {
        'year': year,
        'tournament_number': tournament_number,
        'tournament_month': month,
        'venue': venue,
        'is_tokyo': int(venue == 'Tokyo'),
        'day_of_tournament': row.get('day', 0)
    }


# =============================================================================
# Main Feature Engineering Function
# =============================================================================

def engineer_features(matches_df: pd.DataFrame,
                      rikishi_df: Optional[pd.DataFrame] = None,
                      rank_averages_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Engineer all features for the matches dataset.

    This processes matches chronologically, computing features using only
    information available BEFORE each bout.
    """
    print("Starting feature engineering...")

    # Sort chronologically
    df = matches_df.copy()
    df = df.sort_values(['bashoId', 'day']).reset_index(drop=True)

    # Initialize tracker
    tracker = WrestlerStatsTracker()

    # Build rikishi lookup if available
    rikishi_lookup = {}
    if rikishi_df is not None:
        for _, row in rikishi_df.iterrows():
            rikishi_lookup[row.get('id')] = row.to_dict()

    # Build rank averages lookup
    rank_avg_lookup = {}
    if rank_averages_df is not None:
        for _, row in rank_averages_df.iterrows():
            rank_avg_lookup[row['rank_numeric']] = {
                'avg_elo': row.get('avg_elo_for_rank', 1500),
                'avg_glicko': row.get('avg_glicko_for_rank', 1500)
            }

    # Process each bout
    feature_rows = []
    n = len(df)

    for idx, row in df.iterrows():
        if idx % 25000 == 0:
            print(f"Processing bout {idx:,} / {n:,}")

        bout = row.to_dict()
        east_id = bout['eastId']
        west_id = bout['westId']
        basho_id = bout.get('bashoId', '')

        # Get wrestler info
        east_info = rikishi_lookup.get(east_id, {})
        west_info = rikishi_lookup.get(west_id, {})

        # Physical features
        east_physical = {
            f'east{k}': bout.get(f'east{k}', east_info.get(k.lower()))
            for k in ['Height', 'Weight', 'Birthdate']
        }
        west_physical = {
            f'west{k}': bout.get(f'west{k}', west_info.get(k.lower()))
            for k in ['Height', 'Weight', 'Birthdate']
        }

        east_features = compute_physical_features({**bout, **east_physical}, 'east')
        west_features = compute_physical_features({**bout, **west_physical}, 'west')

        # Career stats
        east_career = tracker.get_career_stats(east_id, basho_id)
        west_career = tracker.get_career_stats(west_id, basho_id)

        for k, v in east_career.items():
            east_features[f'east_{k}'] = v
        for k, v in west_career.items():
            west_features[f'west_{k}'] = v

        # Current basho stats
        east_basho = tracker.get_current_basho_stats(east_id)
        west_basho = tracker.get_current_basho_stats(west_id)

        for k, v in east_basho.items():
            east_features[f'east_{k}'] = v
        for k, v in west_basho.items():
            west_features[f'west_{k}'] = v

        # Recent form
        for n_bouts in [5, 10, 15, 20]:
            east_features[f'east_win_rate_last_{n_bouts}_bouts'] = tracker.get_recent_form(east_id, n_bouts)
            west_features[f'west_win_rate_last_{n_bouts}_bouts'] = tracker.get_recent_form(west_id, n_bouts)

        # Basho win rates
        for n_basho in [1, 2, 3]:
            east_features[f'east_win_rate_last_{n_basho}_basho'] = tracker.get_basho_win_rates(east_id, n_basho)
            west_features[f'west_win_rate_last_{n_basho}_basho'] = tracker.get_basho_win_rates(west_id, n_basho)

        # Kachikoshi/Makekoshi streaks
        east_features['east_kachikoshi_streak'] = tracker.get_kachikoshi_streak(east_id)
        east_features['east_makekoshi_streak'] = tracker.get_makekoshi_streak(east_id)
        west_features['west_kachikoshi_streak'] = tracker.get_kachikoshi_streak(west_id)
        west_features['west_makekoshi_streak'] = tracker.get_makekoshi_streak(west_id)

        # Style profiles
        east_style = tracker.get_style_profile(east_id)
        west_style = tracker.get_style_profile(west_id)

        for k, v in east_style.items():
            east_features[f'east_{k}'] = v
        for k, v in west_style.items():
            west_features[f'west_{k}'] = v

        # Head to head
        h2h = tracker.get_h2h_stats(east_id, west_id)
        h2h_features = {f'east_{k}': v for k, v in h2h.items()}

        # Pressure features
        east_pressure = compute_pressure_features(bout, east_basho, 'east')
        west_pressure = compute_pressure_features(bout, west_basho, 'west')

        # Rank features
        east_rank = bout.get('eastRank', bout.get('east_rank'))
        west_rank = bout.get('westRank', bout.get('west_rank'))
        east_features['east_rank_numeric'] = parse_banzuke_rank(east_rank)
        west_features['west_rank_numeric'] = parse_banzuke_rank(west_rank)

        # Rating vs expected for rank
        if east_features['east_rank_numeric'] in rank_avg_lookup:
            avg = rank_avg_lookup[east_features['east_rank_numeric']]
            east_elo = bout.get('east_elo', 1500)
            east_glicko = bout.get('east_glicko_rating', 1500)
            east_features['east_elo_minus_expected'] = east_elo - avg['avg_elo']
            east_features['east_glicko_minus_expected'] = east_glicko - avg['avg_glicko']

        if west_features['west_rank_numeric'] in rank_avg_lookup:
            avg = rank_avg_lookup[west_features['west_rank_numeric']]
            west_elo = bout.get('west_elo', 1500)
            west_glicko = bout.get('west_glicko_rating', 1500)
            west_features['west_elo_minus_expected'] = west_elo - avg['avg_elo']
            west_features['west_glicko_minus_expected'] = west_glicko - avg['avg_glicko']

        # Pairwise features
        pairwise = compute_pairwise_features(east_features, west_features)

        # Context features
        context = compute_context_features(bout)

        # Combine all features
        all_features = {
            'bout_id': bout.get('bout_id', idx),
            'bashoId': basho_id,
            'day': bout.get('day'),
            'eastId': east_id,
            'westId': west_id,
            'winnerId': bout.get('winnerId'),
            'kimarite': bout.get('kimarite'),
            'east_won': int(bout.get('winnerId') == east_id) if bout.get('winnerId') else None,
            **east_features,
            **west_features,
            **h2h_features,
            **east_pressure,
            **west_pressure,
            **pairwise,
            **context,
            # Copy rating columns
            'east_elo': bout.get('east_elo'),
            'west_elo': bout.get('west_elo'),
            'east_glicko_rating': bout.get('east_glicko_rating'),
            'east_glicko_rd': bout.get('east_glicko_rd'),
            'east_glicko_vol': bout.get('east_glicko_vol'),
            'west_glicko_rating': bout.get('west_glicko_rating'),
            'west_glicko_rd': bout.get('west_glicko_rd'),
            'west_glicko_vol': bout.get('west_glicko_vol'),
            'elo_diff': bout.get('elo_diff'),
            'glicko_rating_diff': bout.get('glicko_rating_diff'),
        }

        feature_rows.append(all_features)

        # NOW record the bout (after computing features)
        tracker.record_bout(bout)

    print(f"Feature engineering complete. Generated {len(feature_rows):,} rows.")
    return pd.DataFrame(feature_rows)
