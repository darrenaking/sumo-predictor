"""
Interest scoring for ranking sumo bouts by viewer appeal.

Factors considered:
- Closeness: How close is the predicted outcome (coin flip more interesting)
- Stakes: Kachikoshi, ozeki kadoban, yusho contention
- Storyline: H2H rivalry, streaks, style clashes
- Star power: Wrestler prominence by rank
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class InterestBreakdown:
    """Breakdown of interest score components."""
    total: float
    closeness: float
    stakes: float
    storyline: float
    star_power: float
    reasons: List[str]


# Weights for interest score components
WEIGHTS = {
    'closeness': 0.30,
    'stakes': 0.35,
    'storyline': 0.20,
    'star_power': 0.15,
}


def compute_closeness_score(p_east_wins: float) -> float:
    """
    Score based on how close the predicted outcome is.

    Coin flip (50%) = 100, 75% favorite = 50, 90% favorite = 20
    """
    return 100 - abs(p_east_wins - 0.5) * 200


def compute_stakes_score(features: Dict, standings: Optional[Dict] = None, day: int = 1) -> Tuple[float, List[str]]:
    """
    Score based on what's at stake in the bout.

    Returns (score, list of stake reasons)
    """
    score = 0
    reasons = []

    # Kachikoshi pressure (7 wins, need 1 more)
    east_wins = features.get('east_basho_wins', 0)
    west_wins = features.get('west_basho_wins', 0)
    east_losses = features.get('east_basho_losses', 0)
    west_losses = features.get('west_basho_losses', 0)

    if east_wins == 7:
        score += 30
        reasons.append("East needs one win for kachikoshi")
    if west_wins == 7:
        score += 30
        reasons.append("West needs one win for kachikoshi")

    # Makekoshi avoidance (7 losses)
    if east_losses == 7:
        score += 20
        reasons.append("East fighting to avoid makekoshi")
    if west_losses == 7:
        score += 20
        reasons.append("West fighting to avoid makekoshi")

    # Final day drama
    if day == 15:
        if east_wins == 7 or west_wins == 7:
            score += 15  # Extra drama on senshuraku
            reasons.append("Senshuraku kachikoshi drama")

    # Ozeki kadoban (ozeki with 6+ losses risks demotion)
    if features.get('east_is_ozeki') and east_losses >= 6:
        score += 35
        reasons.append("East ozeki is kadoban")
    if features.get('west_is_ozeki') and west_losses >= 6:
        score += 35
        reasons.append("West ozeki is kadoban")

    # Yokozuna struggling
    if features.get('east_is_yokozuna') and east_losses > east_wins:
        score += 25
        reasons.append("Yokozuna with losing record")
    if features.get('west_is_yokozuna') and west_losses > west_wins:
        score += 25
        reasons.append("Yokozuna with losing record")

    # Yusho contention (if standings provided)
    if standings:
        leader_wins = standings.get('leader_wins', 0)
        east_id = features.get('eastId')
        west_id = features.get('westId')

        east_behind = leader_wins - east_wins
        west_behind = leader_wins - west_wins

        if east_behind <= 1 or west_behind <= 1:
            score += 40
            if east_behind == 0 or west_behind == 0:
                reasons.append("Leader in yusho race")
            else:
                reasons.append("One loss off the pace")

    # Sanyaku implications (rank near promotion threshold)
    east_rank = features.get('east_rank_numeric', 50)
    west_rank = features.get('west_rank_numeric', 50)

    if 40 <= east_rank <= 55 and east_wins >= 8:  # Near komusubi, has kachikoshi
        score += 15
        reasons.append("East in sanyaku promotion hunt")
    if 40 <= west_rank <= 55 and west_wins >= 8:
        score += 15
        reasons.append("West in sanyaku promotion hunt")

    return min(score, 100), reasons


def compute_storyline_score(features: Dict) -> Tuple[float, List[str]]:
    """
    Score based on narrative interest factors.

    Returns (score, list of storyline reasons)
    """
    score = 0
    reasons = []

    # Head-to-head rivalry
    h2h_total = features.get('east_h2h_total_bouts', 0)
    if h2h_total >= 5:
        h2h_wins = features.get('east_h2h_wins', 0)
        h2h_losses = h2h_total - h2h_wins

        # Close rivalry
        if abs(h2h_wins - h2h_losses) <= 2:
            score += 20
            reasons.append(f"Close rivalry ({h2h_wins}-{h2h_losses})")
        # Revenge opportunity
        elif h2h_wins < h2h_losses - 2:
            score += 15
            reasons.append(f"Revenge match opportunity ({h2h_wins}-{h2h_losses})")

    # First meeting
    if features.get('east_h2h_never_met') or h2h_total == 0:
        score += 10
        reasons.append("First meeting")

    # Current form streaks
    east_win_streak = features.get('east_current_win_streak', 0)
    west_win_streak = features.get('west_current_win_streak', 0)
    east_loss_streak = features.get('east_current_loss_streak', 0)
    west_loss_streak = features.get('west_current_loss_streak', 0)

    if east_win_streak >= 5:
        score += 15
        reasons.append(f"East on {east_win_streak}-bout winning streak")
    if west_win_streak >= 5:
        score += 15
        reasons.append(f"West on {west_win_streak}-bout winning streak")

    # Clash of streaks
    if east_win_streak >= 3 and west_win_streak >= 3:
        score += 10
        reasons.append("Both wrestlers on hot streaks")

    # Style clash (pusher vs grappler)
    east_push = features.get('east_pct_wins_by_push_career', 0)
    east_grapple = features.get('east_pct_wins_by_grapple_career', 0)
    west_push = features.get('west_pct_wins_by_push_career', 0)
    west_grapple = features.get('west_pct_wins_by_grapple_career', 0)

    east_is_pusher = east_push > 0.5
    east_is_grappler = east_grapple > 0.5
    west_is_pusher = west_push > 0.5
    west_is_grappler = west_grapple > 0.5

    if (east_is_pusher and west_is_grappler) or (east_is_grappler and west_is_pusher):
        score += 10
        reasons.append("Pusher vs grappler matchup")

    # Form clash (one hot, one cold)
    east_recent = features.get('east_win_rate_last_3_basho', 0.5)
    west_recent = features.get('west_win_rate_last_3_basho', 0.5)

    if abs(east_recent - west_recent) > 0.15:
        score += 10
        if east_recent > west_recent:
            reasons.append("East in better recent form")
        else:
            reasons.append("West in better recent form")

    return min(score, 100), reasons


def rank_to_star_power(rank_numeric: int) -> float:
    """
    Convert numeric rank to star power score.

    Yokozuna = 100, Ozeki = 80, Sekiwake = 70, Komusubi = 60,
    M1-M5 = 40-50, lower Maegashira = 20-35
    """
    if rank_numeric <= 4:  # Yokozuna
        return 100
    elif rank_numeric <= 10:  # Ozeki
        return 80
    elif rank_numeric <= 30:  # Sekiwake
        return 70
    elif rank_numeric <= 40:  # Komusubi
        return 60
    elif rank_numeric <= 60:  # M1-M5
        return 50 - (rank_numeric - 50) * 2
    else:  # Lower Maegashira
        return max(20, 40 - (rank_numeric - 60))


def compute_star_power_score(features: Dict) -> float:
    """Compute star power based on wrestler ranks."""
    east_rank = features.get('east_rank_numeric', 70)
    west_rank = features.get('west_rank_numeric', 70)

    east_star = rank_to_star_power(east_rank)
    west_star = rank_to_star_power(west_rank)

    return (east_star + west_star) / 2


def compute_interest_scores(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    standings: Optional[Dict] = None,
    day: int = 1
) -> pd.DataFrame:
    """
    Compute interest scores for all bouts.

    Args:
        predictions_df: DataFrame with bout predictions (must have 'pred_east_win_prob')
        features_df: DataFrame with bout features
        standings: Optional dict with current tournament standings
        day: Tournament day (1-15)

    Returns:
        DataFrame with interest scores and breakdowns
    """
    results = []

    for idx in range(len(predictions_df)):
        pred_row = predictions_df.iloc[idx]
        feat_row = features_df.iloc[idx]
        features = feat_row.to_dict()

        p_east = pred_row.get('pred_east_win_prob', 0.5)

        # Compute components
        closeness = compute_closeness_score(p_east)
        stakes, stakes_reasons = compute_stakes_score(features, standings, day)
        storyline, storyline_reasons = compute_storyline_score(features)
        star_power = compute_star_power_score(features)

        # Weighted total
        total = (
            WEIGHTS['closeness'] * closeness +
            WEIGHTS['stakes'] * stakes +
            WEIGHTS['storyline'] * storyline +
            WEIGHTS['star_power'] * star_power
        )

        # Combine reasons
        all_reasons = stakes_reasons + storyline_reasons

        results.append({
            'bout_idx': idx,
            'interest_score': total,
            'closeness_score': closeness,
            'stakes_score': stakes,
            'storyline_score': storyline,
            'star_power_score': star_power,
            'interest_reasons': all_reasons,
        })

    return pd.DataFrame(results)


def rank_bouts(interest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank bouts by interest score.

    Returns DataFrame sorted by interest_score descending with rank column.
    """
    ranked = interest_df.sort_values('interest_score', ascending=False).reset_index(drop=True)
    ranked['interest_rank'] = range(1, len(ranked) + 1)
    return ranked


def select_bout_of_day(interest_df: pd.DataFrame) -> int:
    """
    Select the most interesting bout as "Bout of the Day".

    Returns the bout_idx of the selected bout.
    """
    ranked = rank_bouts(interest_df)
    return ranked.iloc[0]['bout_idx']


def get_interest_label(score: float) -> str:
    """Convert interest score to human-readable label."""
    if score >= 80:
        return "Must Watch"
    elif score >= 65:
        return "Highly Interesting"
    elif score >= 50:
        return "Worth Watching"
    elif score >= 35:
        return "Standard Match"
    else:
        return "Lower Card"
