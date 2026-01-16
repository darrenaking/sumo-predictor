"""
Narrative generation for sumo bout previews.

Generates natural language descriptions of expected bout outcomes based on
model predictions and feature values.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Kimarite Categories and Names
# =============================================================================

KIMARITE_CATEGORIES = {
    'push': ['oshidashi', 'tsukidashi', 'oshitaoshi', 'tsukiotoshi', 'tsukitaoshi', 'okuridashi'],
    'grapple': ['yorikiri', 'uwatenage', 'shitatenage', 'sukuinage', 'kotenage', 'yoritaoshi',
                'uwatedashinage', 'makiotoshi', 'kimedashi', 'sotogake', 'uchigake'],
    'evasion': ['hatakikomi', 'hikiotoshi', 'katasukashi']
}

KIMARITE_DESCRIPTIONS = {
    'yorikiri': 'force out',
    'oshidashi': 'push out',
    'hatakikomi': 'slap down',
    'uwatenage': 'overarm throw',
    'oshitaoshi': 'push down',
    'shitatenage': 'underarm throw',
    'tsukiotoshi': 'thrust down',
    'hikiotoshi': 'hand pull down',
    'kotenage': 'arm lock throw',
    'sukuinage': 'scoop throw',
    'tsukidashi': 'thrust out',
    'okuridashi': 'rear push out',
    'yoritaoshi': 'frontal force down',
    'katasukashi': 'under-shoulder swing down',
    'sotogake': 'outside leg trip',
    'uwatedashinage': 'pulling overarm throw',
    'makiotoshi': 'twist down',
    'tsukitaoshi': 'thrust and push down',
    'kimedashi': 'arm bar force out',
    'uchigake': 'inside leg trip',
}


# =============================================================================
# Narrative Components
# =============================================================================

@dataclass
class BoutPrediction:
    """Container for all prediction outputs for a bout."""
    east_name: str
    west_name: str
    p_east_wins: float
    kimarite_probs: Dict[str, float]  # kimarite -> probability
    category_probs: Dict[str, float]  # push/grapple/evasion -> probability
    predicted_duration: Optional[float] = None
    features: Optional[Dict] = None


def get_confidence_level(p_win: float) -> Tuple[str, str]:
    """
    Convert win probability to confidence description.

    Returns (description, intensity) tuple.
    """
    # Normalize to favorite's probability
    p_favorite = max(p_win, 1 - p_win)

    if p_favorite >= 0.75:
        return "heavy favorite", "should beat"
    elif p_favorite >= 0.65:
        return "solid favorite", "is favored against"
    elif p_favorite >= 0.58:
        return "slight favorite", "has the edge over"
    elif p_favorite >= 0.52:
        return "narrow favorite", "is narrowly favored over"
    else:
        return "coin flip", "faces"


def get_bout_type_description(category_probs: Dict[str, float]) -> str:
    """Describe expected bout type based on kimarite category probabilities."""
    max_cat = max(category_probs.items(), key=lambda x: x[1])

    descriptions = {
        'push': [
            "Expect a pushing/thrusting contest.",
            "This should be a power-pushing battle.",
            "Look for a shoving match at the tachiai.",
        ],
        'grapple': [
            "Expect a belt battle.",
            "This will likely come down to who gets the better grip.",
            "Look for a grappling contest on the mawashi.",
        ],
        'evasion': [
            "Watch for evasive maneuvers.",
            "Someone may try to sidestep or pull down.",
            "Evasion could play a role here.",
        ]
    }

    import random
    return random.choice(descriptions.get(max_cat[0], ["Style uncertain."]))


def get_kimarite_description(kimarite_probs: Dict[str, float], top_n: int = 3) -> str:
    """Describe most likely winning techniques."""
    sorted_kim = sorted(kimarite_probs.items(), key=lambda x: x[1], reverse=True)
    top_kim = sorted_kim[:top_n]

    if not top_kim:
        return ""

    descriptions = []
    for kim, prob in top_kim:
        if prob > 0.15:  # Only mention if reasonably likely
            eng_name = KIMARITE_DESCRIPTIONS.get(kim, kim)
            descriptions.append(f"{kim} ({eng_name})")

    if len(descriptions) == 1:
        return f"Most likely technique: {descriptions[0]}."
    elif len(descriptions) > 1:
        return f"Likely techniques: {', '.join(descriptions[:-1])}, or {descriptions[-1]}."
    return ""


def get_closeness_description(p_win: float, predicted_duration: Optional[float] = None) -> str:
    """Describe expected bout closeness."""
    p_favorite = max(p_win, 1 - p_win)

    if p_favorite >= 0.75:
        closeness = "one-sided affair"
    elif p_favorite >= 0.65:
        closeness = "competitive match"
    elif p_favorite >= 0.55:
        closeness = "close contest"
    else:
        closeness = "real toss-up"

    if predicted_duration:
        if predicted_duration < 3:
            return f"Expected to be a quick {closeness}."
        elif predicted_duration > 15:
            return f"Could be a lengthy {closeness}."
        else:
            return f"Should be a {closeness}."

    return f"Expected to be a {closeness}."


def get_pressure_narrative(features: Dict, prefix: str) -> Optional[str]:
    """Generate narrative for pressure situations."""
    narratives = []

    # Kachikoshi pressure
    if features.get(f'{prefix}_needs_one_win_for_kachikoshi'):
        day = features.get('day_of_tournament', 0)
        if day == 15:
            narratives.append("fighting for kachikoshi on the final day")
        else:
            narratives.append("needs one more win for kachikoshi")

    if features.get(f'{prefix}_already_makekoshi'):
        narratives.append("already has a losing record")

    if features.get(f'{prefix}_already_kachikoshi'):
        narratives.append("has already secured kachikoshi")

    # Rank-based pressure
    if features.get(f'{prefix}_is_ozeki'):
        # Check for kadoban (would need ozeki_kadoban feature)
        narratives.append("is fighting as ozeki")

    if features.get(f'{prefix}_is_yokozuna'):
        if features.get(f'{prefix}_yokozuna_losing_record_so_far'):
            narratives.append("is struggling as yokozuna")
        if features.get(f'{prefix}_yokozuna_multiple_losses_early'):
            narratives.append("has had early losses, putting pressure on his tournament")

    return "; ".join(narratives) if narratives else None


def get_h2h_narrative(features: Dict) -> Optional[str]:
    """Generate head-to-head history narrative."""
    h2h_total = features.get('east_h2h_total_bouts', 0)

    if features.get('east_h2h_never_met') or h2h_total == 0:
        return "This is their first meeting."

    h2h_wins = features.get('east_h2h_wins', 0)
    h2h_losses = features.get('east_h2h_losses', 0)
    streak = features.get('east_h2h_current_streak', 0)

    # Get wrestler references (east/west)
    east_ref = "The east wrestler"
    west_ref = "the west wrestler"

    if h2h_total >= 5:
        if h2h_wins > h2h_losses + 3:
            return f"{east_ref} dominates this matchup historically ({h2h_wins}-{h2h_losses})."
        elif h2h_losses > h2h_wins + 3:
            return f"{east_ref} has struggled in this matchup ({h2h_wins}-{h2h_losses})."

    if abs(streak) >= 3:
        if streak > 0:
            return f"{east_ref} has won {streak} straight in this matchup."
        else:
            return f"{east_ref} has lost {abs(streak)} straight to {west_ref}."

    if h2h_total >= 3:
        return f"Head-to-head record: {h2h_wins}-{h2h_losses}."

    return None


def get_rating_vs_rank_narrative(features: Dict) -> Optional[str]:
    """Generate narrative about rating vs official rank discrepancy."""
    narratives = []

    east_elo_diff = features.get('east_elo_minus_expected', 0)
    west_elo_diff = features.get('west_elo_minus_expected', 0)

    threshold = 100  # ELO points above/below expected

    if east_elo_diff > threshold:
        narratives.append("The east wrestler is performing above his rank")
    elif east_elo_diff < -threshold:
        narratives.append("The east wrestler may be overranked based on recent results")

    if west_elo_diff > threshold:
        narratives.append("The west wrestler is performing above his rank")
    elif west_elo_diff < -threshold:
        narratives.append("The west wrestler may be overranked based on recent results")

    return ". ".join(narratives) + "." if narratives else None


def get_style_matchup_narrative(features: Dict) -> Optional[str]:
    """Generate narrative about style matchup."""
    east_push = features.get('east_pct_wins_by_push', 0)
    east_grapple = features.get('east_pct_wins_by_grapple', 0)
    west_push = features.get('west_pct_wins_by_push', 0)
    west_grapple = features.get('west_pct_wins_by_grapple', 0)

    def get_style(push, grapple):
        if push > 0.5:
            return "pusher"
        elif grapple > 0.5:
            return "grappler"
        else:
            return "balanced"

    east_style = get_style(east_push, east_grapple)
    west_style = get_style(west_push, west_grapple)

    if east_style == west_style:
        return None  # No interesting contrast

    if east_style == "pusher" and west_style == "grappler":
        return "Classic pusher vs grappler matchup — the outcome may depend on whether it stays at distance or goes to the belt."
    elif east_style == "grappler" and west_style == "pusher":
        return "Grappler vs pusher matchup — watch for who dictates the pace."

    return None


# =============================================================================
# Main Narrative Generator
# =============================================================================

def generate_bout_narrative(prediction: BoutPrediction,
                           east_name: Optional[str] = None,
                           west_name: Optional[str] = None) -> str:
    """
    Generate a complete narrative bout preview.

    Args:
        prediction: BoutPrediction with all model outputs
        east_name: Optional wrestler name (defaults to "East wrestler")
        west_name: Optional wrestler name (defaults to "West wrestler")

    Returns:
        Multi-sentence narrative description
    """
    east = east_name or prediction.east_name or "The east wrestler"
    west = west_name or prediction.west_name or "the west wrestler"

    p_east = prediction.p_east_wins
    features = prediction.features or {}

    # Determine favorite
    if p_east >= 0.5:
        favorite, underdog = east, west
        p_favorite = p_east
    else:
        favorite, underdog = west, east
        p_favorite = 1 - p_east

    # Build narrative parts
    parts = []

    # 1. Main prediction with confidence
    confidence_desc, verb = get_confidence_level(p_favorite)

    if p_favorite >= 0.52:
        parts.append(f"{favorite} {verb} {underdog} ({p_favorite*100:.0f}% confidence).")
    else:
        parts.append(f"Coin flip between {east} and {west}. Could go either way.")

    # 2. Bout type
    if prediction.category_probs:
        bout_type = get_bout_type_description(prediction.category_probs)
        parts.append(bout_type)

    # 3. Likely kimarite
    if prediction.kimarite_probs:
        kim_desc = get_kimarite_description(prediction.kimarite_probs)
        if kim_desc:
            parts.append(kim_desc)

    # 4. Closeness
    closeness = get_closeness_description(p_east, prediction.predicted_duration)
    parts.append(closeness)

    # 5. Style matchup
    if features:
        style_narrative = get_style_matchup_narrative(features)
        if style_narrative:
            parts.append(style_narrative)

    # 6. Head-to-head
    if features:
        h2h_narrative = get_h2h_narrative(features)
        if h2h_narrative:
            parts.append(h2h_narrative)

    # 7. Pressure situations
    if features:
        east_pressure = get_pressure_narrative(features, 'east')
        west_pressure = get_pressure_narrative(features, 'west')

        if east_pressure:
            parts.append(f"{east} is {east_pressure}.")
        if west_pressure:
            parts.append(f"{west} is {west_pressure}.")

    # 8. Rating vs rank
    if features:
        rating_narrative = get_rating_vs_rank_narrative(features)
        if rating_narrative:
            parts.append(rating_narrative)

    return " ".join(parts)


def generate_batch_narratives(predictions_df: pd.DataFrame,
                              kimarite_probs: np.ndarray,
                              kimarite_classes: List[str],
                              name_lookup: Optional[Dict[int, str]] = None) -> List[str]:
    """
    Generate narratives for a batch of predictions.

    Args:
        predictions_df: DataFrame with predictions and features
        kimarite_probs: Array of kimarite probabilities (n_samples x n_classes)
        kimarite_classes: List of kimarite class names
        name_lookup: Optional dict mapping wrestler ID to name

    Returns:
        List of narrative strings
    """
    narratives = []

    for idx, row in predictions_df.iterrows():
        # Build kimarite probability dict
        kim_probs = {k: kimarite_probs[idx, i] for i, k in enumerate(kimarite_classes)}

        # Sum into categories
        cat_probs = {'push': 0, 'grapple': 0, 'evasion': 0}
        for kim, prob in kim_probs.items():
            for cat, members in KIMARITE_CATEGORIES.items():
                if kim in members:
                    cat_probs[cat] += prob
                    break

        # Get names
        east_id = row.get('eastId')
        west_id = row.get('westId')
        east_name = name_lookup.get(east_id, f"Wrestler {east_id}") if name_lookup else None
        west_name = name_lookup.get(west_id, f"Wrestler {west_id}") if name_lookup else None

        # Build prediction object
        prediction = BoutPrediction(
            east_name=east_name,
            west_name=west_name,
            p_east_wins=row.get('pred_east_win_prob', 0.5),
            kimarite_probs=kim_probs,
            category_probs=cat_probs,
            features=row.to_dict()
        )

        narrative = generate_bout_narrative(prediction, east_name, west_name)
        narratives.append(narrative)

    return narratives
