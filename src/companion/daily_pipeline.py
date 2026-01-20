"""
Daily pipeline for generating sumo companion app content.

Orchestrates:
1. Fetching matchups from sumo-api.com
2. Loading models and generating predictions
3. Computing interest scores
4. Running yusho simulations
5. Generating HTML output
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_collection import (
    fetch_basho_torikumi,
    fetch_torikumi_from_banzuke,
    fetch_current_basho_standings,
    fetch_all_rikishi,
    fetch_head_to_head,
    parse_banzuke_rank,
    make_request,
)
from src.rating_systems import EloSystem, Glicko2System
from src.narrative import (
    generate_bout_narrative,
    BoutPrediction,
    KIMARITE_CATEGORIES,
)
from src.companion.interest_scoring import (
    compute_interest_scores,
    rank_bouts,
    select_bout_of_day,
    get_interest_label,
)
from src.companion.yusho_simulation import (
    simulate_yusho_race,
    get_current_standings,
    YushoRaceResult,
)
from src.companion.html_generator import (
    render_preview_page,
    render_results_page,
    render_index_page,
)


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
KAGGLE_OUTPUT = PROJECT_ROOT / "kaggle-output" / "04-v3"
SITE_OUTPUT = PROJECT_ROOT / "site"

# Rank to expected ELO mapping (higher rank = higher expected ELO)
RANK_EXPECTED_ELO = {
    'Y': 1700,   # Yokozuna
    'O': 1650,   # Ozeki
    'S': 1600,   # Sekiwake
    'K': 1575,   # Komusubi
    'M1': 1550, 'M2': 1530, 'M3': 1510, 'M4': 1490, 'M5': 1470,
    'M6': 1450, 'M7': 1430, 'M8': 1410, 'M9': 1390, 'M10': 1370,
    'M11': 1350, 'M12': 1330, 'M13': 1310, 'M14': 1290, 'M15': 1270,
    'M16': 1250, 'M17': 1230,
}


def get_expected_elo_for_rank(rank_str: str) -> float:
    """Get expected ELO rating for a given rank."""
    if not rank_str:
        return 1400
    # Extract rank letter/number
    rank_str = rank_str.strip().upper()
    if rank_str.startswith('Y'):
        return RANK_EXPECTED_ELO['Y']
    elif rank_str.startswith('O'):
        return RANK_EXPECTED_ELO['O']
    elif rank_str.startswith('S'):
        return RANK_EXPECTED_ELO['S']
    elif rank_str.startswith('K'):
        return RANK_EXPECTED_ELO['K']
    elif rank_str.startswith('M'):
        # Extract number
        import re
        match = re.search(r'M(\d+)', rank_str)
        if match:
            num = int(match.group(1))
            return RANK_EXPECTED_ELO.get(f'M{num}', 1400)
    return 1400


def get_rank_analysis(actual_elo: float, rank_str: str) -> Optional[str]:
    """Determine if wrestler is underrated or overrated by their official rank."""
    expected = get_expected_elo_for_rank(rank_str)
    diff = actual_elo - expected
    if diff > 80:
        return "underrated"  # Performing above their rank
    elif diff < -80:
        return "overrated"  # Performing below their rank
    return None


def get_expected_style(east_push_pct: float, east_grapple_pct: float,
                       west_push_pct: float, west_grapple_pct: float) -> str:
    """Determine expected bout style based on wrestler tendencies."""
    east_is_pusher = east_push_pct > 0.55
    east_is_grappler = east_grapple_pct > 0.55
    west_is_pusher = west_push_pct > 0.55
    west_is_grappler = west_grapple_pct > 0.55

    if east_is_grappler and west_is_grappler:
        return "Belt battle"
    elif east_is_pusher and west_is_pusher:
        return "Pushing match"
    elif (east_is_pusher and west_is_grappler) or (east_is_grappler and west_is_pusher):
        return "Style clash"
    else:
        return None


def format_h2h_storyline(east_name: str, west_name: str,
                         east_h2h_wins: int, total_bouts: int) -> Optional[str]:
    """Format H2H record into a storyline string.

    Only returns meaningful storylines - the actual record is shown separately.
    """
    # First meeting is shown separately, don't add to storylines
    if total_bouts == 0:
        return None

    west_h2h_wins = total_bouts - east_h2h_wins

    # Only flag notable situations without repeating the record
    if total_bouts >= 3:
        if east_h2h_wins == 0:
            return f"{east_name} seeking first career win"
        elif west_h2h_wins == 0:
            return f"{west_name} seeking first career win"
    return None


def get_streak_storylines(east_name: str, west_name: str,
                          east_form: dict, west_form: dict) -> List[str]:
    """Get storylines for winning/losing streaks > 3."""
    storylines = []

    if east_form and east_form.get('streak', 0) > 3:
        streak = east_form['streak']
        if east_form.get('streak_type') == 'W':
            storylines.append(f"{east_name} on {streak}-bout winning streak")
        else:
            storylines.append(f"{east_name} on {streak}-bout losing streak")

    if west_form and west_form.get('streak', 0) > 3:
        streak = west_form['streak']
        if west_form.get('streak_type') == 'W':
            storylines.append(f"{west_name} on {streak}-bout winning streak")
        else:
            storylines.append(f"{west_name} on {streak}-bout losing streak")

    return storylines


def get_stakes_storyline(features: Dict, east_name: str, west_name: str, day: int) -> List[str]:
    """Get stakes-related storylines."""
    storylines = []

    east_wins = features.get('east_basho_wins', 0)
    east_losses = features.get('east_basho_losses', 0)
    west_wins = features.get('west_basho_wins', 0)
    west_losses = features.get('west_basho_losses', 0)

    # Kachikoshi pressure
    if east_wins == 7:
        storylines.append(f"{east_name} needs 1 win for kachikoshi")
    if west_wins == 7:
        storylines.append(f"{west_name} needs 1 win for kachikoshi")

    # Makekoshi danger
    if east_losses == 7:
        storylines.append(f"{east_name} must win to avoid makekoshi")
    if west_losses == 7:
        storylines.append(f"{west_name} must win to avoid makekoshi")

    # Ozeki kadoban
    if features.get('east_is_ozeki') and east_losses >= 6:
        storylines.append(f"{east_name} (ozeki) fighting to avoid demotion")
    if features.get('west_is_ozeki') and west_losses >= 6:
        storylines.append(f"{west_name} (ozeki) fighting to avoid demotion")

    return storylines


def get_wrestler_last_results(wrestler_id: int, historical_results: pd.DataFrame, basho_id: str) -> list:
    """Get list of recent results (W/L) for a wrestler in this basho."""
    if historical_results is None or historical_results.empty:
        return []

    basho_results = historical_results[historical_results['bashoId'] == basho_id].copy()
    if basho_results.empty:
        return []

    # Get bouts involving this wrestler
    wrestler_bouts = basho_results[
        (basho_results['eastId'] == wrestler_id) | (basho_results['westId'] == wrestler_id)
    ].copy()

    if wrestler_bouts.empty:
        return []

    # Sort by day
    wrestler_bouts = wrestler_bouts.sort_values('day')

    # Build result list
    results = []
    for _, bout in wrestler_bouts.iterrows():
        if pd.notna(bout.get('winnerId')):
            results.append('W' if bout['winnerId'] == wrestler_id else 'L')

    return results


def get_current_streak(last_results: list) -> tuple:
    """Calculate current winning or losing streak from results list.

    Returns (streak_type, streak_length) where streak_type is 'W' or 'L'.
    """
    if not last_results:
        return (None, 0)

    # Count from the end
    streak_type = last_results[-1]
    streak_length = 0
    for result in reversed(last_results):
        if result == streak_type:
            streak_length += 1
        else:
            break

    return (streak_type, streak_length)


def format_recent_form(wins: int, losses: int, last_results: list = None) -> dict:
    """Format current basho record as recent form dict with last 3 results and streak."""
    if wins == 0 and losses == 0:
        return None
    total = wins + losses
    if total == 0:
        return None

    result = {'record': f"{wins}-{losses}", 'last_3': [], 'streak': 0, 'streak_type': None}

    # Get last 3 results for dot display
    if last_results:
        result['last_3'] = last_results[-3:]
        streak_type, streak_length = get_current_streak(last_results)
        result['streak'] = streak_length
        result['streak_type'] = streak_type

    return result


def get_wrestler_style(wrestler_id: int, historical_results: pd.DataFrame) -> Dict:
    """
    Analyze wrestler's style from historical bout results.
    Returns dict with style info.
    """
    if historical_results is None or historical_results.empty:
        return {'style': None, 'push_pct': 0.5, 'grapple_pct': 0.5}

    # Get bouts where this wrestler won
    won_bouts = historical_results[historical_results['winnerId'] == wrestler_id]

    if len(won_bouts) == 0:
        return {'style': None, 'push_pct': 0.5, 'grapple_pct': 0.5}

    # Categorize kimarite
    push_kimarite = ['oshidashi', 'oshitaoshi', 'tsukidashi', 'tsukitaoshi', 'hatakikomi', 'hikiotoshi']
    grapple_kimarite = ['yorikiri', 'yoritaoshi', 'uwatenage', 'shitatenage', 'uwatedashinage', 'shitatedashinage', 'sukuinage', 'kotenage']

    total = len(won_bouts)
    push_wins = sum(1 for k in won_bouts['kimarite'] if k and k.lower() in push_kimarite)
    grapple_wins = sum(1 for k in won_bouts['kimarite'] if k and k.lower() in grapple_kimarite)

    push_pct = push_wins / total if total > 0 else 0
    grapple_pct = grapple_wins / total if total > 0 else 0

    if push_pct > 0.5:
        style = "Pusher"
    elif grapple_pct > 0.5:
        style = "Grappler"
    else:
        style = "Balanced"

    return {'style': style, 'push_pct': push_pct, 'grapple_pct': grapple_pct}


def load_models():
    """Load trained LightGBM models."""
    import lightgbm as lgb
    import joblib

    winner_model = lgb.Booster(model_file=str(KAGGLE_OUTPUT / "winner_model.lgb"))
    kimarite_model = lgb.Booster(model_file=str(KAGGLE_OUTPUT / "kimarite_model.lgb"))
    kimarite_encoder = joblib.load(KAGGLE_OUTPUT / "kimarite_encoder.joblib")

    # Load feature columns
    with open(KAGGLE_OUTPUT / "feature_columns.csv") as f:
        feature_columns = [line.strip() for line in f if line.strip()]

    return winner_model, kimarite_model, kimarite_encoder, feature_columns


def load_rikishi_data() -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Load wrestler data and create name lookup."""
    # Try to load from cache first
    cache_path = PROJECT_ROOT / "data" / "rikishi.parquet"

    if cache_path.exists():
        rikishi_df = pd.read_parquet(cache_path)
    else:
        # Fetch from API
        print("Fetching rikishi data from API...")
        rikishi_df = fetch_all_rikishi()

        # Cache it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        rikishi_df.to_parquet(cache_path)

    # Create name lookup
    name_lookup = {}
    for _, row in rikishi_df.iterrows():
        w_id = row.get('id')
        # Prefer shikona (ring name), fall back to sumodbId or id
        name = row.get('shikonaEn') or row.get('shikona') or f"Wrestler {w_id}"
        if w_id:
            name_lookup[w_id] = name

    return rikishi_df, name_lookup


def fetch_day_matchups(basho_id: str, day: int, division: str = "Makuuchi") -> pd.DataFrame:
    """
    Fetch matchups for a specific day.

    Returns DataFrame with bout information.
    Tries torikumi endpoint first, falls back to banzuke-based extraction.
    """
    # Try torikumi endpoint first
    data = fetch_basho_torikumi(basho_id, division, day)

    if data and 'torikumi' in data:
        bouts = []
        for i, bout in enumerate(data['torikumi']):
            bouts.append({
                'bout_number': i + 1,
                'bashoId': basho_id,
                'day': day,
                'division': division,
                'eastId': bout.get('eastId'),
                'westId': bout.get('westId'),
                'eastShikona': bout.get('eastShikona'),
                'westShikona': bout.get('westShikona'),
                'eastRank': bout.get('eastRank'),
                'westRank': bout.get('westRank'),
                'winnerId': bout.get('winnerId'),
                'kimarite': bout.get('kimarite'),
            })
        return pd.DataFrame(bouts)

    # Fallback to banzuke-based extraction
    print(f"Torikumi endpoint unavailable, using banzuke fallback for {basho_id} day {day}")
    banzuke_bouts = fetch_torikumi_from_banzuke(basho_id, day, division)

    if not banzuke_bouts:
        print(f"No matchups found for {basho_id} day {day}")
        return pd.DataFrame()

    bouts = []
    for i, bout in enumerate(banzuke_bouts):
        bouts.append({
            'bout_number': i + 1,
            'bashoId': basho_id,
            'day': day,
            'division': division,
            'eastId': bout.get('eastId'),
            'westId': bout.get('westId'),
            'eastShikona': bout.get('eastShikona'),
            'westShikona': bout.get('westShikona'),
            'eastRank': bout.get('eastRank'),
            'westRank': bout.get('westRank'),
            'winnerId': bout.get('winnerId'),
            'kimarite': bout.get('kimarite'),
        })

    return pd.DataFrame(bouts)


def fetch_basho_results_through_day(basho_id: str, through_day: int) -> pd.DataFrame:
    """Fetch all results for a basho through a given day."""
    all_bouts = []

    for day in range(1, through_day + 1):
        day_bouts = fetch_day_matchups(basho_id, day)
        if not day_bouts.empty:
            all_bouts.append(day_bouts)

    if not all_bouts:
        return pd.DataFrame()

    return pd.concat(all_bouts, ignore_index=True)


def compute_bout_features(
    bout: Dict,
    rikishi_df: pd.DataFrame,
    historical_results: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Compute features for a single bout.

    Simplified version - in production would use full feature engineering.
    """
    east_id = bout['eastId']
    west_id = bout['westId']

    # Get wrestler info
    east_info = rikishi_df[rikishi_df['id'] == east_id].iloc[0] if len(rikishi_df[rikishi_df['id'] == east_id]) > 0 else {}
    west_info = rikishi_df[rikishi_df['id'] == west_id].iloc[0] if len(rikishi_df[rikishi_df['id'] == west_id]) > 0 else {}

    # Parse ranks
    east_rank = parse_banzuke_rank(bout.get('eastRank', ''))
    west_rank = parse_banzuke_rank(bout.get('westRank', ''))

    features = {
        'eastId': east_id,
        'westId': west_id,
        'east_rank_numeric': east_rank,
        'west_rank_numeric': west_rank,
        'rank_diff': west_rank - east_rank,
        'east_is_yokozuna': east_rank <= 4,
        'west_is_yokozuna': west_rank <= 4,
        'east_is_ozeki': 5 <= east_rank <= 10,
        'west_is_ozeki': 5 <= west_rank <= 10,
    }

    # Add basho context if we have historical results
    if historical_results is not None and not historical_results.empty:
        basho_results = historical_results[historical_results['bashoId'] == bout['bashoId']]

        # Current tournament record for each wrestler
        east_wins = len(basho_results[(basho_results['winnerId'] == east_id)])
        east_losses = len(basho_results[
            ((basho_results['eastId'] == east_id) | (basho_results['westId'] == east_id)) &
            (basho_results['winnerId'] != east_id) &
            (basho_results['winnerId'].notna())
        ])

        west_wins = len(basho_results[(basho_results['winnerId'] == west_id)])
        west_losses = len(basho_results[
            ((basho_results['eastId'] == west_id) | (basho_results['westId'] == west_id)) &
            (basho_results['winnerId'] != west_id) &
            (basho_results['winnerId'].notna())
        ])

        features.update({
            'east_basho_wins': east_wins,
            'east_basho_losses': east_losses,
            'west_basho_wins': west_wins,
            'west_basho_losses': west_losses,
        })

        # H2H
        h2h = historical_results[
            ((historical_results['eastId'] == east_id) & (historical_results['westId'] == west_id)) |
            ((historical_results['eastId'] == west_id) & (historical_results['westId'] == east_id))
        ]

        if len(h2h) > 0:
            east_h2h_wins = len(h2h[h2h['winnerId'] == east_id])
            features['east_h2h_total_bouts'] = len(h2h)
            features['east_h2h_wins'] = east_h2h_wins
            features['east_h2h_losses'] = len(h2h) - east_h2h_wins
            features['east_h2h_never_met'] = False
        else:
            features['east_h2h_total_bouts'] = 0
            features['east_h2h_wins'] = 0
            features['east_h2h_losses'] = 0
            features['east_h2h_never_met'] = True

    return features


def generate_predictions(
    bouts_df: pd.DataFrame,
    features_list: List[Dict],
    winner_model,
    kimarite_model,
    kimarite_encoder,
    feature_columns: List[str]
) -> pd.DataFrame:
    """Generate predictions for all bouts."""
    # For now, use a simplified prediction based on rank
    # In production, would build full feature matrix and use model

    predictions = []

    for i, bout in bouts_df.iterrows():
        features = features_list[i]

        # Simple rank-based prediction
        east_rank = features.get('east_rank_numeric', 50)
        west_rank = features.get('west_rank_numeric', 50)

        # Lower rank number = higher rank = more likely to win
        rank_diff = west_rank - east_rank

        # Sigmoid to convert rank diff to probability
        # Each rank difference ~ 2-3% advantage
        p_east = 1 / (1 + np.exp(-rank_diff * 0.05))

        # Clamp to reasonable range
        p_east = max(0.25, min(0.75, p_east))

        predictions.append({
            'bout_idx': i,
            'pred_east_win_prob': p_east,
            'pred_west_win_prob': 1 - p_east,
            'predicted_winner': 'east' if p_east > 0.5 else 'west',
        })

    return pd.DataFrame(predictions)


def generate_preview(basho_id: str, day: int, output_dir: Optional[Path] = None):
    """
    Generate preview page for a tournament day.

    Args:
        basho_id: Tournament ID (YYYYMM)
        day: Day number (1-15)
        output_dir: Output directory (defaults to site/{basho_id}/)
    """
    print(f"Generating preview for {basho_id} day {day}...")

    # Set up output directory
    if output_dir is None:
        basho_dir = f"{basho_id[:4]}-{basho_id[4:]}"
        output_dir = SITE_OUTPUT / basho_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    rikishi_df, name_lookup = load_rikishi_data()

    # Fetch today's matchups
    bouts_df = fetch_day_matchups(basho_id, day)

    if bouts_df.empty:
        print(f"No matchups found for {basho_id} day {day}")
        return

    # Fetch results through yesterday (for context)
    historical_results = None
    if day > 1:
        historical_results = fetch_basho_results_through_day(basho_id, day - 1)

    # Compute features for each bout
    features_list = []
    for _, bout in bouts_df.iterrows():
        features = compute_bout_features(bout.to_dict(), rikishi_df, historical_results)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Generate predictions (simplified for now)
    predictions_df = generate_predictions(
        bouts_df, features_list,
        None, None, None, []  # Models not used in simplified version
    )

    # Compute interest scores
    standings = None
    if historical_results is not None and not historical_results.empty:
        standings_df = get_current_standings(historical_results, basho_id, day - 1)
        if not standings_df.empty:
            standings = {'leader_wins': standings_df['wins'].max()}

    interest_df = compute_interest_scores(predictions_df, features_df, standings, day)
    ranked_interest = rank_bouts(interest_df)

    # Select bout of the day
    bout_of_day_idx = select_bout_of_day(interest_df)

    # Run yusho simulation (if we have standings)
    yusho_result = None
    if standings is not None and day > 1:
        # Build prediction dict for remaining bouts
        bout_predictions = {}
        for i, row in predictions_df.iterrows():
            east_id = bouts_df.iloc[i]['eastId']
            west_id = bouts_df.iloc[i]['westId']
            bout_predictions[(east_id, west_id)] = row['pred_east_win_prob']

        standings_df = get_current_standings(historical_results, basho_id, day - 1)

        yusho_result = simulate_yusho_race(
            standings_df,
            bouts_df,
            bout_predictions,
            name_lookup,
            num_simulations=10000,
        )

    # Initialize rating systems and compute ratings from historical data
    elo = EloSystem(k_factor=32.0)
    glicko = Glicko2System(tau=0.5)

    # Process all historical bouts to build ratings
    if historical_results is not None and not historical_results.empty:
        for _, hbout in historical_results.iterrows():
            if pd.notna(hbout.get('winnerId')):
                winner_id = hbout['winnerId']
                east_id = hbout['eastId']
                west_id = hbout['westId']
                loser_id = west_id if winner_id == east_id else east_id
                elo.update(winner_id, loser_id)
                glicko.update_single_bout(winner_id, loser_id)

    # Build bout data for template (in bout order, not interest order)
    bout_data = []
    for idx in range(len(bouts_df)):
        bout = bouts_df.iloc[idx]
        pred = predictions_df.iloc[idx]
        interest = interest_df[interest_df['bout_idx'] == idx].iloc[0]
        features = features_list[idx]

        east_id = bout['eastId']
        west_id = bout['westId']
        east_name = bout['eastShikona']
        west_name = bout['westShikona']

        p_east = pred['pred_east_win_prob']

        # Get ratings
        east_elo = elo.get_rating(east_id)
        west_elo = elo.get_rating(west_id)
        east_glicko, east_rd, _ = glicko.get_rating(east_id)
        west_glicko, west_rd, _ = glicko.get_rating(west_id)

        # Rank analysis (underrated/overrated)
        east_rank_analysis = get_rank_analysis(east_elo, bout['eastRank'])
        west_rank_analysis = get_rank_analysis(west_elo, bout['westRank'])

        # Recent form (current basho record with streak)
        east_last_results = get_wrestler_last_results(east_id, historical_results, basho_id)
        west_last_results = get_wrestler_last_results(west_id, historical_results, basho_id)
        east_form = format_recent_form(
            features.get('east_basho_wins', 0),
            features.get('east_basho_losses', 0),
            east_last_results
        )
        west_form = format_recent_form(
            features.get('west_basho_wins', 0),
            features.get('west_basho_losses', 0),
            west_last_results
        )

        # Get wrestler styles
        east_style_info = get_wrestler_style(east_id, historical_results)
        west_style_info = get_wrestler_style(west_id, historical_results)

        # Compute expected bout style
        expected_style = get_expected_style(
            east_style_info['push_pct'], east_style_info['grapple_pct'],
            west_style_info['push_pct'], west_style_info['grapple_pct']
        )

        # Format prediction string - simpler for close matches
        if abs(p_east - 0.5) < 0.08:
            pred_str = "Close match"
        elif p_east >= 0.5:
            pred_str = f"{east_name} ({p_east*100:.0f}%)"
        else:
            pred_str = f"{west_name} ({(1-p_east)*100:.0f}%)"

        # Fetch lifetime H2H record from API
        h2h_data = fetch_head_to_head(east_id, west_id)
        h2h_east_wins = h2h_data['east_wins']
        h2h_west_wins = h2h_data['west_wins']
        h2h_total = h2h_data['total']

        # Build storylines
        storylines = []

        # H2H storyline
        h2h_storyline = format_h2h_storyline(
            east_name, west_name,
            h2h_east_wins,
            h2h_total
        )
        if h2h_storyline:
            storylines.append(h2h_storyline)

        # Stakes storylines
        stakes = get_stakes_storyline(features, east_name, west_name, day)
        storylines.extend(stakes)

        # Streak storylines (> 3 bouts)
        streak_stories = get_streak_storylines(east_name, west_name, east_form, west_form)
        storylines.extend(streak_stories)

        # Must watch tag - based on interest score
        is_must_watch = interest['interest_score'] >= 70

        bout_data.append({
            'bout_number': bout['bout_number'],
            'east_name': east_name,
            'west_name': west_name,
            'east_rank': bout['eastRank'],
            'west_rank': bout['westRank'],
            'prediction': pred_str,
            'p_east': p_east,
            'is_close': abs(p_east - 0.5) < 0.08,
            'interest_score': interest['interest_score'],
            'interest_label': get_interest_label(interest['interest_score']),
            'interest_reasons': interest['interest_reasons'],
            'is_bout_of_day': idx == bout_of_day_idx,
            'is_must_watch': is_must_watch,
            'features': features,
            # Ratings
            'east_elo': round(east_elo),
            'west_elo': round(west_elo),
            'east_glicko': round(east_glicko),
            'west_glicko': round(west_glicko),
            'east_rank_analysis': east_rank_analysis,
            'west_rank_analysis': west_rank_analysis,
            # Recent form and style
            'east_form': east_form,
            'west_form': west_form,
            'east_style': east_style_info['style'],
            'west_style': west_style_info['style'],
            'expected_style': expected_style,
            'storylines': storylines,
            # Head-to-head (lifetime)
            'h2h_east_wins': h2h_east_wins,
            'h2h_west_wins': h2h_west_wins,
            'h2h_total': h2h_total,
        })

    # Render HTML
    html = render_preview_page(
        basho_id=basho_id,
        day=day,
        bouts=bout_data,
        yusho_race=yusho_result,
        name_lookup=name_lookup,
    )

    # Write output
    output_file = output_dir / f"day-{day:02d}-preview.html"
    output_file.write_text(html)
    print(f"Written: {output_file}")


def generate_results(basho_id: str, day: int, output_dir: Optional[Path] = None):
    """
    Generate results/recap page for a tournament day.

    Args:
        basho_id: Tournament ID (YYYYMM)
        day: Day number (1-15)
        output_dir: Output directory
    """
    print(f"Generating results for {basho_id} day {day}...")

    # Set up output directory
    if output_dir is None:
        basho_dir = f"{basho_id[:4]}-{basho_id[4:]}"
        output_dir = SITE_OUTPUT / basho_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    rikishi_df, name_lookup = load_rikishi_data()

    # Fetch today's results
    bouts_df = fetch_day_matchups(basho_id, day)

    if bouts_df.empty:
        print(f"No results found for {basho_id} day {day}")
        return

    # Check if results are available
    if bouts_df['winnerId'].isna().all():
        print(f"Results not yet available for {basho_id} day {day}")
        return

    # Fetch all results through today
    all_results = fetch_basho_results_through_day(basho_id, day)

    # Compute features and predictions (as they would have been before the day)
    historical_results = fetch_basho_results_through_day(basho_id, day - 1) if day > 1 else None

    features_list = []
    for _, bout in bouts_df.iterrows():
        features = compute_bout_features(bout.to_dict(), rikishi_df, historical_results)
        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    predictions_df = generate_predictions(
        bouts_df, features_list,
        None, None, None, []
    )

    # Analyze results
    results_data = []
    upsets = []
    correct_predictions = 0

    for i, bout in bouts_df.iterrows():
        pred = predictions_df.iloc[i]
        features = features_list[i]

        winner_id = bout['winnerId']
        east_id = bout['eastId']
        west_id = bout['westId']

        p_east = pred['pred_east_win_prob']
        predicted_winner_id = east_id if p_east > 0.5 else west_id
        predicted_prob = max(p_east, 1 - p_east)

        actual_winner = 'east' if winner_id == east_id else 'west'
        correct = predicted_winner_id == winner_id

        if correct:
            correct_predictions += 1

        # Check for upset
        is_upset = not correct and predicted_prob >= 0.65

        result = {
            'bout_number': bout['bout_number'],
            'east_name': bout['eastShikona'],
            'west_name': bout['westShikona'],
            'east_rank': bout['eastRank'],
            'west_rank': bout['westRank'],
            'winner': bout['eastShikona'] if actual_winner == 'east' else bout['westShikona'],
            'kimarite': bout['kimarite'],
            'p_east': p_east,
            'predicted_correct': correct,
            'is_upset': is_upset,
            'upset_magnitude': predicted_prob if is_upset else 0,
        }

        results_data.append(result)

        if is_upset:
            upsets.append(result)

    # Daily accuracy
    accuracy = correct_predictions / len(bouts_df) * 100 if len(bouts_df) > 0 else 0

    # Get updated standings
    standings_df = get_current_standings(all_results, basho_id, day)

    # Render HTML
    html = render_results_page(
        basho_id=basho_id,
        day=day,
        results=results_data,
        upsets=upsets,
        accuracy=accuracy,
        standings=standings_df,
        name_lookup=name_lookup,
    )

    # Write output
    output_file = output_dir / f"day-{day:02d}-results.html"
    output_file.write_text(html)
    print(f"Written: {output_file}")


def run_daily_pipeline(basho_id: str, day: int, mode: str = 'both'):
    """
    Run the daily generation pipeline.

    Args:
        basho_id: Tournament ID (YYYYMM)
        day: Day number (1-15)
        mode: 'preview', 'results', or 'both'
    """
    print(f"Running daily pipeline for {basho_id} day {day} (mode: {mode})")

    if mode in ('preview', 'both'):
        generate_preview(basho_id, day)

    if mode in ('results', 'both') and day > 0:
        generate_results(basho_id, day)

    # Regenerate index page
    render_index_page(SITE_OUTPUT)


def main():
    parser = argparse.ArgumentParser(description='Generate sumo companion app content')
    parser.add_argument('--basho', required=True, help='Tournament ID (YYYYMM)')
    parser.add_argument('--day', type=int, required=True, help='Day number (1-15)')
    parser.add_argument('--mode', choices=['preview', 'results', 'both'], default='both',
                       help='Generation mode')
    parser.add_argument('--fetch', action='store_true', help='Fetch latest data from API')

    args = parser.parse_args()

    run_daily_pipeline(args.basho, args.day, args.mode)


if __name__ == '__main__':
    main()
