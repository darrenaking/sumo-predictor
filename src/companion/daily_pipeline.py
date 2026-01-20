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
    fetch_all_rikishi,
    parse_banzuke_rank,
    make_request,
)
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
    """
    data = fetch_basho_torikumi(basho_id, division, day)

    if not data or 'torikumi' not in data:
        print(f"No matchups found for {basho_id} day {day}")
        return pd.DataFrame()

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
            'winnerId': bout.get('winnerId'),  # None if not yet fought
            'kimarite': bout.get('kimarite'),  # None if not yet fought
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

    # Build bout data for template
    bout_data = []
    for idx in ranked_interest['bout_idx']:
        bout = bouts_df.iloc[idx]
        pred = predictions_df.iloc[idx]
        interest = interest_df[interest_df['bout_idx'] == idx].iloc[0]
        features = features_list[idx]

        p_east = pred['pred_east_win_prob']

        # Determine favorite
        if p_east >= 0.5:
            favorite = bout['eastShikona']
            favorite_prob = p_east
        else:
            favorite = bout['westShikona']
            favorite_prob = 1 - p_east

        # Format prediction string
        if favorite_prob >= 0.65:
            pred_str = f"{favorite} favored ({favorite_prob*100:.0f}%)"
        elif favorite_prob >= 0.55:
            pred_str = f"{favorite} slight edge ({favorite_prob*100:.0f}%)"
        else:
            pred_str = f"Coin flip ({p_east*100:.0f}-{(1-p_east)*100:.0f})"

        bout_data.append({
            'bout_number': bout['bout_number'],
            'east_name': bout['eastShikona'],
            'west_name': bout['westShikona'],
            'east_rank': bout['eastRank'],
            'west_rank': bout['westRank'],
            'prediction': pred_str,
            'p_east': p_east,
            'interest_score': interest['interest_score'],
            'interest_label': get_interest_label(interest['interest_score']),
            'interest_reasons': interest['interest_reasons'],
            'is_bout_of_day': idx == bout_of_day_idx,
            'features': features,
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
