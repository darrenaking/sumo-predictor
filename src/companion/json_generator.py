"""
JSON data generation for sumo companion app.

Generates JSON files that are loaded by the client-side JavaScript app.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import numpy as np

from src.companion.yusho_simulation import YushoRaceResult


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# Basho month names
BASHO_NAMES = {
    '01': 'Hatsu Basho (January)',
    '03': 'Haru Basho (March)',
    '05': 'Natsu Basho (May)',
    '07': 'Nagoya Basho (July)',
    '09': 'Aki Basho (September)',
    '11': 'Kyushu Basho (November)',
}


def get_basho_name(basho_id: str) -> str:
    """Get human-readable basho name."""
    year = basho_id[:4]
    month = basho_id[4:]
    name = BASHO_NAMES.get(month, f'Basho {month}')
    return f"{year} {name}"


def format_basho_dir(basho_id: str) -> str:
    """Format basho ID as directory name (YYYY-MM)."""
    return f"{basho_id[:4]}-{basho_id[4:]}"


def generate_preview_json(
    basho_id: str,
    day: int,
    bouts: List[Dict],
    yusho_race: Optional[YushoRaceResult],
    name_lookup: Dict[int, str],
) -> Dict:
    """
    Generate JSON data for a daily preview.

    Args:
        basho_id: Tournament ID
        day: Day number
        bouts: List of bout data dicts
        yusho_race: Yusho race simulation results
        name_lookup: Wrestler ID to name mapping

    Returns:
        Dict ready for JSON serialization
    """
    # Format yusho race data
    yusho_data = None
    if yusho_race and yusho_race.contenders:
        yusho_data = {
            'leader_wins': yusho_race.leader_wins,
            'days_remaining': yusho_race.days_remaining,
            'playoff_probability': yusho_race.playoff_probability,
            'contenders': [
                {
                    'name': c.wrestler_name,
                    'record': f"{c.current_wins}-{c.current_losses}",
                    'magic_number': c.magic_number,
                }
                for c in yusho_race.contenders
            ]
        }

    # Clean bout data for JSON (remove internal fields)
    clean_bouts = []
    for bout in bouts:
        clean_bout = {
            'bout_number': bout['bout_number'],
            'east_name': bout['east_name'],
            'west_name': bout['west_name'],
            'east_rank': bout['east_rank'],
            'west_rank': bout['west_rank'],
            'p_east': bout['p_east'],
            'is_bout_of_day': bout.get('is_bout_of_day', False),
            'is_must_watch': bout.get('is_must_watch', False),
            'is_belt_battle': bout.get('is_belt_battle', False),
            'is_pushing_match': bout.get('is_pushing_match', False),
            'east_form': bout.get('east_form'),
            'west_form': bout.get('west_form'),
            'east_style': bout.get('east_style'),
            'west_style': bout.get('west_style'),
            'storylines': bout.get('storylines', []),
            'h2h_east_wins': bout.get('h2h_east_wins', 0),
            'h2h_west_wins': bout.get('h2h_west_wins', 0),
            'h2h_total': bout.get('h2h_total', 0),
        }
        clean_bouts.append(clean_bout)

    return {
        'basho_id': basho_id,
        'basho_name': get_basho_name(basho_id),
        'day': day,
        'bouts': clean_bouts,
        'yusho_race': yusho_data,
        'generated_at': datetime.now().isoformat(),
    }


def generate_results_json(
    basho_id: str,
    day: int,
    results: List[Dict],
    upsets: List[Dict],
    accuracy: float,
    standings: Any,  # DataFrame
    name_lookup: Dict[int, str],
) -> Dict:
    """
    Generate JSON data for daily results.

    Args:
        basho_id: Tournament ID
        day: Day number
        results: List of result data dicts
        upsets: List of upset bouts
        accuracy: Prediction accuracy percentage
        standings: Current standings DataFrame
        name_lookup: Wrestler ID to name mapping

    Returns:
        Dict ready for JSON serialization
    """
    # Format standings
    standings_data = []
    if standings is not None and not standings.empty:
        for _, row in standings.head(15).iterrows():
            w_id = row['wrestler_id']
            standings_data.append({
                'name': name_lookup.get(w_id, f"Wrestler {w_id}"),
                'wins': int(row['wins']),
                'losses': int(row['losses']),
            })

    # Clean results data
    clean_results = []
    for r in results:
        clean_results.append({
            'bout_number': r['bout_number'],
            'east_name': r['east_name'],
            'west_name': r['west_name'],
            'east_rank': r['east_rank'],
            'west_rank': r['west_rank'],
            'winner': r['winner'],
            'kimarite': r.get('kimarite'),
            'predicted_correct': r['predicted_correct'],
        })

    # Clean upsets data
    clean_upsets = []
    for u in upsets:
        clean_upsets.append({
            'east_name': u['east_name'],
            'west_name': u['west_name'],
            'winner': u['winner'],
            'kimarite': u.get('kimarite'),
            'upset_magnitude': u['upset_magnitude'],
        })

    return {
        'basho_id': basho_id,
        'basho_name': get_basho_name(basho_id),
        'day': day,
        'results': clean_results,
        'upsets': clean_upsets,
        'accuracy': accuracy,
        'standings': standings_data,
        'generated_at': datetime.now().isoformat(),
    }


def write_day_json(data: Dict, output_dir: Path, basho_id: str, day: int):
    """
    Write combined day data (preview + results) to a single JSON file.

    Args:
        data: Combined data dict
        output_dir: Site data output directory
        basho_id: Tournament ID
        day: Day number
    """
    basho_dir = format_basho_dir(basho_id)
    day_dir = output_dir / "data" / basho_dir
    day_dir.mkdir(parents=True, exist_ok=True)

    output_file = day_dir / f"day-{day:02d}.json"
    # Convert numpy types before serialization
    clean_data = convert_numpy_types(data)
    with open(output_file, 'w') as f:
        json.dump(clean_data, f, indent=2)

    print(f"Written: {output_file}")


def generate_index_json(output_dir: Path) -> Dict:
    """
    Generate tournament index JSON by scanning data directory.

    Args:
        output_dir: Site output directory

    Returns:
        List of tournament data dicts
    """
    data_dir = output_dir / "data"
    if not data_dir.exists():
        return []

    tournaments = []
    for subdir in sorted(data_dir.iterdir(), reverse=True):
        if subdir.is_dir() and '-' in subdir.name:
            try:
                year, month = subdir.name.split('-')
                basho_id = f"{year}{month}"

                # Check which days have data
                days_available = []
                for day in range(1, 16):
                    day_str = f"{day:02d}"
                    day_file = subdir / f"day-{day_str}.json"

                    if day_file.exists():
                        # Read to check what data is available
                        with open(day_file) as f:
                            day_data = json.load(f)

                        has_preview = 'bouts' in day_data and len(day_data.get('bouts', [])) > 0
                        has_results = 'results' in day_data and len(day_data.get('results', [])) > 0
                    else:
                        has_preview = False
                        has_results = False

                    days_available.append({
                        'day': day,
                        'has_preview': has_preview,
                        'has_results': has_results,
                    })

                tournaments.append({
                    'id': basho_id,
                    'dir': subdir.name,
                    'name': get_basho_name(basho_id),
                    'days': days_available,
                })
            except ValueError:
                continue

    # Write index.json
    index_file = data_dir / "index.json"
    with open(index_file, 'w') as f:
        json.dump(tournaments, f, indent=2, cls=NumpyEncoder)
    print(f"Written: {index_file}")

    return tournaments
