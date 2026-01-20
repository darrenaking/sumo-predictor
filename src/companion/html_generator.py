"""
HTML generation for sumo companion app using Jinja2 templates.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.companion.yusho_simulation import YushoRaceResult


# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Set up Jinja2 environment
env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml']),
)


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


def render_preview_page(
    basho_id: str,
    day: int,
    bouts: List[Dict],
    yusho_race: Optional[YushoRaceResult],
    name_lookup: Dict[int, str],
) -> str:
    """
    Render the daily preview page.

    Args:
        basho_id: Tournament ID
        day: Day number
        bouts: List of bout data dicts (already ranked by interest)
        yusho_race: Yusho race simulation results
        name_lookup: Wrestler ID to name mapping

    Returns:
        Rendered HTML string
    """
    template = env.get_template('day_preview.html')

    # Find bout of the day
    bout_of_day = next((b for b in bouts if b.get('is_bout_of_day')), bouts[0] if bouts else None)

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
                    'probability': f"{c.yusho_probability:.1f}%",
                    'magic_number': c.magic_number,
                }
                for c in yusho_race.contenders
                if c.yusho_probability >= 0.5
            ]
        }

    return template.render(
        basho_id=basho_id,
        basho_name=get_basho_name(basho_id),
        day=day,
        bout_of_day=bout_of_day,
        bouts=bouts,
        yusho_race=yusho_data,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )


def render_results_page(
    basho_id: str,
    day: int,
    results: List[Dict],
    upsets: List[Dict],
    accuracy: float,
    standings: Any,  # DataFrame
    name_lookup: Dict[int, str],
) -> str:
    """
    Render the daily results page.

    Args:
        basho_id: Tournament ID
        day: Day number
        results: List of result data dicts
        upsets: List of upset bouts
        accuracy: Prediction accuracy percentage
        standings: Current standings DataFrame
        name_lookup: Wrestler ID to name mapping

    Returns:
        Rendered HTML string
    """
    template = env.get_template('day_results.html')

    # Format standings
    standings_data = []
    if standings is not None and not standings.empty:
        for _, row in standings.head(15).iterrows():
            w_id = row['wrestler_id']
            standings_data.append({
                'name': name_lookup.get(w_id, f"Wrestler {w_id}"),
                'wins': row['wins'],
                'losses': row['losses'],
            })

    return template.render(
        basho_id=basho_id,
        basho_name=get_basho_name(basho_id),
        day=day,
        results=results,
        upsets=upsets,
        accuracy=accuracy,
        standings=standings_data,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )


def render_index_page(output_dir: Path) -> str:
    """
    Render the main index page with tournament selector.

    Args:
        output_dir: Site output directory

    Returns:
        Rendered HTML string
    """
    template = env.get_template('index.html')

    # Find available tournaments
    tournaments = []
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir() and '-' in subdir.name:
            # Parse YYYY-MM format
            try:
                year, month = subdir.name.split('-')
                basho_id = f"{year}{month}"
                tournaments.append({
                    'id': basho_id,
                    'dir': subdir.name,
                    'name': get_basho_name(basho_id),
                })
            except ValueError:
                continue

    html = template.render(
        tournaments=tournaments,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )

    # Write index
    index_path = output_dir / 'index.html'
    index_path.write_text(html)
    print(f"Written: {index_path}")

    return html


def render_wrestler_profile(
    wrestler_id: int,
    wrestler_data: Dict,
    career_stats: Dict,
    recent_bouts: List[Dict],
    style_breakdown: Dict,
) -> str:
    """
    Render a wrestler profile page.

    Args:
        wrestler_id: Wrestler ID
        wrestler_data: Basic wrestler info
        career_stats: Career statistics
        recent_bouts: Recent bout history
        style_breakdown: Style analysis (push/grapple/evasion percentages)

    Returns:
        Rendered HTML string
    """
    template = env.get_template('wrestler_profile.html')

    return template.render(
        wrestler=wrestler_data,
        stats=career_stats,
        recent_bouts=recent_bouts,
        style=style_breakdown,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
    )
