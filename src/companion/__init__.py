"""
Sumo Companion App - Daily previews and recaps for sumo tournaments.

Generates static HTML pages with bout predictions, interest rankings,
and spoiler-free viewing support.
"""

from .daily_pipeline import run_daily_pipeline, generate_preview, generate_results
from .interest_scoring import compute_interest_scores, rank_bouts
from .yusho_simulation import simulate_yusho_race

__all__ = [
    'run_daily_pipeline',
    'generate_preview',
    'generate_results',
    'compute_interest_scores',
    'rank_bouts',
    'simulate_yusho_race',
]
