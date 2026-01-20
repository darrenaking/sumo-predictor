"""
Monte Carlo simulation for yusho (tournament championship) predictions.

Simulates remaining tournament outcomes to estimate:
- Probability of each wrestler winning the yusho
- Expected final record distributions
- Playoff scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class YushoContender:
    """Yusho contender with probability and expected record."""
    wrestler_id: int
    wrestler_name: str
    current_wins: int
    current_losses: int
    yusho_probability: float
    expected_wins: float
    wins_std: float
    magic_number: Optional[int]  # Wins needed + leader losses to clinch


@dataclass
class YushoRaceResult:
    """Complete yusho race simulation results."""
    contenders: List[YushoContender]
    leader_wins: int
    days_remaining: int
    playoff_probability: float
    simulation_count: int


def get_current_standings(
    results_df: pd.DataFrame,
    basho_id: str,
    through_day: int
) -> pd.DataFrame:
    """
    Get current tournament standings through a given day.

    Args:
        results_df: DataFrame with match results
        basho_id: Tournament ID (YYYYMM)
        through_day: Calculate standings through this day

    Returns:
        DataFrame with wrestler_id, wins, losses sorted by wins descending
    """
    # Filter to this basho and through the specified day
    basho_results = results_df[
        (results_df['bashoId'] == basho_id) &
        (results_df['day'] <= through_day)
    ]

    if basho_results.empty:
        return pd.DataFrame(columns=['wrestler_id', 'wins', 'losses'])

    # Count wins for each wrestler
    wins = defaultdict(int)
    losses = defaultdict(int)

    for _, row in basho_results.iterrows():
        winner_id = row['winnerId']
        east_id = row['eastId']
        west_id = row['westId']

        if winner_id == east_id:
            wins[east_id] += 1
            losses[west_id] += 1
        else:
            wins[west_id] += 1
            losses[east_id] += 1

    # Build standings
    all_wrestlers = set(wins.keys()) | set(losses.keys())
    standings = []

    for w_id in all_wrestlers:
        standings.append({
            'wrestler_id': w_id,
            'wins': wins[w_id],
            'losses': losses[w_id],
        })

    standings_df = pd.DataFrame(standings)
    return standings_df.sort_values('wins', ascending=False).reset_index(drop=True)


def get_remaining_schedule(
    schedule_df: pd.DataFrame,
    basho_id: str,
    from_day: int
) -> pd.DataFrame:
    """
    Get remaining bouts in the tournament.

    Args:
        schedule_df: DataFrame with scheduled bouts
        basho_id: Tournament ID
        from_day: Start from this day (inclusive)

    Returns:
        DataFrame with remaining bouts
    """
    return schedule_df[
        (schedule_df['bashoId'] == basho_id) &
        (schedule_df['day'] >= from_day)
    ].copy()


def simulate_yusho_race(
    current_standings: pd.DataFrame,
    remaining_bouts: pd.DataFrame,
    bout_predictions: Dict[Tuple[int, int], float],
    wrestler_names: Dict[int, str],
    num_simulations: int = 10000,
    top_n: int = 10
) -> YushoRaceResult:
    """
    Run Monte Carlo simulation for yusho predictions.

    Args:
        current_standings: DataFrame with wrestler_id, wins, losses
        remaining_bouts: DataFrame with eastId, westId, day for remaining bouts
        bout_predictions: Dict mapping (eastId, westId) -> P(east wins)
        wrestler_names: Dict mapping wrestler_id -> name
        num_simulations: Number of simulations to run
        top_n: Number of top contenders to return

    Returns:
        YushoRaceResult with contender probabilities
    """
    if current_standings.empty:
        return YushoRaceResult(
            contenders=[],
            leader_wins=0,
            days_remaining=15,
            playoff_probability=0.0,
            simulation_count=num_simulations
        )

    # Initialize tracking
    wrestler_ids = set(current_standings['wrestler_id'].tolist())
    initial_wins = current_standings.set_index('wrestler_id')['wins'].to_dict()
    initial_losses = current_standings.set_index('wrestler_id')['losses'].to_dict()

    leader_wins = current_standings['wins'].max()
    days_remaining = 15 - (leader_wins + current_standings['losses'].min())

    # Prepare bout list
    bouts_list = []
    for _, row in remaining_bouts.iterrows():
        east_id = row['eastId']
        west_id = row['westId']

        # Get prediction probability
        p_east = bout_predictions.get((east_id, west_id), 0.5)

        bouts_list.append({
            'east_id': east_id,
            'west_id': west_id,
            'p_east': p_east,
        })

    # Run simulations
    yusho_counts = defaultdict(float)
    final_wins = defaultdict(list)
    playoff_count = 0

    rng = np.random.default_rng(seed=42)

    for _ in range(num_simulations):
        sim_wins = initial_wins.copy()

        # Simulate each remaining bout
        for bout in bouts_list:
            if rng.random() < bout['p_east']:
                sim_wins[bout['east_id']] = sim_wins.get(bout['east_id'], 0) + 1
            else:
                sim_wins[bout['west_id']] = sim_wins.get(bout['west_id'], 0) + 1

        # Determine winner(s)
        max_wins = max(sim_wins.values()) if sim_wins else 0
        winners = [w for w, wins in sim_wins.items() if wins == max_wins]

        if len(winners) > 1:
            playoff_count += 1
            # In playoff, split credit (simplified - real playoffs have brackets)
            for winner in winners:
                yusho_counts[winner] += 1.0 / len(winners)
        else:
            yusho_counts[winners[0]] += 1.0

        # Track final wins
        for w_id, wins in sim_wins.items():
            final_wins[w_id].append(wins)

    # Build contenders list
    contenders = []

    for w_id in wrestler_ids:
        yusho_prob = yusho_counts.get(w_id, 0) / num_simulations * 100
        wins_list = final_wins.get(w_id, [initial_wins.get(w_id, 0)])

        expected_wins = np.mean(wins_list)
        wins_std = np.std(wins_list)

        current_w = initial_wins.get(w_id, 0)

        # Magic number: wins needed assuming leader loses all remaining
        # Magic number = (15 - current_wins) - (days_remaining)
        # Simplified: wins to guarantee at least tie with current leader
        magic = None
        if days_remaining > 0:
            wins_needed = leader_wins + 1 - current_w
            if wins_needed > 0:
                magic = wins_needed

        contenders.append(YushoContender(
            wrestler_id=w_id,
            wrestler_name=wrestler_names.get(w_id, f"Wrestler {w_id}"),
            current_wins=current_w,
            current_losses=initial_losses.get(w_id, 0),
            yusho_probability=yusho_prob,
            expected_wins=expected_wins,
            wins_std=wins_std,
            magic_number=magic,
        ))

    # Sort by yusho probability
    contenders.sort(key=lambda x: x.yusho_probability, reverse=True)

    return YushoRaceResult(
        contenders=contenders[:top_n],
        leader_wins=leader_wins,
        days_remaining=days_remaining,
        playoff_probability=playoff_count / num_simulations * 100,
        simulation_count=num_simulations,
    )


def format_yusho_race(result: YushoRaceResult) -> str:
    """Format yusho race results as text."""
    lines = []
    lines.append(f"Yusho Race (after Day {15 - result.days_remaining})")
    lines.append(f"Leader: {result.leader_wins} wins")
    lines.append("")

    for c in result.contenders:
        if c.yusho_probability < 0.1:
            continue

        record = f"{c.current_wins}-{c.current_losses}"
        prob = f"{c.yusho_probability:.1f}%"

        magic_str = ""
        if c.magic_number and c.magic_number <= result.days_remaining:
            magic_str = f" (magic #{c.magic_number})"

        lines.append(f"  {c.wrestler_name}: {record} - {prob}{magic_str}")

    if result.playoff_probability > 5:
        lines.append(f"\nPlayoff probability: {result.playoff_probability:.1f}%")

    return "\n".join(lines)
