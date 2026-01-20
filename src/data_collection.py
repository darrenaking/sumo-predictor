"""
Data collection utilities for sumo-api.com

API Endpoints:
- /api/rikishis — all wrestler profiles
- /api/rikishi/:id — individual wrestler details
- /api/rikishi/:id/matches — full match history
- /api/rikishi/:id/matches/:opponentId — head-to-head records
- /api/basho/:bashoId — tournament info
- /api/basho/:bashoId/torikumi/:division/:day — daily matchups
- /api/kimarite — winning technique reference
"""

import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime


BASE_URL = "https://sumo-api.com/api"

# Rate limiting settings
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds to wait on rate limit


def make_request(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Make a request to the sumo-api.com API with retry logic."""
    url = f"{BASE_URL}{endpoint}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                time.sleep(REQUEST_DELAY)
                return response.json()
            elif response.status_code == 429:  # Rate limited
                print(f"Rate limited, waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            elif response.status_code == 404:
                return None
            else:
                print(f"Error {response.status_code} for {endpoint}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}): {e}")
            time.sleep(RETRY_DELAY)

    return None


def fetch_all_rikishi() -> pd.DataFrame:
    """Fetch all wrestler profiles."""
    print("Fetching all rikishi...")

    all_rikishi = []
    skip = 0
    limit = 1000  # API default limit

    while True:
        data = make_request("/rikishis", params={"skip": skip, "limit": limit})

        if not data or "records" not in data:
            break

        records = data["records"]
        if not records:
            break

        all_rikishi.extend(records)
        print(f"  Fetched {len(all_rikishi)} rikishi...")

        if len(records) < limit:
            break

        skip += limit

    print(f"Total rikishi fetched: {len(all_rikishi)}")
    return pd.DataFrame(all_rikishi)


def fetch_rikishi_details(rikishi_id: int) -> Optional[Dict]:
    """Fetch detailed info for a specific wrestler."""
    return make_request(f"/rikishi/{rikishi_id}")


def fetch_head_to_head(rikishi_id: int, opponent_id: int) -> Dict:
    """
    Fetch head-to-head record between two wrestlers.

    Returns dict with 'east_wins' and 'west_wins' where east is rikishi_id.
    """
    data = make_request(f"/rikishi/{rikishi_id}/matches/{opponent_id}")

    if not data:
        return {'east_wins': 0, 'west_wins': 0, 'total': 0}

    matches = data.get('matches')
    if not matches:
        return {'east_wins': 0, 'west_wins': 0, 'total': 0}

    east_wins = sum(1 for m in matches if m.get('winnerId') == rikishi_id)
    west_wins = len(matches) - east_wins

    return {
        'east_wins': east_wins,
        'west_wins': west_wins,
        'total': len(matches)
    }


def fetch_rikishi_matches(rikishi_id: int) -> pd.DataFrame:
    """Fetch all matches for a specific wrestler."""
    all_matches = []
    skip = 0
    limit = 1000

    while True:
        data = make_request(f"/rikishi/{rikishi_id}/matches",
                           params={"skip": skip, "limit": limit})

        if not data or "records" not in data:
            break

        records = data["records"]
        if not records:
            break

        all_matches.extend(records)

        if len(records) < limit:
            break

        skip += limit

    return pd.DataFrame(all_matches) if all_matches else pd.DataFrame()


def fetch_basho_info(basho_id: str) -> Optional[Dict]:
    """
    Fetch tournament information.

    basho_id format: YYYYMM (e.g., "202401" for January 2024)
    """
    return make_request(f"/basho/{basho_id}")


def fetch_basho_torikumi(basho_id: str, division: str = "Makuuchi", day: int = 1) -> Optional[Dict]:
    """
    Fetch daily matchups for a tournament.

    division: Makuuchi, Juryo, Makushita, Sandanme, Jonidan, Jonokuchi
    day: 1-15
    """
    return make_request(f"/basho/{basho_id}/torikumi/{division}/{day}")


def fetch_kimarite_reference() -> pd.DataFrame:
    """Fetch the kimarite (winning technique) reference."""
    data = make_request("/kimarite")
    if data and "records" in data:
        return pd.DataFrame(data["records"])
    return pd.DataFrame()


def fetch_basho_banzuke(basho_id: str, division: str = "Makuuchi") -> Optional[Dict]:
    """
    Fetch banzuke (rankings) for a tournament with embedded match records.

    This is an alternative to torikumi endpoint that includes all matchups
    and results in the wrestler's record field.
    """
    return make_request(f"/basho/{basho_id}/banzuke/{division}")


def fetch_torikumi_from_banzuke(basho_id: str, day: int, division: str = "Makuuchi") -> List[Dict]:
    """
    Extract torikumi (matchups) for a specific day from banzuke data.

    This is a fallback when /torikumi endpoint returns 404.
    Returns list of bout dicts with eastId, westId, eastShikona, westShikona,
    winnerId (if completed), kimarite (if completed).
    """
    banzuke = fetch_basho_banzuke(basho_id, division)
    if not banzuke:
        return []

    # Build wrestler lookup from banzuke
    wrestlers = {}
    for side in ['east', 'west']:
        for w in banzuke.get(side, []):
            wrestlers[w['rikishiID']] = {
                'id': w['rikishiID'],
                'shikona': w['shikonaEn'],
                'rank': w['rank'],
                'record': w.get('record', []),
                'wins': w.get('wins', 0),
                'losses': w.get('losses', 0),
            }

    # Extract matchups for the given day (0-indexed in record array)
    day_idx = day - 1
    bouts = []
    seen_pairs = set()

    for w_id, w_data in wrestlers.items():
        if day_idx >= len(w_data['record']):
            continue

        bout_record = w_data['record'][day_idx]
        opponent_id = bout_record.get('opponentID')

        if not opponent_id or opponent_id not in wrestlers:
            continue

        # Avoid duplicate bouts (A vs B and B vs A)
        pair = tuple(sorted([w_id, opponent_id]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        opponent = wrestlers[opponent_id]
        result = bout_record.get('result', '')
        kimarite = bout_record.get('kimarite', '')

        # Determine winner if bout completed
        winner_id = None
        if result == 'win':
            winner_id = w_id
        elif result == 'loss':
            winner_id = opponent_id

        # Use higher rank as east (traditional ordering)
        # Lower rankValue = higher rank
        w_rank_val = wrestlers[w_id].get('rankValue', 999) if 'rankValue' not in w_data else 999
        o_rank_val = wrestlers[opponent_id].get('rankValue', 999)

        # Determine east/west based on rank or just use current wrestler as east
        east_id, west_id = w_id, opponent_id

        bouts.append({
            'eastId': east_id,
            'westId': west_id,
            'eastShikona': wrestlers[east_id]['shikona'],
            'westShikona': wrestlers[west_id]['shikona'],
            'eastRank': wrestlers[east_id]['rank'],
            'westRank': wrestlers[west_id]['rank'],
            'winnerId': winner_id,
            'kimarite': kimarite if winner_id else None,
            'day': day,
            'bashoId': basho_id,
        })

    return bouts


def fetch_current_basho_standings(basho_id: str, division: str = "Makuuchi") -> pd.DataFrame:
    """
    Fetch current tournament standings from banzuke.

    Returns DataFrame with wrestler_id, name, rank, wins, losses.
    """
    banzuke = fetch_basho_banzuke(basho_id, division)
    if not banzuke:
        return pd.DataFrame()

    standings = []
    for side in ['east', 'west']:
        for w in banzuke.get(side, []):
            standings.append({
                'wrestler_id': w['rikishiID'],
                'name': w['shikonaEn'],
                'rank': w['rank'],
                'wins': w.get('wins', 0),
                'losses': w.get('losses', 0),
                'absences': w.get('absences', 0),
            })

    df = pd.DataFrame(standings)
    if not df.empty:
        df = df.sort_values('wins', ascending=False)
    return df


def generate_basho_ids(start_year: int = 1958, end_year: int = 2025) -> List[str]:
    """
    Generate all valid basho IDs.

    Basho are held in January (01), March (03), May (05),
    July (07), September (09), November (11).
    """
    basho_months = ["01", "03", "05", "07", "09", "11"]
    basho_ids = []

    for year in range(start_year, end_year + 1):
        for month in basho_months:
            basho_ids.append(f"{year}{month}")

    return basho_ids


def fetch_all_matches_for_basho(basho_id: str, division: str = "Makuuchi") -> pd.DataFrame:
    """Fetch all matches for a tournament in a division."""
    all_matches = []

    for day in range(1, 16):
        data = fetch_basho_torikumi(basho_id, division, day)
        if data and "torikumi" in data:
            for match in data["torikumi"]:
                match["basho_id"] = basho_id
                match["day"] = day
                match["division"] = division
                all_matches.append(match)

    return pd.DataFrame(all_matches) if all_matches else pd.DataFrame()


# Kimarite categories mapping
KIMARITE_CATEGORIES = {
    "push": [
        "oshidashi",    # push out
        "tsukidashi",   # thrust out
        "oshitaoshi",   # push down
        "tsukiotoshi",  # thrust down
        "tsukitaoshi",  # thrust and push down
        "okuridashi",   # rear push out
        "abisetaoshi",  # backwards body drop
    ],
    "grapple": [
        "yorikiri",     # force out (most common)
        "uwatenage",    # overarm throw
        "shitatenage",  # underarm throw
        "sukuinage",    # scoop throw
        "kotenage",     # arm lock throw
        "kubinage",     # headlock throw
        "yoritaoshi",   # frontal force down
        "uwatedashinage", # pulling overarm throw
        "shitatedashinage", # pulling underarm throw
        "kakenage",     # hooking inner thigh throw
        "kirikaeshi",   # twisting backward knee trip
        "tsukaminage",  # lifting throw
        "tsuridashi",   # lift out
        "tsuriotoshi",  # lift and drop
        "utchari",      # backward pivot throw
        "sotogake",     # outside leg trip
        "uchigake",     # inside leg trip
        "kimedashi",    # arm bar force out
        "kimekiri",     # arm bar force down
        "katasukashi",  # under-shoulder swing down
        "okurinage",    # rear throw
        "okuritaoshi",  # rear push down
        "okurihineri",  # rear twist down
        "okuridashi",   # rear push out
        "okuritsuridashi", # rear lift out
        "amiuchi",      # fisherman's throw
        "sabaori",      # forward force down
        "waridashi",    # split-thigh push out
        "makiotoshi",   # twist down
        "uwatehineri",  # overarm twist down
        "shitatehineri", # underarm twist down
    ],
    "evasion": [
        "hatakikomi",   # slap down
        "hikiotoshi",   # hand pull down
        "hikkake",      # arm grabbing force out
        "okuritaoshi",  # rear push down
        "ketaguri",     # ankle sweep
        "kekaeshi",     # minor inner foot sweep
        "ashitori",     # leg pick
        "tsumadori",    # ankle grab
        "sotogake",     # outside leg trip
        "uchigake",     # inside leg trip
        "chongake",     # ankle hook
        "kawazugake",   # hooking backward counter throw
        "komatasukui",  # over-thigh scooping throw
        "okuritaoshi",  # rear push down
        "tottari",      # arm bar throw
        "izori",        # backward force down
        "shumokuzori",  # bell hammer backward force down
        "tasukizori",   # backward belt throw
        "nichonage",    # two-handed throw
    ],
}


def categorize_kimarite(kimarite: str) -> str:
    """Categorize a kimarite into push/grapple/evasion."""
    kimarite_lower = kimarite.lower() if kimarite else ""

    for category, techniques in KIMARITE_CATEGORIES.items():
        if kimarite_lower in techniques:
            return category

    # Default to grapple for unknown techniques
    return "grapple"


def parse_banzuke_rank(rank_str: str) -> int:
    """
    Convert banzuke rank string to numeric value.

    Lower number = higher rank.
    Y1e=1, Y1w=2, Y2e=3, Y2w=4, O1e=5, O1w=6, ...

    Ranks: Y (Yokozuna), O (Ozeki), S (Sekiwake), K (Komusubi),
           M (Maegashira), J (Juryo)
    """
    if not rank_str:
        return 999

    rank_str = rank_str.strip().upper()

    # Rank prefixes and their base values
    rank_bases = {
        "Y": 0,      # Yokozuna
        "O": 10,     # Ozeki
        "S": 30,     # Sekiwake
        "K": 40,     # Komusubi
        "M": 50,     # Maegashira (1-17ish)
        "J": 100,    # Juryo (1-14)
    }

    # Extract components
    import re
    match = re.match(r"([YOSKM]|J)(\d+)?([EW])?", rank_str)

    if not match:
        return 999

    rank_letter = match.group(1)
    rank_num = int(match.group(2)) if match.group(2) else 1
    direction = match.group(3) if match.group(3) else "E"

    base = rank_bases.get(rank_letter, 999)

    # Each rank number adds 2 (for east and west)
    numeric = base + (rank_num - 1) * 2

    # West is one higher (lower rank) than East
    if direction == "W":
        numeric += 1

    return numeric


def save_data(df: pd.DataFrame, output_path: str, filename: str):
    """Save DataFrame to parquet format."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    filepath = path / f"{filename}.parquet"
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")


def load_data(input_path: str, filename: str) -> pd.DataFrame:
    """Load DataFrame from parquet format."""
    filepath = Path(input_path) / f"{filename}.parquet"
    return pd.read_parquet(filepath)


# For Kaggle environment detection
def is_kaggle() -> bool:
    """Check if running in Kaggle environment."""
    import os
    return os.path.exists("/kaggle/input")


def get_input_path() -> str:
    """Get the appropriate input path based on environment."""
    if is_kaggle():
        return "/kaggle/input/sumo-data"
    return "./data"


def get_output_path() -> str:
    """Get the appropriate output path based on environment."""
    if is_kaggle():
        return "/kaggle/working"
    return "./output"
