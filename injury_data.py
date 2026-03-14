import os, requests

API_KEY  = os.environ.get("APIFOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

# API-Football team names → our internal names (matching live_data.py TEAM_MAP values)
TEAM_MAP = {
    "Arsenal":                "Arsenal",
    "Aston Villa":            "Aston Villa",
    "Bournemouth":            "Bournemouth",
    "Brentford":              "Brentford",
    "Brighton":               "Brighton",
    "Burnley":                "Burnley",
    "Chelsea":                "Chelsea",
    "Crystal Palace":         "Crystal Palace",
    "Everton":                "Everton",
    "Fulham":                 "Fulham",
    "Ipswich":                "Ipswich",
    "Leeds":                  "Leeds United",
    "Leicester":              "Leicester",
    "Liverpool":              "Liverpool",
    "Manchester City":        "Man City",
    "Manchester United":      "Man United",
    "Newcastle":              "Newcastle",
    "Nottingham Forest":      "Nott'm Forest",
    "Southampton":            "Southampton",
    "Sunderland":             "Sunderland",
    "Tottenham":              "Tottenham",
    "West Ham":               "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves":                 "Wolves",
}

# Cache: {team_name: injury_count}
_injury_cache: dict[str, int] = {}


def fetch_injuries(season: int = 2025) -> dict[str, int]:
    """Fetch current EPL injuries from API-Football. Returns {team: count}."""
    if not API_KEY:
        return {}
    r = requests.get(
        f"{BASE_URL}/injuries",
        headers=HEADERS,
        params={"league": 39, "season": season},
        timeout=15,
    )
    r.raise_for_status()
    counts: dict[str, int] = {}
    for entry in r.json().get("response", []):
        raw_name = entry.get("team", {}).get("name", "")
        team = TEAM_MAP.get(raw_name, raw_name)
        counts[team] = counts.get(team, 0) + 1
    return counts


def load_injuries() -> dict[str, int]:
    """Fetch and cache injuries. Returns cached data on failure."""
    global _injury_cache
    try:
        _injury_cache = fetch_injuries()
    except Exception as e:
        print(f"⚠️ Injury data failed: {e}")
    return _injury_cache


def get_injury_count(team: str) -> int:
    """Return number of currently injured players for a team."""
    return _injury_cache.get(team, 0)
