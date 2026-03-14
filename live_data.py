import os, requests, pandas as pd

API_KEY  = os.environ.get("FOOTBALL_DATA_API_KEY", "")
HEADERS  = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"

TEAM_MAP = {
    "Arsenal FC":                  "Arsenal",
    "Aston Villa FC":              "Aston Villa",
    "AFC Bournemouth":             "Bournemouth",
    "Brentford FC":                "Brentford",
    "Brighton & Hove Albion FC":   "Brighton",
    "Burnley FC":                  "Burnley",
    "Chelsea FC":                  "Chelsea",
    "Crystal Palace FC":           "Crystal Palace",
    "Everton FC":                  "Everton",
    "Fulham FC":                   "Fulham",
    "Ipswich Town FC":             "Ipswich",
    "Leeds United FC":             "Leeds United",
    "Leicester City FC":           "Leicester",
    "Liverpool FC":                "Liverpool",
    "Manchester City FC":          "Man City",
    "Manchester United FC":        "Man United",
    "Newcastle United FC":         "Newcastle",
    "Nottingham Forest FC":        "Nott'm Forest",
    "Southampton FC":              "Southampton",
    "Sunderland AFC":              "Sunderland",
    "Tottenham Hotspur FC":        "Tottenham",
    "West Ham United FC":          "West Ham",
    "Wolverhampton Wanderers FC":  "Wolves",
}

def fetch_current_season():
    r = requests.get(
        f"{BASE_URL}/competitions/PL/matches",
        headers=HEADERS,
        params={"status": "FINISHED"},
        timeout=15
    )
    r.raise_for_status()
    rows = []
    for m in r.json()["matches"]:
        hg = m["score"]["fullTime"]["home"]
        ag = m["score"]["fullTime"]["away"]
        if hg is None: continue
        rows.append({
            "date":       pd.to_datetime(m["utcDate"]),
            "home_team":  TEAM_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"]),
            "away_team":  TEAM_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"]),
            "home_goals": hg,
            "away_goals": ag,
            "hs": 0, "as_": 0, "hst": 0, "ast": 0,
            "result":     "H" if hg > ag else ("A" if ag > hg else "D"),
            "matchday":   m["matchday"],
            "elo_home":   1500,
            "elo_away":   1500,
        })
    return pd.DataFrame(rows)

def fetch_standings():
    r = requests.get(
        f"{BASE_URL}/competitions/PL/standings",
        headers=HEADERS,
        timeout=15
    )
    r.raise_for_status()
    table = r.json()["standings"][0]["table"]
    return {
        TEAM_MAP.get(row["team"]["name"], row["team"]["name"]): {
            "position": row["position"],
            "points":   row["points"],
            "gd":       row["goalDifference"],
        }
        for row in table
    }

def fetch_upcoming():
    r = requests.get(
        f"{BASE_URL}/competitions/PL/matches",
        headers=HEADERS,
        params={"status": "SCHEDULED"},
        timeout=15
    )
    r.raise_for_status()
    return [
        {
            "date":      m["utcDate"][:10],
            "home_team": TEAM_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"]),
            "away_team": TEAM_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"]),
            "matchday":  m["matchday"],
        }
        for m in r.json()["matches"]
    ]
