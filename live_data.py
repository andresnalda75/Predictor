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


CREST_MAP = {
    "Arsenal":        "https://crests.football-data.org/57.png",
    "Aston Villa":    "https://crests.football-data.org/58.png",
    "Bournemouth":    "https://crests.football-data.org/1044.png",
    "Brentford":      "https://crests.football-data.org/402.png",
    "Brighton":       "https://crests.football-data.org/397.png",
    "Burnley":        "https://crests.football-data.org/328.png",
    "Chelsea":        "https://crests.football-data.org/61.png",
    "Crystal Palace": "https://crests.football-data.org/354.png",
    "Everton":        "https://crests.football-data.org/62.png",
    "Fulham":         "https://crests.football-data.org/63.png",
    "Leeds United":   "https://crests.football-data.org/341.png",
    "Liverpool":      "https://crests.football-data.org/64.png",
    "Man City":       "https://crests.football-data.org/65.png",
    "Man United":     "https://crests.football-data.org/66.png",
    "Newcastle":      "https://crests.football-data.org/67.png",
    "Nott'm Forest": "https://crests.football-data.org/351.png",
    "Sunderland":     "https://crests.football-data.org/71.png",
    "Tottenham":      "https://crests.football-data.org/73.png",
    "West Ham":       "https://crests.football-data.org/563.png",
    "Wolves":         "https://crests.football-data.org/76.png",
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
            "date":       pd.to_datetime(m["utcDate"]).replace(tzinfo=None),
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
    fixtures = []
    for m in r.json()["matches"]:
        home = TEAM_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
        away = TEAM_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
        fixtures.append({
            "date":       m["utcDate"][:10],
            "home_team":  home,
            "away_team":  away,
            "matchday":   m["matchday"],
            "home_crest": CREST_MAP.get(home, ""),
            "away_crest": CREST_MAP.get(away, ""),
        })
    return fixtures
