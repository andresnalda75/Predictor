"""
Reconcile predictions — find rows where actual_outcome is NULL and match_date
is in the past, fetch results from football-data.org, update the database.

Usage:
    python scripts/reconcile_predictions.py

Requires FOOTBALL_DATA_API_KEY environment variable.
"""
import os
import sys
import sqlite3
import requests
from datetime import datetime, timedelta

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
HEADERS = {"X-Auth-Token": API_KEY}
BASE_URL = "https://api.football-data.org/v4"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions.db")

TEAM_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds United",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def fetch_finished_matches():
    """Fetch all finished EPL matches for the current season."""
    if not API_KEY:
        print("ERROR: FOOTBALL_DATA_API_KEY not set")
        sys.exit(1)
    r = requests.get(
        f"{BASE_URL}/competitions/PL/matches",
        headers=HEADERS,
        params={"status": "FINISHED"},
        timeout=15,
    )
    r.raise_for_status()
    results = {}
    for m in r.json()["matches"]:
        hg = m["score"]["fullTime"]["home"]
        ag = m["score"]["fullTime"]["away"]
        if hg is None:
            continue
        home = TEAM_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
        away = TEAM_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
        date = m["utcDate"][:10]
        result = "H" if hg > ag else ("A" if ag > hg else "D")
        results[(date, home, away)] = result
    return results


def reconcile():
    """Find unresolved predictions and update with actual results."""
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Find predictions with no actual outcome where match date has passed
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cur.execute(
        "SELECT id, match_date, home_team, away_team FROM predictions "
        "WHERE actual_outcome IS NULL AND match_date < ?",
        (today,),
    )
    pending = cur.fetchall()

    if not pending:
        print("No predictions to reconcile.")
        conn.close()
        return

    print(f"Found {len(pending)} unresolved prediction(s). Fetching results...")
    finished = fetch_finished_matches()

    updated = 0
    for row_id, match_date, home, away in pending:
        key = (match_date, home, away)
        actual = finished.get(key)
        if actual is None:
            print(f"  ⏳ {match_date} {home} vs {away} — result not yet available")
            continue
        cur.execute(
            "SELECT predicted_outcome FROM predictions WHERE id = ?", (row_id,)
        )
        predicted = cur.fetchone()[0]
        correct = 1 if predicted == actual else 0
        cur.execute(
            "UPDATE predictions SET actual_outcome = ?, correct = ? WHERE id = ?",
            (actual, correct, row_id),
        )
        symbol = "✅" if correct else "❌"
        print(f"  {symbol} {match_date} {home} vs {away} — predicted {predicted}, actual {actual}")
        updated += 1

    conn.commit()
    conn.close()
    print(f"\nReconciled {updated}/{len(pending)} prediction(s).")


if __name__ == "__main__":
    reconcile()
