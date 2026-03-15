#!/usr/bin/env python3
"""
Fetch historical xG data from understat.com and merge into hist_matches.csv.

Uses direct HTTP requests to understat's JSON API — no async, no aiohttp,
no understat package. Only requires `requests` and `pandas`.

API endpoint: https://understat.com/getLeagueData/EPL/{year}
Returns JSON with a "dates" array of per-match xG data.

Usage:
    python scripts/fetch_xg_understat.py
"""

import json
import os
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_MATCHES = os.path.join(BASE, "data", "hist_matches.csv")

# ---------------------------------------------------------------------------
# Team name mapping: understat → hist_matches.csv
# ---------------------------------------------------------------------------
UNDERSTAT_TO_HIST = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Newcastle United": "Newcastle",
}


def map_team(name):
    """Map understat team name to hist_matches.csv convention."""
    return UNDERSTAT_TO_HIST.get(name, name)


def fetch_season(year):
    """Fetch all match results with xG for a single EPL season.

    Args:
        year: Starting year of the season (e.g. 2017 for 2017-18).

    Returns:
        List of dicts with date, home_team, away_team, home_xg, away_xg.
    """
    url = f"https://understat.com/getLeagueData/EPL/{year}"
    resp = requests.get(url, headers={"X-Requested-With": "XMLHttpRequest"})
    resp.raise_for_status()

    data = resp.json()
    dates = data["dates"]
    results = [m for m in dates if m["isResult"]]

    matches = []
    for m in results:
        matches.append({
            "date": m["datetime"][:10],
            "home_team": map_team(m["h"]["title"]),
            "away_team": map_team(m["a"]["title"]),
            "home_xg": round(float(m["xG"]["h"]), 2),
            "away_xg": round(float(m["xG"]["a"]), 2),
        })

    return matches


def fetch_all_seasons():
    """Fetch xG data for EPL seasons 2017-18 through 2024-25."""
    seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    all_matches = []
    for year in seasons:
        print(f"  Fetching {year}/{year + 1}...")
        matches = fetch_season(year)
        all_matches.extend(matches)
        print(f"    → {len(matches)} matches")
        time.sleep(1)  # polite rate limiting

    xg_df = pd.DataFrame(all_matches)
    print(f"\n  Total xG records: {len(xg_df)}")
    return xg_df


def merge_xg_into_hist(xg_df):
    """Merge xG data into hist_matches.csv by date + home_team + away_team."""
    hist = pd.read_csv(HIST_MATCHES)
    print(f"\n  hist_matches.csv: {len(hist)} rows")

    # Normalise date format for matching
    hist["date_str"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    xg_df["date_str"] = pd.to_datetime(xg_df["date"]).dt.strftime("%Y-%m-%d")

    # Drop existing xG columns if re-running
    for col in ["home_xg", "away_xg"]:
        if col in hist.columns:
            hist = hist.drop(columns=[col])
            print(f"    Dropped existing {col} column")

    # Merge
    merged = hist.merge(
        xg_df[["date_str", "home_team", "away_team", "home_xg", "away_xg"]],
        on=["date_str", "home_team", "away_team"],
        how="left",
    )

    # Drop helper column
    merged = merged.drop(columns=["date_str"])

    matched = merged["home_xg"].notna().sum()
    total = len(merged)
    print(f"  Matched: {matched}/{total} ({matched / total:.1%})")
    print(f"  Unmatched: {total - matched}")

    # Show unmatched xG rows (team name mapping issues)
    xg_merged = xg_df.merge(
        hist[["date_str", "home_team", "away_team"]],
        on=["date_str", "home_team", "away_team"],
        how="left",
        indicator=True,
    )
    unmatched_xg = xg_merged[xg_merged["_merge"] == "left_only"]
    if len(unmatched_xg) > 0:
        print(f"\n  WARNING: {len(unmatched_xg)} xG rows didn't match hist_matches.csv:")
        for _, r in unmatched_xg.head(10).iterrows():
            print(f"    {r['date_str']} {r['home_team']} vs {r['away_team']}")

    return merged


def main():
    print("=" * 60)
    print("FETCH xG DATA FROM UNDERSTAT")
    print("=" * 60)

    # 1. Fetch xG from understat
    print("\n1. Fetching xG data from understat.com...")
    xg_df = fetch_all_seasons()

    # 2. Merge into hist_matches.csv
    print("\n2. Merging xG into hist_matches.csv...")
    merged = merge_xg_into_hist(xg_df)

    # 3. Save
    print("\n3. Saving updated hist_matches.csv...")
    merged.to_csv(HIST_MATCHES, index=False)
    print(f"  Saved {len(merged)} rows to hist_matches.csv")

    # 4. Sample output
    print("\n4. Sample rows with xG data:")
    sample = merged[merged["home_xg"].notna()].head(10)
    cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"]
    print(sample[cols].to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
