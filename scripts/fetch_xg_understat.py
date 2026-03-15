#!/usr/bin/env python3
"""
Fetch historical xG data from understat.com and merge into hist_matches.csv.

Uses the `understat` Python package (pip install understat) to scrape per-match
expected goals (home_xg, away_xg) for EPL seasons 2017-2025.

Usage:
    python scripts/fetch_xg_understat.py
"""

import asyncio
import os
import sys
from datetime import datetime

import pandas as pd
from understat import Understat

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
    "Sheffield United": "Sheffield United",
    "Tottenham": "Tottenham",
    "Newcastle United": "Newcastle",
    "Brighton": "Brighton",
    "Leicester": "Leicester",
    "Leeds": "Leeds",
    "West Ham": "West Ham",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "Southampton": "Southampton",
    "Burnley": "Burnley",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Watford": "Watford",
    "Norwich": "Norwich",
    "Huddersfield": "Huddersfield",
    "Cardiff": "Cardiff",
    "Swansea": "Swansea",
    "Stoke": "Stoke",
    "Luton": "Luton",
    "Ipswich": "Ipswich",
    "Middlesbrough": "Middlesbrough",
    "Hull": "Hull",
    "QPR": "QPR",
    "Sunderland": "Sunderland",
}


def map_team(name):
    """Map understat team name to hist_matches.csv convention."""
    return UNDERSTAT_TO_HIST.get(name, name)


async def fetch_all_seasons():
    """Fetch xG data for EPL seasons 2017-18 through 2024-25."""
    # Understat uses the starting year: 2017 = 2017-18 season
    seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    all_matches = []
    import aiohttp
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        for year in seasons:
            print(f"  Fetching {year}/{year + 1}...")
            results = await understat.get_league_results("epl", year)

            for match in results:
                date_str = match["datetime"][:10]  # "YYYY-MM-DD"
                home_team = map_team(match["h"]["title"])
                away_team = map_team(match["a"]["title"])
                home_xg = float(match["xG"]["h"])
                away_xg = float(match["xG"]["a"])

                all_matches.append({
                    "date": date_str,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_xg": round(home_xg, 2),
                    "away_xg": round(away_xg, 2),
                })

            print(f"    → {len(results)} matches fetched")

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
    xg_df = asyncio.run(fetch_all_seasons())

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
