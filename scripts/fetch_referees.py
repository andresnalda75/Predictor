#!/usr/bin/env python3
"""
Add referee + card data to hist_matches.csv from football-data.co.uk CSVs.

Each season's CSV has: Referee, HY (home yellows), HR (home reds),
AY (away yellows), AR (away reds).

Merges by date + HomeTeam + AwayTeam. Output: updated hist_matches.csv
with new columns: referee, home_yellows, home_reds, away_yellows, away_reds.

Usage:
    python scripts/fetch_referees.py
"""

import os
import pandas as pd
import requests
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_MATCHES = os.path.join(BASE, "data", "hist_matches.csv")

SEASON_CODES = [1415, 1516, 1617, 1718, 1819, 1920, 2021, 2122, 2223, 2324, 2425]

# football-data.co.uk team names → our team names
FD_TO_APP = {
    "Man City": "Man City",
    "Man United": "Man United",
    "Nott'm Forest": "Nott'm Forest",
    "Wolves": "Wolves",
    "Newcastle": "Newcastle",
    "Tottenham": "Tottenham",
    "Brighton": "Brighton",
    "West Ham": "West Ham",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Bournemouth": "Bournemouth",
    "Sheffield United": "Sheffield United",
    "West Brom": "West Brom",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
}


def map_team(name):
    return FD_TO_APP.get(name, name)


def fetch_season(sc):
    """Download one season CSV from football-data.co.uk."""
    url = f"https://www.football-data.co.uk/mmz4281/{sc}/E0.csv"
    resp = requests.get(url)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    # Standardise column names
    cols = {}
    for c in df.columns:
        if c in ("HomeTeam", "Home"):
            cols[c] = "home_team"
        elif c in ("AwayTeam", "Away"):
            cols[c] = "away_team"
        elif c == "Date":
            cols[c] = "date"
    df = df.rename(columns=cols)

    # Parse date — football-data.co.uk uses DD/MM/YYYY or DD/MM/YY
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # Map team names
    df["home_team"] = df["home_team"].map(map_team)
    df["away_team"] = df["away_team"].map(map_team)

    # Extract referee + cards
    result = df[["date", "home_team", "away_team"]].copy()
    result["referee"] = df.get("Referee", "")
    result["home_yellows"] = pd.to_numeric(df.get("HY", 0), errors="coerce").fillna(0).astype(int)
    result["home_reds"] = pd.to_numeric(df.get("HR", 0), errors="coerce").fillna(0).astype(int)
    result["away_yellows"] = pd.to_numeric(df.get("AY", 0), errors="coerce").fillna(0).astype(int)
    result["away_reds"] = pd.to_numeric(df.get("AR", 0), errors="coerce").fillna(0).astype(int)

    return result


def main():
    print("=" * 60)
    print("FETCH REFEREE + CARD DATA")
    print("=" * 60)

    all_ref = []
    for sc in SEASON_CODES:
        print(f"  Fetching {sc}...")
        df = fetch_season(sc)
        all_ref.append(df)
        print(f"    {len(df)} matches, {df['referee'].nunique()} referees")
        time.sleep(0.5)

    ref_df = pd.concat(all_ref, ignore_index=True)
    print(f"\n  Total: {len(ref_df)} matches, {ref_df['referee'].nunique()} unique referees")

    # Merge into hist_matches.csv
    hist = pd.read_csv(HIST_MATCHES)
    print(f"  hist_matches.csv: {len(hist)} rows")

    hist["date_str"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    ref_df["date_str"] = ref_df["date"].dt.strftime("%Y-%m-%d")

    # Drop existing referee columns if re-running
    for col in ["referee", "home_yellows", "home_reds", "away_yellows", "away_reds"]:
        if col in hist.columns:
            hist = hist.drop(columns=[col])

    merged = hist.merge(
        ref_df[["date_str", "home_team", "away_team", "referee",
                "home_yellows", "home_reds", "away_yellows", "away_reds"]],
        on=["date_str", "home_team", "away_team"],
        how="left",
    )
    merged = merged.drop(columns=["date_str"])

    matched = merged["referee"].notna().sum()
    print(f"  Matched: {matched}/{len(merged)} ({matched/len(merged):.1%})")

    # Fill unmatched
    merged["referee"] = merged["referee"].fillna("Unknown")
    for col in ["home_yellows", "home_reds", "away_yellows", "away_reds"]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged.to_csv(HIST_MATCHES, index=False)
    print(f"  Saved {len(merged)} rows to hist_matches.csv")

    # Sample
    print(f"\n  Sample:")
    sample = merged[merged["referee"] != "Unknown"].tail(5)
    print(sample[["date", "home_team", "away_team", "referee",
                   "home_yellows", "away_yellows"]].to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
