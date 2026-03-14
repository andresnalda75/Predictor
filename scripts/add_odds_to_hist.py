"""
add_odds_to_hist.py — Download B365 odds from football-data.co.uk and merge into hist_matches.csv.

Usage:
    python scripts/add_odds_to_hist.py

Downloads EPL season CSVs, extracts B365H/B365D/B365A columns, and merges them
into data/hist_matches.csv by matching on date + home_team + away_team.
"""

import os
import io
import pandas as pd

try:
    import requests
except ImportError:
    import urllib.request
    requests = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST_MATCHES = os.path.join(ROOT, "data", "hist_matches.csv")

# Season codes used in hist_matches.csv → football-data.co.uk URL codes
SEASON_URL_MAP = {
    "1415": "1415", "1516": "1516", "1617": "1617", "1718": "1718",
    "1819": "1819", "1920": "1920", "2021": "2021", "2122": "2122",
    "2223": "2223", "2324": "2324", "2425": "2425",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"


def fetch_csv(url):
    """Download CSV from URL, return as string."""
    if requests:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    else:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8-sig")


def download_odds():
    """Download all season CSVs and return combined DataFrame with odds."""
    all_rows = []
    for season_code, url_code in SEASON_URL_MAP.items():
        url = BASE_URL.format(season=url_code)
        print(f"  Downloading {season_code} from {url}...")
        try:
            text = fetch_csv(url)
            df = pd.read_csv(io.StringIO(text))

            # Standardise date format to YYYY-MM-DD
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

            # Keep only what we need for the merge
            cols_needed = ["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]
            missing = [c for c in cols_needed if c not in df.columns]
            if missing:
                print(f"    WARNING: missing columns {missing} — skipping season {season_code}")
                continue

            subset = df[cols_needed].copy()
            subset.columns = ["date", "home_team", "away_team", "B365H", "B365D", "B365A"]
            subset["season_code"] = int(season_code)
            all_rows.append(subset)
            print(f"    Got {len(subset)} matches with odds")
        except Exception as e:
            print(f"    ERROR downloading {season_code}: {e}")

    if not all_rows:
        raise RuntimeError("No odds data downloaded!")
    return pd.concat(all_rows, ignore_index=True)


def main():
    print("=" * 60)
    print("ADD BOOKMAKER ODDS TO hist_matches.csv")
    print("=" * 60)

    # Load existing data
    print("\n1. Loading hist_matches.csv...")
    hist = pd.read_csv(HIST_MATCHES, parse_dates=["date"])
    print(f"   {len(hist)} matches")

    # Check if odds already present
    if "B365H" in hist.columns:
        non_null = hist["B365H"].notna().sum()
        print(f"   B365H already exists ({non_null} non-null values)")
        if non_null == len(hist):
            print("   All rows have odds — nothing to do.")
            return
        print("   Re-downloading to fill gaps...")

    # Download odds
    print("\n2. Downloading odds from football-data.co.uk...")
    odds = download_odds()

    # Merge on date + home_team + away_team
    print("\n3. Merging odds into hist_matches...")

    # Normalise dates for matching
    hist["date_str"] = hist["date"].dt.strftime("%Y-%m-%d")
    odds["date_str"] = odds["date"].dt.strftime("%Y-%m-%d")

    # Drop old odds columns if they exist
    for col in ["B365H", "B365D", "B365A"]:
        if col in hist.columns:
            hist = hist.drop(columns=[col])

    merged = hist.merge(
        odds[["date_str", "home_team", "away_team", "B365H", "B365D", "B365A"]],
        on=["date_str", "home_team", "away_team"],
        how="left",
    )
    merged = merged.drop(columns=["date_str"])

    matched = merged["B365H"].notna().sum()
    missing = merged["B365H"].isna().sum()
    print(f"   Matched: {matched}/{len(merged)} ({matched/len(merged):.1%})")
    if missing > 0:
        print(f"   Missing: {missing} matches without odds")
        missing_rows = merged[merged["B365H"].isna()][["date", "home_team", "away_team", "season_code"]]
        print(missing_rows.head(10).to_string(index=False))

    # Save
    print("\n4. Saving updated hist_matches.csv...")
    merged.to_csv(HIST_MATCHES, index=False)
    print(f"   Saved {len(merged)} rows with columns: {list(merged.columns)}")

    # Verify
    print("\n5. Verification — sample rows with odds:")
    sample = merged[merged["B365H"].notna()].tail(5)[
        ["date", "home_team", "away_team", "result", "B365H", "B365D", "B365A"]
    ]
    print(sample.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
