#!/usr/bin/env python3
"""
Fetch FIFA/EA FC player ratings and aggregate by EPL team and position.

Data source: Kaggle stefanoleone992/ea-sports-fc-24-complete-player-dataset
Contains FIFA 15 through FC 24 (seasons 1415 through 2324).
FC 25 (season 2425) is handled by carrying forward FC 24 ratings.

Output: data/fifa_ratings.csv with columns:
    season_code, team, avg_gk, avg_def, avg_mid, avg_att, avg_overall, squad_depth

Usage:
    python scripts/fetch_fifa_ratings.py
"""

import os
import zipfile
import pandas as pd
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT = os.path.join(DATA_DIR, "fifa_ratings.csv")
CACHE_DIR = os.path.join(DATA_DIR, "raw")
CACHE_ZIP = os.path.join(CACHE_DIR, "fc24_dataset.zip")
CACHE_CSV = os.path.join(CACHE_DIR, "male_players.csv")

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/stefanoleone992/ea-sports-fc-24-complete-player-dataset"

EPL_LEAGUE_ID = 13  # English Premier League

# Position groupings
POS_GK = {"GK"}
POS_DEF = {"CB", "LB", "RB", "LWB", "RWB"}
POS_MID = {"CM", "CDM", "CAM", "LM", "RM"}
POS_ATT = {"ST", "CF", "LW", "RW"}

# FIFA team names → our team names
FIFA_TO_APP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "AFC Bournemouth": "Bournemouth",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
    "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield United",
    "Norwich City": "Norwich",
    "Huddersfield Town": "Huddersfield",
    "Cardiff City": "Cardiff",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Queens Park Rangers": "QPR",
    "Sunderland": "Sunderland",
    "Middlesbrough": "Middlesbrough",
}


def classify_position(positions_str):
    """Classify a player's primary role from comma-separated positions."""
    if pd.isna(positions_str):
        return "UNK"
    positions = {p.strip() for p in positions_str.split(",")}
    if positions & POS_GK:
        return "GK"
    if positions & POS_ATT:
        return "ATT"
    if positions & POS_MID:
        return "MID"
    if positions & POS_DEF:
        return "DEF"
    return "UNK"


def download_dataset():
    """Download FIFA dataset from Kaggle if not cached."""
    if os.path.exists(CACHE_CSV):
        print(f"  Using cached {CACHE_CSV}")
        return CACHE_CSV

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  Downloading from Kaggle...")
    resp = requests.get(KAGGLE_URL, stream=True)
    resp.raise_for_status()
    with open(CACHE_ZIP, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  Downloaded {os.path.getsize(CACHE_ZIP) / 1e6:.1f} MB")

    print(f"  Extracting...")
    with zipfile.ZipFile(CACHE_ZIP) as zf:
        zf.extract("male_players.csv", CACHE_DIR)

    os.remove(CACHE_ZIP)
    return CACHE_CSV


def build_ratings(csv_path):
    """Build team-level FIFA ratings from player data."""
    df = pd.read_csv(csv_path, low_memory=False,
                     usecols=["fifa_version", "league_id", "club_name",
                              "overall", "player_positions"])

    # Filter EPL only
    epl = df[df["league_id"] == EPL_LEAGUE_ID].copy()
    print(f"  EPL players: {len(epl)}")

    # Classify positions
    epl["role"] = epl["player_positions"].apply(classify_position)

    # Map FIFA version → season_code
    # FIFA 15 → 2014/15 → 1415
    epl["season_code"] = epl["fifa_version"].apply(
        lambda v: int(f"{(int(v) + 1999) % 100}{(int(v) + 2000) % 100}")
    )

    # Map team names
    epl["team"] = epl["club_name"].map(lambda x: FIFA_TO_APP.get(x, x))

    rows = []
    for (sc, team), group in epl.groupby(["season_code", "team"]):
        gk = group[group["role"] == "GK"]["overall"]
        defs = group[group["role"] == "DEF"]["overall"]
        mids = group[group["role"] == "MID"]["overall"]
        atts = group[group["role"] == "ATT"]["overall"]

        avg_gk = round(gk.mean(), 1) if len(gk) > 0 else None
        avg_def = round(defs.mean(), 1) if len(defs) > 0 else None
        avg_mid = round(mids.mean(), 1) if len(mids) > 0 else None
        avg_att = round(atts.mean(), 1) if len(atts) > 0 else None
        avg_overall = round(group["overall"].mean(), 1)
        # Squad depth: average of top 15 players
        top15 = group.nlargest(15, "overall")["overall"]
        squad_depth = round(top15.mean(), 1)

        rows.append({
            "season_code": sc,
            "team": team,
            "avg_gk": avg_gk,
            "avg_def": avg_def,
            "avg_mid": avg_mid,
            "avg_att": avg_att,
            "avg_overall": avg_overall,
            "squad_depth": squad_depth,
        })

    ratings = pd.DataFrame(rows)

    # Carry forward FC 24 ratings for season 2425 (FC 25 not in dataset)
    fc24 = ratings[ratings["season_code"] == 2324].copy()
    fc24["season_code"] = 2425
    ratings = pd.concat([ratings, fc24], ignore_index=True)

    # Fill any remaining NaN per-role ratings with season average
    for col in ["avg_gk", "avg_def", "avg_mid", "avg_att"]:
        for sc in ratings["season_code"].unique():
            mask = (ratings["season_code"] == sc) & ratings[col].isna()
            if mask.any():
                season_avg = ratings.loc[
                    (ratings["season_code"] == sc) & ratings[col].notna(), col
                ].mean()
                ratings.loc[mask, col] = round(season_avg, 1)

    return ratings.sort_values(["season_code", "team"]).reset_index(drop=True)


def main():
    print("=" * 60)
    print("FETCH FIFA RATINGS")
    print("=" * 60)

    print("\n1. Loading data...")
    csv_path = download_dataset()

    print("\n2. Building team ratings...")
    ratings = build_ratings(csv_path)

    print(f"\n3. Saving fifa_ratings.csv...")
    ratings.to_csv(OUTPUT, index=False)
    print(f"  {len(ratings)} rows, {len(ratings.columns)} columns")
    print(f"  Seasons: {sorted(ratings['season_code'].unique())}")
    print(f"  NaN check: {ratings.isna().sum().sum()} total NaN")

    print(f"\n4. Sample (season 2425):")
    sample = ratings[ratings["season_code"] == 2425].head(10)
    print(sample.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
