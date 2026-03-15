#!/usr/bin/env python3
"""
Scrape squad market values from transfermarkt.com for EPL teams.

Source: transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/1
Covers seasons 2014/15 through 2024/25.

Output: data/transfermarkt_values.csv with columns:
    season_code, team, squad_value, avg_player_value,
    squad_value_norm, avg_player_norm

Usage:
    python scripts/fetch_transfermarkt.py
"""

import os
import re
import time
import pandas as pd
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "data", "transfermarkt_values.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}

TM_TO_APP = {
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
    "Ipswich Town": "Ipswich",
    "Chelsea FC": "Chelsea",
    "Arsenal FC": "Arsenal",
    "Liverpool FC": "Liverpool",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Brentford FC": "Brentford",
    "Southampton FC": "Southampton",
    "Burnley FC": "Burnley",
    "Watford FC": "Watford",
    "Norwich City": "Norwich",
    "Sheffield United": "Sheffield United",
    "Luton Town": "Luton",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Queens Park Rangers": "QPR",
    "Sunderland AFC": "Sunderland",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "Middlesbrough FC": "Middlesbrough",
    "West Bromwich Albion": "West Brom",
}


def parse_value(s):
    """Parse euro value string like '€1.36bn' or '€950.95m' to float."""
    s = s.replace("€", "").replace("\xa0", "").replace("&nbsp;", "").strip()
    if not s or s == "-":
        return 0
    if "bn" in s:
        return float(s.replace("bn", "")) * 1e9
    elif "m" in s:
        return float(s.replace("m", "")) * 1e6
    elif "k" in s or "Th." in s:
        return float(s.replace("k", "").replace("Th.", "")) * 1e3
    return float(s)


def scrape_season(saison_id):
    """Scrape one season from transfermarkt.com."""
    url = f"https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/1?saison_id={saison_id}"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    html = resp.text

    # Extract team names
    teams = re.findall(
        r'<td class="hauptlink no-border-links">(.*?)</td>', html, re.DOTALL
    )
    teams = [re.sub(r"<[^>]+>", "", t).replace("\xa0", "").strip() for t in teams]

    # Extract right-aligned values (first 2 are league total, then pairs per team)
    raw_values = re.findall(r'<td class="rechts">(.*?)</td>', html)
    raw_values = [
        re.sub(r"<[^>]+>", "", v).strip()
        for v in raw_values
        if re.sub(r"<[^>]+>", "", v).strip()
    ]

    # Skip first 2 values (league avg + league total)
    team_values = raw_values[2:]

    rows = []
    for i, tm_name in enumerate(teams):
        avg_str = team_values[i * 2] if i * 2 < len(team_values) else "0"
        total_str = team_values[i * 2 + 1] if i * 2 + 1 < len(team_values) else "0"

        team = TM_TO_APP.get(tm_name, tm_name)
        rows.append({
            "team": team,
            "squad_value": int(parse_value(total_str)),
            "avg_player_value": int(parse_value(avg_str)),
        })

    return rows


def main():
    print("=" * 60)
    print("FETCH TRANSFERMARKT SQUAD VALUES")
    print("=" * 60)

    all_rows = []
    for start_year in range(2014, 2025):
        sc = int(f"{start_year % 100}{(start_year + 1) % 100}")
        print(f"  Scraping {sc} (saison_id={start_year})...")
        rows = scrape_season(start_year)
        for r in rows:
            r["season_code"] = sc
        all_rows.extend(rows)
        print(f"    {len(rows)} teams")
        time.sleep(2)  # polite rate limiting

    df = pd.DataFrame(all_rows)

    # Normalize within each season (ratio to season mean)
    for sc in df["season_code"].unique():
        mask = df["season_code"] == sc
        sv_mean = df.loc[mask, "squad_value"].mean()
        av_mean = df.loc[mask, "avg_player_value"].mean()
        df.loc[mask, "squad_value_norm"] = round(df.loc[mask, "squad_value"] / sv_mean, 3) if sv_mean else 1.0
        df.loc[mask, "avg_player_norm"] = round(df.loc[mask, "avg_player_value"] / av_mean, 3) if av_mean else 1.0

    df = df.sort_values(["season_code", "team"]).reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)

    print(f"\n  Saved {len(df)} rows to transfermarkt_values.csv")
    print(f"  Seasons: {sorted(df['season_code'].unique())}")
    for sc in sorted(df["season_code"].unique()):
        n = len(df[df["season_code"] == sc])
        print(f"    {sc}: {n} teams")

    print(f"\n  Sample (2425, top 5 by value):")
    s = df[df["season_code"] == 2425].sort_values("squad_value", ascending=False).head(5)
    print(s[["team", "squad_value", "avg_player_value", "squad_value_norm"]].to_string(index=False))

    print(f"\n  NaN check: {df.isna().sum().sum()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
