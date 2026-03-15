#!/usr/bin/env python3
"""
Fetch live EPL odds from The Odds API and return implied probabilities.

API: https://api.the-odds-api.com/v4/sports/soccer_epl/odds/
Free tier: 500 requests/month. Each call returns all upcoming EPL fixtures.

Usage:
    # As standalone test:
    ODDS_API_KEY=your_key python scripts/fetch_live_odds.py

    # As module (used by app.py):
    from scripts.fetch_live_odds import get_live_odds
    odds_cache = get_live_odds()
    match_odds = odds_cache.get(("Arsenal", "Chelsea"))
"""

import os
import time
import logging

import requests

logger = logging.getLogger(__name__)

# Preferred bookmakers in priority order (best available used)
PREFERRED_BOOKMAKERS = [
    "williamhill", "paddypower", "ladbrokes_uk", "coral",
    "skybet", "unibet_uk", "sport888", "betway", "betvictor",
]

# The Odds API team names → hist_matches.csv / app.py team names
ODDS_API_TO_APP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton",
    "West Ham United": "West Ham",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "West Bromwich Albion": "West Brom",
    "Sheffield United": "Sheffield United",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
}

# Cache: dict of (home, away) → {home_odds, draw_odds, away_odds, implied_home, ...}
_odds_cache = {}
_cache_timestamp = 0
CACHE_TTL = 3600  # 1 hour — odds don't change that fast, saves API quota


def _map_team(name):
    """Map Odds API team name to app.py convention."""
    return ODDS_API_TO_APP.get(name, name)


def _extract_best_odds(bookmakers):
    """Extract h2h odds from the best available bookmaker.

    Returns (home_odds, draw_odds, away_odds, bookmaker_key) or None.
    """
    # Try preferred bookmakers first
    by_key = {b["key"]: b for b in bookmakers}
    for pref in PREFERRED_BOOKMAKERS:
        if pref in by_key:
            bk = by_key[pref]
            return _parse_h2h(bk)

    # Fall back to first available
    if bookmakers:
        return _parse_h2h(bookmakers[0])

    return None


def _parse_h2h(bookmaker):
    """Parse h2h market from a bookmaker entry.

    Returns (home_odds, draw_odds, away_odds, bookmaker_key) or None.
    """
    for market in bookmaker["markets"]:
        if market["key"] == "h2h":
            outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
            # Outcomes use the full team names — we need home/away/draw
            draw_odds = outcomes.get("Draw")
            # The other two are the team names
            team_odds = {k: v for k, v in outcomes.items() if k != "Draw"}
            if len(team_odds) == 2 and draw_odds:
                teams = list(team_odds.keys())
                return (
                    team_odds[teams[0]], draw_odds, team_odds[teams[1]],
                    teams[0], teams[1], bookmaker["key"]
                )
    return None


def fetch_odds(api_key=None):
    """Fetch live odds from The Odds API.

    Returns dict: (home_team, away_team) → {
        home_odds, draw_odds, away_odds,
        implied_home, implied_draw, implied_away,
        home_edge, favourite, bookmaker, source
    }
    """
    global _odds_cache, _cache_timestamp

    # Check cache
    if _odds_cache and (time.time() - _cache_timestamp) < CACHE_TTL:
        logger.info("Odds cache hit (%d matches)", len(_odds_cache))
        return _odds_cache

    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        logger.warning("ODDS_API_KEY not set — cannot fetch live odds")
        return {}

    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/",
            params={
                "apiKey": key,
                "regions": "uk",
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Odds API request failed: %s", e)
        return {}

    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info("Odds API: %s requests remaining this month", remaining)

    data = resp.json()
    cache = {}

    for match in data:
        parsed = _extract_best_odds(match["bookmakers"])
        if not parsed:
            continue

        raw_home_odds, draw_odds, raw_away_odds, team_a, team_b, bk_key = parsed

        # Determine which team is home/away based on the API's home_team field
        api_home = match["home_team"]
        if team_a == api_home:
            home_odds, away_odds = raw_home_odds, raw_away_odds
        else:
            home_odds, away_odds = raw_away_odds, raw_home_odds

        home = _map_team(api_home)
        away = _map_team(match["away_team"])

        # Implied probabilities (1/odds, then normalise to sum to 1)
        raw_h = 1.0 / home_odds
        raw_d = 1.0 / draw_odds
        raw_a = 1.0 / away_odds
        total = raw_h + raw_d + raw_a  # overround
        imp_h = raw_h / total
        imp_d = raw_d / total
        imp_a = raw_a / total

        home_edge = imp_h - (1.0 / 3)
        probs = [imp_h, imp_d, imp_a]
        favourite = int(max(range(3), key=lambda i: probs[i]))  # 0=H, 1=D, 2=A

        cache[(home, away)] = {
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
            "implied_home": round(imp_h, 4),
            "implied_draw": round(imp_d, 4),
            "implied_away": round(imp_a, 4),
            "home_edge": round(home_edge, 4),
            "favourite": favourite,
            "bookmaker": bk_key,
            "source": "odds_api",
        }

    _odds_cache = cache
    _cache_timestamp = time.time()
    logger.info("Odds fetched: %d matches from The Odds API", len(cache))
    return cache


def get_match_odds(home, away, elo_home=1500, elo_away=1500):
    """Get odds for a specific match.

    Tries live odds first, falls back to ELO-derived proxy.
    Returns (implied_home, implied_draw, implied_away, home_edge, favourite, source).
    """
    cache = fetch_odds()
    match = cache.get((home, away))

    if match:
        logger.info("Real odds for %s vs %s: H=%.2f D=%.2f A=%.2f (%s)",
                     home, away, match["home_odds"], match["draw_odds"],
                     match["away_odds"], match["bookmaker"])
        return (
            match["implied_home"], match["implied_draw"], match["implied_away"],
            match["home_edge"], match["favourite"], "odds_api"
        )

    # Fallback: ELO-derived proxy
    logger.info("No live odds for %s vs %s — using ELO proxy", home, away)
    return _elo_proxy(elo_home, elo_away)


def _elo_proxy(elo_home, elo_away):
    """ELO-derived implied probabilities (fallback when no live odds)."""
    import numpy as np
    exp_h = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    elo_gap = abs(elo_home - elo_away)
    draw_prob = 0.26 * max(0.5, 1 - elo_gap / 600)
    p_home = exp_h * (1 - draw_prob)
    p_away = (1 - exp_h) * (1 - draw_prob)
    p_draw = 1 - p_home - p_away
    home_edge = p_home - (1.0 / 3)
    probs = [p_home, p_draw, p_away]
    favourite = int(np.argmax(probs))
    return p_home, p_draw, p_away, home_edge, favourite, "elo_proxy"


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("Set ODDS_API_KEY environment variable to test")
        print("  export ODDS_API_KEY=your_key")
        exit(1)

    cache = fetch_odds(key)
    if not cache:
        print("No odds returned — check API key and quota")
        exit(1)

    print(f"\n{'Match':<40} {'Bookie':<15} {'H':>6} {'D':>6} {'A':>6}  {'P(H)':>6} {'P(D)':>6} {'P(A)':>6}")
    print("-" * 95)
    for (home, away), odds in sorted(cache.items()):
        label = f"{home} vs {away}"
        print(f"{label:<40} {odds['bookmaker']:<15} "
              f"{odds['home_odds']:>6.2f} {odds['draw_odds']:>6.2f} {odds['away_odds']:>6.2f}  "
              f"{odds['implied_home']:>5.1%} {odds['implied_draw']:>5.1%} {odds['implied_away']:>5.1%}")
