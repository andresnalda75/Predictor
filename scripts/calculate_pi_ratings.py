#!/usr/bin/env python3
"""
Calculate Pi-ratings for all teams from hist_matches.csv.

Pi-ratings (Constantinou & Fenton 2013) maintain separate home (R_h) and
away (R_a) ratings per team. They outperform ELO for football because they
account for home/away performance asymmetry and use a log-scaled goal-
difference transform that applies diminishing returns to large margins.

The key insight vs ELO: a 4-0 win provides less NEW information than the
raw scoreline suggests — Pi-ratings' psi() function captures this.

Parameters (optimised for EPL, Constantinou & Fenton 2013):
    LAMBDA = 0.035   learning rate
    GAMMA  = 0.70    cross-learning factor (home <-> away contamination)

Update per match (home team i vs away team j, goal diff D = hg - ag):
    E        = R_h[i] - R_a[j]              expected goal diff
    delta    = psi(D) - psi(E)              discrepancy (transformed)
    R_h[i]  += LAMBDA * delta
    R_a[i]  += GAMMA * LAMBDA * delta       cross-learning
    R_a[j]  -= LAMBDA * delta
    R_h[j]  -= GAMMA * LAMBDA * delta       cross-learning

Transformation: psi(x) = sign(x) * 3 * log10(1 + |x|)
    psi(0)=0, psi(1)≈0.90, psi(3)≈1.81, psi(7)≈2.71

Ratings persist across seasons — a relegated team carries its rating back
up, which is empirically more predictive than resetting each season.

Output:
    data/pi_ratings.csv  — one row per match with pre-match columns:
        pi_home          R_h of the home team before the match
        pi_away          R_a of the away team before the match
        pi_diff          pi_home - pi_away  (the raw match predictor)
        pi_home_overall  (R_h + R_a) / 2 for home team
        pi_away_overall  (R_h + R_a) / 2 for away team
"""

import math
import os
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
LAMBDA = 0.035   # learning rate
GAMMA  = 0.70    # cross-learning (home <-> away rating contamination)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT  = os.path.join(BASE, "data", "hist_matches.csv")
OUTPUT = os.path.join(BASE, "data", "pi_ratings.csv")


# ---------------------------------------------------------------------------
# Transformation function
# ---------------------------------------------------------------------------
def psi(x: float) -> float:
    """Log-scale goal-difference transform with diminishing returns."""
    if x == 0:
        return 0.0
    return math.copysign(3.0 * math.log10(1.0 + abs(x)), x)


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------
def calculate_pi_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process matches in chronological order, recording pre-match Pi-ratings
    then applying the update. Ratings persist across seasons.

    Returns a copy of df with five new columns appended.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Each team has a home rating (R_h) and away rating (R_a), both start at 0
    R_h: dict[str, float] = {}
    R_a: dict[str, float] = {}

    pi_home_col:    list[float] = []
    pi_away_col:    list[float] = []
    pi_diff_col:    list[float] = []
    pi_home_ov_col: list[float] = []
    pi_away_ov_col: list[float] = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        D    = float(row["home_goals"]) - float(row["away_goals"])

        # Initialise unseen teams (newly promoted sides etc.)
        if home not in R_h:
            R_h[home] = 0.0
            R_a[home] = 0.0
        if away not in R_h:
            R_h[away] = 0.0
            R_a[away] = 0.0

        # --- Record PRE-MATCH ratings ---
        rh_i = R_h[home]
        ra_j = R_a[away]
        pi_home_col.append(rh_i)
        pi_away_col.append(ra_j)
        pi_diff_col.append(rh_i - ra_j)
        pi_home_ov_col.append((R_h[home] + R_a[home]) / 2.0)
        pi_away_ov_col.append((R_h[away] + R_a[away]) / 2.0)

        # --- Update ratings ---
        E     = rh_i - ra_j
        delta = LAMBDA * (psi(D) - psi(E))

        R_h[home] += delta
        R_a[home] += GAMMA * delta    # cross-learning: home result informs away rating
        R_a[away] -= delta
        R_h[away] -= GAMMA * delta    # cross-learning: away result informs home rating

    out = df.copy()
    out["pi_home"]         = pi_home_col
    out["pi_away"]         = pi_away_col
    out["pi_diff"]         = pi_diff_col
    out["pi_home_overall"] = pi_home_ov_col
    out["pi_away_overall"] = pi_away_ov_col

    return out


# ---------------------------------------------------------------------------
# Final ratings per team (post last match)
# ---------------------------------------------------------------------------
def get_final_ratings(out: pd.DataFrame):
    """Replay updates to get post-last-match ratings per team."""
    R_h: dict[str, float] = {}
    R_a: dict[str, float] = {}
    for _, row in out.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        D = float(row["home_goals"]) - float(row["away_goals"])
        if home not in R_h: R_h[home] = R_a[home] = 0.0
        if away not in R_h: R_h[away] = R_a[away] = 0.0
        E = R_h[home] - R_a[away]
        delta = LAMBDA * (psi(D) - psi(E))
        R_h[home] += delta;  R_a[home] += GAMMA * delta
        R_a[away] -= delta;  R_h[away] -= GAMMA * delta
    final_ratings = {t: (R_h[t] + R_a[t]) / 2.0 for t in R_h}
    return R_h, R_a, final_ratings


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(out: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Pi-ratings calculated for {len(out)} matches")
    print(f"{'='*60}")

    # Final ratings per team (after last match in dataset)
    teams: dict[str, dict] = {}
    for _, row in out.iterrows():
        for side, team_col, pi_col in [("home", "home_team", "pi_home"),
                                        ("away", "away_team", "pi_away")]:
            t = row[team_col]
            if t not in teams:
                teams[t] = {}
            teams[t][f"last_{side}_rating"] = row[pi_col]
            teams[t]["last_overall"] = (
                row["pi_home_overall"] if side == "home" else row["pi_away_overall"]
            )

    R_h, R_a, final_ratings = get_final_ratings(out)

    print("\nTop 10 teams by final Pi-rating (end of 2024/25):")
    for rank, (team, rating) in enumerate(
        sorted(final_ratings.items(), key=lambda x: -x[1])[:10], 1
    ):
        print(f"  {rank:2d}. {team:<22}  R_h={R_h[team]:+.4f}  R_a={R_a[team]:+.4f}  overall={rating:+.4f}")

    print("\nBottom 5 teams by final Pi-rating:")
    for team, rating in sorted(final_ratings.items(), key=lambda x: x[1])[:5]:
        print(f"      {team:<22}  overall={rating:+.4f}")

    print(f"\npi_diff range: {out['pi_diff'].min():.4f} to {out['pi_diff'].max():.4f}")
    print(f"pi_diff mean:  {out['pi_diff'].mean():.4f}  (should be near 0)")
    print(f"pi_diff std:   {out['pi_diff'].std():.4f}")

    # Correlation between pi_diff and actual outcome
    out2 = out.copy()
    out2["actual_gd"] = out2["home_goals"] - out2["away_goals"]
    corr = out2["pi_diff"].corr(out2["actual_gd"])
    print(f"\nCorrelation pi_diff <-> actual goal diff: {corr:.4f}")

    # Direction accuracy: does positive pi_diff predict home win?
    out2["pi_pred_home"] = out2["pi_diff"] > 0
    out2["actual_home"]  = out2["result"] == "H"
    # Only count non-draws and non-zero pi_diff for directional accuracy
    decisive = out2[(out2["result"] != "D") & (out2["pi_diff"] != 0)]
    dir_acc = (decisive["pi_pred_home"] == decisive["actual_home"]).mean()
    print(f"Directional accuracy (excl. draws, n={len(decisive)}): {dir_acc:.1%}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Reading {INPUT}")
    df = pd.read_csv(INPUT, parse_dates=["date"])
    print(f"Loaded {len(df)} matches across seasons: {sorted(df['season_code'].unique())}")

    out = calculate_pi_ratings(df)
    print_diagnostics(out)

    out.to_csv(OUTPUT, index=False)
    print(f"Saved to {OUTPUT}")

    # Save final team ratings for app.py to load at startup
    R_h, R_a, final = get_final_ratings(out)
    team_rows = [
        {"team": t, "pi_r_h": R_h[t], "pi_r_a": R_a[t], "pi_overall": final[t]}
        for t in sorted(R_h.keys())
    ]
    team_path = os.path.join(BASE, "data", "pi_team_ratings.csv")
    pd.DataFrame(team_rows).to_csv(team_path, index=False)
    print(f"Saved team ratings to {team_path}")
    print(f"New columns: pi_home, pi_away, pi_diff, pi_home_overall, pi_away_overall")
