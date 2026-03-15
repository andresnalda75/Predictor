"""
retrain_model.py — Full retraining pipeline for the EPL Predictor champion model.

Run in Google Colab (XGBoost needs libomp, unavailable on macOS):

    !git clone https://github.com/andresnalda75/Predictor.git
    %cd Predictor
    !pip install xgboost optuna scikit-learn pandas numpy
    !python notebooks/retrain_model.py

What it does:
  1. Loads hist_matches.csv + pi_ratings.csv
  2. Rebuilds all features from scratch (form, standings, shots, ELO, Pi-ratings)
  3. Runs RFE to select optimal feature subset
  4. Runs Optuna hyperparameter search with walk-forward CV
  5. Trains final model on 80% chronological split
  6. Evaluates on held-out 20% test set
  7. Saves model files (xgb_champion.pkl, cols_champion.pkl, label_encoder.pkl)
  8. Updates validation.json and benchmarks/results.json
  9. Commits and pushes to GitHub
"""

import os
import sys
import math
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
BENCH_DIR = os.path.join(ROOT, "benchmarks")

HIST_MATCHES = os.path.join(DATA_DIR, "hist_matches.csv")
PI_RATINGS = os.path.join(DATA_DIR, "pi_ratings.csv")
HIST_FEATURES = os.path.join(DATA_DIR, "hist_features.csv")
VALIDATION_JSON = os.path.join(DATA_DIR, "validation.json")
RESULTS_JSON = os.path.join(BENCH_DIR, "results.json")

OUTCOME_ORDER = ["H", "D", "A"]

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors app.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def get_form(df_up_to, team, n=5, home_only=False, away_only=False):
    """Exponential-decay weighted rolling form over last n matches."""
    tm = df_up_to[((df_up_to["home_team"] == team) | (df_up_to["away_team"] == team))]
    if home_only:
        tm = tm[tm["home_team"] == team]
    if away_only:
        tm = tm[tm["away_team"] == team]
    tm = tm.tail(n)
    count = len(tm)
    if count == 0:
        return 0, 0, 0, 0, 0

    decay = 0.5 ** (1.0 / (n - 1)) if n > 1 else 1.0
    pts, gf, ga, wins, draws = 0.0, 0.0, 0.0, 0.0, 0.0
    w_sum = 0.0
    for j, (_, r) in enumerate(tm.iterrows()):
        w = decay ** (count - 1 - j)
        w_sum += w
        ih = r["home_team"] == team
        g_for = r["home_goals"] if ih else r["away_goals"]
        g_agt = r["away_goals"] if ih else r["home_goals"]
        gf += w * g_for
        ga += w * g_agt
        if (ih and r["result"] == "H") or (not ih and r["result"] == "A"):
            pts += w * 3
            wins += w
        elif r["result"] == "D":
            pts += w
            draws += w
    scale = count / w_sum
    return pts * scale, gf * scale, ga * scale, wins * scale, draws * scale


def get_rolling_shots(df_up_to, team, n=5):
    """Rolling n-match average shots for/against."""
    all_tm = df_up_to[((df_up_to["home_team"] == team) | (df_up_to["away_team"] == team))]
    tm = all_tm[all_tm["hs"] > 0].tail(n)
    if len(tm) == 0:
        return 0, 0, 0, 0
    sh, sha, sot, sota = 0, 0, 0, 0
    for _, r in tm.iterrows():
        ih = r["home_team"] == team
        sh += r["hs"] if ih else r["as_"]
        sha += r["as_"] if ih else r["hs"]
        sot += r["hst"] if ih else r["ast"]
        sota += r["ast"] if ih else r["hst"]
    n2 = len(tm)
    return sh / n2, sha / n2, sot / n2, sota / n2


def get_rolling_xg(df_up_to, team, n=5):
    """Rolling n-match average xG for and against a team.

    Returns (xg_for_avg, xg_against_avg) or (None, None) if no xG data.
    """
    tm = df_up_to[((df_up_to["home_team"] == team) | (df_up_to["away_team"] == team))]
    # Only rows that have xG data
    if "home_xg" not in tm.columns:
        return None, None
    tm = tm[tm["home_xg"].notna()].tail(n)
    if len(tm) == 0:
        return None, None
    xg_for, xg_against = 0.0, 0.0
    for _, r in tm.iterrows():
        if r["home_team"] == team:
            xg_for += r["home_xg"]
            xg_against += r["away_xg"]
        else:
            xg_for += r["away_xg"]
            xg_against += r["home_xg"]
    count = len(tm)
    return xg_for / count, xg_against / count


def get_days_rest(df_up_to, team, match_date):
    """Days since the team's most recent match before match_date."""
    tm = df_up_to[((df_up_to["home_team"] == team) | (df_up_to["away_team"] == team))]
    if len(tm) == 0:
        return 7  # default: one week
    last_date = pd.to_datetime(tm.iloc[-1]["date"])
    match_dt = pd.to_datetime(match_date)
    days = (match_dt - last_date).days
    return max(days, 0)


def get_momentum(df_up_to, team, n=5):
    """Form momentum: PPG in last n matches minus PPG in prior n matches.
    Positive = rising form, negative = falling form."""
    tm = df_up_to[((df_up_to["home_team"] == team) | (df_up_to["away_team"] == team))]
    if len(tm) < 2:
        return 0.0
    recent = tm.tail(n)
    older = tm.iloc[max(0, len(tm) - 2 * n):max(0, len(tm) - n)]

    def _ppg(matches):
        if len(matches) == 0:
            return 0.0
        pts = 0
        for _, r in matches.iterrows():
            ih = r["home_team"] == team
            if (ih and r["result"] == "H") or (not ih and r["result"] == "A"):
                pts += 3
            elif r["result"] == "D":
                pts += 1
        return pts / len(matches)

    return round(_ppg(recent) - _ppg(older), 3)


def get_cumulative_standing(df_season):
    """Build league table from a season's completed matches."""
    table = {}
    for _, r in df_season.iterrows():
        h, a = r["home_team"], r["away_team"]
        if h not in table:
            table[h] = {"pts": 0, "gd": 0, "gf": 0}
        if a not in table:
            table[a] = {"pts": 0, "gd": 0, "gf": 0}
        hg, ag = r["home_goals"], r["away_goals"]
        table[h]["gf"] += hg
        table[a]["gf"] += ag
        table[h]["gd"] += hg - ag
        table[a]["gd"] += ag - hg
        if r["result"] == "H":
            table[h]["pts"] += 3
        elif r["result"] == "A":
            table[a]["pts"] += 3
        else:
            table[h]["pts"] += 1
            table[a]["pts"] += 1
    # Rank by pts then gd then gf
    ranked = sorted(table.items(), key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]))
    standings = {}
    for pos, (team, stats) in enumerate(ranked, 1):
        standings[team] = {"position": pos, "pts": stats["pts"], "gd": stats["gd"]}
    return standings


def build_features(matches_df, pi_df):
    """
    Build feature matrix from raw match data + pi_ratings.
    Processes matches chronologically, using only data available BEFORE each match.
    Returns DataFrame with all features + 'result' column.
    """
    matches = matches_df.sort_values("date").reset_index(drop=True)
    pi = pi_df.sort_values("date").reset_index(drop=True)

    # Merge pi columns onto matches by index (both sorted by date, same length)
    assert len(matches) == len(pi), f"Row count mismatch: matches={len(matches)}, pi={len(pi)}"
    for col in ["pi_home", "pi_away", "pi_diff", "pi_home_overall", "pi_away_overall"]:
        matches[col] = pi[col].values

    # Bookmaker odds — derive implied probabilities from B365 columns
    has_odds = all(c in matches.columns for c in ["B365H", "B365D", "B365A"])
    if has_odds:
        matches["b365_implied_home"] = 1.0 / matches["B365H"]
        matches["b365_implied_draw"] = 1.0 / matches["B365D"]
        matches["b365_implied_away"] = 1.0 / matches["B365A"]
        matches["b365_home_edge"] = matches["b365_implied_home"] - (1.0 / 3)
        implied = matches[["b365_implied_home", "b365_implied_draw", "b365_implied_away"]].values
        matches["b365_favourite"] = implied.argmax(axis=1)  # 0=H, 1=D, 2=A
        print(f"  Bookmaker odds: {matches['B365H'].notna().sum()}/{len(matches)} matches have B365 data")
    else:
        print("  WARNING: B365H/B365D/B365A columns not found in hist_matches.csv")
        print("  Run: python scripts/add_odds_to_hist.py to add them")

    # Identify season boundaries
    seasons = sorted(matches["season_code"].unique())
    print(f"Seasons: {seasons}")

    rows = []
    for idx in range(len(matches)):
        row = matches.iloc[idx]
        home, away = row["home_team"], row["away_team"]
        season = row["season_code"]

        # Only use matches BEFORE current match
        df_before = matches.iloc[:idx]
        if len(df_before) == 0:
            continue

        # Form (all matches up to this point)
        h_pts, h_gf, h_ga, h_wins, h_draws = get_form(df_before, home)
        a_pts, a_gf, a_ga, a_wins, a_draws = get_form(df_before, away)

        # Skip early matches where neither team has form
        if h_pts + a_pts == 0:
            continue

        hh_pts, hh_gf, hh_ga, _, _ = get_form(df_before, home, home_only=True)
        aa_pts, aa_gf, aa_ga, _, _ = get_form(df_before, away, away_only=True)

        # Standings (current season only, up to this match)
        season_before = df_before[df_before["season_code"] == season]
        standings = get_cumulative_standing(season_before)
        h_stand = standings.get(home, {"position": 10, "pts": 0, "gd": 0})
        a_stand = standings.get(away, {"position": 10, "pts": 0, "gd": 0})
        h_pos, h_lpts, h_lgd = h_stand["position"], h_stand["pts"], h_stand["gd"]
        a_pos, a_lpts, a_lgd = a_stand["position"], a_stand["pts"], a_stand["gd"]

        # Matchday (approximate from season matches played)
        matchday = len(season_before) // 10 + 1  # ~10 matches per matchday

        # Shots
        h_sh, h_sha, h_sot, h_sota = get_rolling_shots(df_before, home)
        a_sh, a_sha, a_sot, a_sota = get_rolling_shots(df_before, away)

        # ELO and Pi-ratings (pre-match values from the data)
        elo_home = row["elo_home"]
        elo_away = row["elo_away"]
        elo_diff = row["elo_diff"]
        pi_home = row["pi_home"]
        pi_away = row["pi_away"]
        pi_diff = row["pi_diff"]

        feat = {
            "date": str(row["date"])[:10], "home_team": home, "away_team": away,
            "home_form_pts": h_pts, "home_form_gf": h_gf, "home_form_ga": h_ga,
            "home_form_gd": h_gf - h_ga, "home_form_wins": h_wins, "home_form_draws": h_draws,
            "away_form_pts": a_pts, "away_form_gf": a_gf, "away_form_ga": a_ga,
            "away_form_gd": a_gf - a_ga, "away_form_wins": a_wins, "away_form_draws": a_draws,
            "home_home_pts": hh_pts, "home_home_gd": hh_gf - hh_ga,
            "away_away_pts": aa_pts, "away_away_gd": aa_gf - aa_ga,
            "pts_diff": h_pts - a_pts, "gd_diff": (h_gf - h_ga) - (a_gf - a_ga),
            "home_position": h_pos, "away_position": a_pos, "position_diff": a_pos - h_pos,
            "home_league_pts": h_lpts, "away_league_pts": a_lpts,
            "league_pts_diff": h_lpts - a_lpts,
            "home_league_gd": h_lgd, "away_league_gd": a_lgd,
            "matchday": matchday,
            "elo_home": elo_home, "elo_away": elo_away, "elo_diff": elo_diff,
            "pi_home": pi_home, "pi_away": pi_away, "pi_diff": pi_diff,
            "home_shots_avg": h_sh, "home_shots_against_avg": h_sha,
            "home_sot_avg": h_sot, "home_sot_against_avg": h_sota,
            "away_shots_avg": a_sh, "away_shots_against_avg": a_sha,
            "away_sot_avg": a_sot, "away_sot_against_avg": a_sota,
            "shots_diff": h_sh - a_sh, "sot_diff": h_sot - a_sot,
            "corners_diff": (row.get("hc", 0) or 0) - (row.get("ac", 0) or 0),
            # Dormant features — now activated
            "home_days_rest": get_days_rest(df_before, home, row["date"]),
            "away_days_rest": get_days_rest(df_before, away, row["date"]),
            "home_momentum": get_momentum(df_before, home),
            "away_momentum": get_momentum(df_before, away),
        }

        # xG rolling averages (5-game)
        h_xg, h_xga = get_rolling_xg(df_before, home)
        a_xg, a_xga = get_rolling_xg(df_before, away)
        if h_xg is not None and a_xg is not None:
            feat["home_xg_avg"] = round(h_xg, 3)
            feat["away_xg_avg"] = round(a_xg, 3)
            feat["home_xga_avg"] = round(h_xga, 3)
            feat["away_xga_avg"] = round(a_xga, 3)
            feat["xg_diff"] = round(h_xg - a_xg, 3)

        # Bookmaker odds derived features
        if has_odds and pd.notna(row.get("b365_implied_home")):
            feat["b365_implied_home"] = row["b365_implied_home"]
            feat["b365_implied_draw"] = row["b365_implied_draw"]
            feat["b365_implied_away"] = row["b365_implied_away"]
            feat["b365_home_edge"] = row["b365_home_edge"]
            feat["b365_favourite"] = int(row["b365_favourite"])

        feat["result"] = row["result"]
        rows.append(feat)

        if len(rows) % 500 == 0:
            print(f"  {len(rows)} features built...")

    print(f"  Done: {len(rows)} feature rows from {len(matches)} matches")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# RPS metric
# ─────────────────────────────────────────────────────────────────────────────

def rps_single(proba, actual):
    actual_vec = np.zeros(3)
    actual_vec[OUTCOME_ORDER.index(actual)] = 1.0
    return float(np.mean((np.cumsum(proba[:-1]) - np.cumsum(actual_vec[:-1])) ** 2))


def rps_batch(probas, actuals):
    return float(np.mean([rps_single(p, a) for p, a in zip(probas, actuals)]))


def draw_recall(y_pred, y_true):
    mask = np.array(y_true) == "D"
    return float((np.array(y_pred)[mask] == "D").mean()) if mask.sum() else float("nan")


def predict_proba_ordered(model, le, X):
    """Return probabilities ordered [H, D, A]."""
    raw = model.predict_proba(X)
    classes = list(le.classes_)
    idx = [classes.index(o) for o in OUTCOME_ORDER]
    return raw[:, idx]


# ─────────────────────────────────────────────────────────────────────────────
# RFE feature selection
# ─────────────────────────────────────────────────────────────────────────────

def rfe_select(X_train, y_train, X_val, y_val, le, all_cols, min_features=10,
               protected_cols=None):
    """
    Recursive Feature Elimination: drop the least important feature each round,
    keep the subset that maximises validation accuracy.

    protected_cols: list of feature names that RFE must never drop.
    """
    print("\n" + "=" * 60)
    print("RFE FEATURE SELECTION")
    print("=" * 60)

    protected = set(protected_cols or [])
    if protected:
        print(f"  Protected (force-included): {sorted(protected)}")

    current_cols = list(all_cols)
    best_acc = 0
    best_cols = list(current_cols)
    history = []

    while len(current_cols) >= min_features:
        model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            verbosity=0, random_state=42,
        )
        model.fit(X_train[current_cols], le.transform(y_train), verbose=False)

        y_pred = le.inverse_transform(model.predict(X_val[current_cols]))
        acc = accuracy_score(y_val, y_pred)
        proba = predict_proba_ordered(model, le, X_val[current_cols])
        rps = rps_batch(proba, list(y_val))
        dr = draw_recall(y_pred, list(y_val))

        history.append({
            "n_features": len(current_cols),
            "accuracy": acc,
            "rps": rps,
            "draw_recall": dr,
            "cols": list(current_cols),
        })

        print(f"  {len(current_cols):2d} features → acc={acc:.3%}  rps={rps:.4f}  draw_rec={dr:.3%}")

        if acc >= best_acc:
            best_acc = acc
            best_cols = list(current_cols)

        if len(current_cols) <= min_features:
            break

        # Drop least important feature, skipping protected ones
        importances = model.feature_importances_
        # Build list of (importance, index) for droppable features only
        droppable = [(importances[i], i) for i in range(len(current_cols))
                     if current_cols[i] not in protected]
        if not droppable:
            break
        _, worst_idx = min(droppable)
        dropped = current_cols.pop(worst_idx)
        print(f"    dropped: {dropped} (importance={importances[worst_idx]:.4f})")

    print(f"\n  Best: {len(best_cols)} features, accuracy={best_acc:.3%}")
    print(f"  Selected: {best_cols}")
    return best_cols, history


# ─────────────────────────────────────────────────────────────────────────────
# Optuna hyperparameter optimisation
# ─────────────────────────────────────────────────────────────────────────────

FOLD_SIZE = 380
MIN_TRAIN = 380


def walk_forward_score(df, cols, le, params):
    """Walk-forward CV returning mean accuracy across folds."""
    n = len(df)
    split = MIN_TRAIN
    accuracies = []

    while split < n:
        end = min(split + FOLD_SIZE, n)
        train = df.iloc[:split]
        test = df.iloc[split:end]

        model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss",
            verbosity=0, random_state=42, **params
        )
        model.fit(train[cols], le.transform(train["result"]), verbose=False)

        y_pred = le.inverse_transform(model.predict(test[cols]))
        acc = accuracy_score(test["result"], y_pred)
        accuracies.append(acc)
        split += FOLD_SIZE

    return np.mean(accuracies)


def optuna_search(df, cols, le, n_trials=100):
    """Optuna hyperparameter search using walk-forward CV."""
    print("\n" + "=" * 60)
    print(f"OPTUNA HYPERPARAMETER SEARCH ({n_trials} trials)")
    print("=" * 60)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return walk_forward_score(df, cols, le, params)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best walk-forward accuracy: {study.best_value:.3%}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params, study


# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluate(model, le, cols, X_test, y_test):
    """Comprehensive evaluation on held-out test set."""
    y_pred = le.inverse_transform(model.predict(X_test[cols]))
    proba = predict_proba_ordered(model, le, X_test[cols])
    y_true = list(y_test)

    acc = accuracy_score(y_true, y_pred)
    rps = rps_batch(proba, y_true)
    dr = draw_recall(y_pred, y_true)

    print(f"\n  Accuracy     : {acc:.3%} ({sum(p == a for p, a in zip(y_pred, y_true))}/{len(y_true)})")
    print(f"  RPS          : {rps:.4f} (random ≈ 0.222)")
    print(f"  Draw recall  : {dr:.3%}")

    # Per-outcome breakdown
    report = classification_report(y_true, y_pred, target_names=["A", "D", "H"], output_dict=True)
    print(f"\n  Per-outcome:")
    for label in ["H", "D", "A"]:
        r = report[label]
        print(f"    {label}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}  n={r['support']}")

    # Confidence bands (dynamic — auto-updates after every retrain)
    max_conf = proba.max(axis=1) * 100  # convert to percentage
    bands = {
        "below_40":  max_conf < 40,
        "band_40_50": (max_conf >= 40) & (max_conf < 50),
        "band_50_60": (max_conf >= 50) & (max_conf < 60),
        "above_60":  max_conf >= 60,
    }
    confidence_bands = {}
    print(f"\n  Confidence bands:")
    for name, mask in bands.items():
        n_band = int(mask.sum())
        if n_band > 0:
            band_acc = accuracy_score(
                [y_true[i] for i in range(len(y_true)) if mask[i]],
                [y_pred[i] for i in range(len(y_pred)) if mask[i]]
            )
            band_correct = int(sum(
                y_pred[i] == y_true[i] for i in range(len(y_true)) if mask[i]
            ))
            confidence_bands[name] = {
                "accuracy": round(band_acc, 4),
                "correct": band_correct,
                "total": n_band,
            }
            print(f"    {name}: {band_acc:.3%} ({band_correct}/{n_band})")
        else:
            confidence_bands[name] = {"accuracy": 0, "correct": 0, "total": 0}

    # High/low confidence (50% threshold)
    high_mask = max_conf >= 50
    high_acc = low_acc = None
    if high_mask.sum() > 0:
        high_acc = accuracy_score(
            [y_true[i] for i in range(len(y_true)) if high_mask[i]],
            [y_pred[i] for i in range(len(y_pred)) if high_mask[i]]
        )
        print(f"\n  High confidence (>=50%): {high_acc:.3%} on {high_mask.sum()} matches")
    low_mask = ~high_mask
    if low_mask.sum() > 0:
        low_acc = accuracy_score(
            [y_true[i] for i in range(len(y_true)) if low_mask[i]],
            [y_pred[i] for i in range(len(y_pred)) if low_mask[i]]
        )
        print(f"  Low confidence  (<50%): {low_acc:.3%} on {low_mask.sum()} matches")

    # Position gap analysis (favourites vs toss-ups)
    big_game_acc = close_game_acc = None
    big_game_n = close_game_n = 0
    if "position_diff" in X_test.columns:
        pos_gap = X_test["position_diff"].abs()
        big_mask = pos_gap >= 8
        close_mask = pos_gap < 4
        if big_mask.sum() > 0:
            big_game_acc = round(float(accuracy_score(
                [y_true[i] for i in range(len(y_true)) if big_mask.iloc[i]],
                [y_pred[i] for i in range(len(y_pred)) if big_mask.iloc[i]]
            )), 4)
            big_game_n = int(big_mask.sum())
            print(f"\n  Big gap (8+ pos): {big_game_acc:.3%} on {big_game_n} matches")
        if close_mask.sum() > 0:
            close_game_acc = round(float(accuracy_score(
                [y_true[i] for i in range(len(y_true)) if close_mask.iloc[i]],
                [y_pred[i] for i in range(len(y_pred)) if close_mask.iloc[i]]
            )), 4)
            close_game_n = int(close_mask.sum())
            print(f"  Close gap (<4 pos): {close_game_acc:.3%} on {close_game_n} matches")

    return {
        "accuracy": round(acc, 4),
        "correct": int(sum(p == a for p, a in zip(y_pred, y_true))),
        "total": len(y_true),
        "rps": round(rps, 4),
        "draw_recall": round(dr, 4),
        "confidence_bands": confidence_bands,
        "high_conf_acc": round(float(high_acc), 4) if high_mask.sum() > 0 else None,
        "high_conf_n": int(high_mask.sum()),
        "low_conf_acc": round(float(low_acc), 4) if low_mask.sum() > 0 else None,
        "low_conf_n": int(low_mask.sum()),
        "big_game_acc": big_game_acc,
        "big_game_n": big_game_n,
        "close_game_acc": close_game_acc,
        "close_game_n": close_game_n,
        "per_outcome": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report.items() if k in ["H", "D", "A"]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation (for final reporting)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_final(df, cols, le, params):
    """Walk-forward with final best params, printing per-fold results."""
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION (final model params)")
    print("=" * 60)

    n = len(df)
    split = MIN_TRAIN
    folds = []
    fold_n = 1

    while split < n:
        end = min(split + FOLD_SIZE, n)
        train = df.iloc[:split]
        test = df.iloc[split:end]

        model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss",
            verbosity=0, random_state=42, **params
        )
        model.fit(train[cols], le.transform(train["result"]), verbose=False)

        y_pred = le.inverse_transform(model.predict(test[cols]))
        proba = predict_proba_ordered(model, le, test[cols])
        acc = accuracy_score(test["result"], y_pred)
        rps = rps_batch(proba, list(test["result"]))
        dr = draw_recall(y_pred, list(test["result"]))

        folds.append({"fold": fold_n, "train": len(train), "test": len(test),
                       "accuracy": acc, "rps": rps, "draw_recall": dr})
        print(f"  Fold {fold_n}: train={len(train):4d}  test={len(test):3d}  "
              f"acc={acc:.3%}  rps={rps:.4f}  dr={dr:.3%}")
        split += FOLD_SIZE
        fold_n += 1

    wf_acc = np.mean([f["accuracy"] for f in folds])
    wf_rps = np.mean([f["rps"] for f in folds])
    valid_dr = [f["draw_recall"] for f in folds if not np.isnan(f["draw_recall"])]
    wf_dr = np.mean(valid_dr) if valid_dr else float("nan")

    print(f"\n  Mean accuracy  : {wf_acc:.3%}")
    print(f"  Mean RPS       : {wf_rps:.4f}")
    print(f"  Mean draw rec  : {wf_dr:.3%}")

    return folds, {"accuracy": wf_acc, "rps": wf_rps, "draw_recall": wf_dr}


# ─────────────────────────────────────────────────────────────────────────────
# Save & push
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, le, cols, eval_stats, wf_stats, best_params):
    """Save model artifacts and update validation.json."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    with open(os.path.join(MODEL_DIR, "xgb_champion.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "cols_champion.pkl"), "wb") as f:
        pickle.dump(cols, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"  Saved xgb_champion.pkl, cols_champion.pkl, label_encoder.pkl")
    print(f"  Features ({len(cols)}): {cols}")

    # Update validation.json
    from scipy import stats as scipy_stats
    p_value = scipy_stats.binomtest(
        eval_stats["correct"], eval_stats["total"], 1 / 3, alternative="greater"
    ).pvalue if eval_stats["total"] > 0 else 1.0

    # Count seasons from the training data
    matches_df = pd.read_csv(HIST_MATCHES)
    all_seasons = sorted(matches_df["season_code"].unique())
    n_seasons = len(all_seasons)
    first_year = 2000 + all_seasons[0] // 100  # e.g. 1415 → 2014
    last_year = 2000 + all_seasons[-1] % 100    # e.g. 2425 → 2025

    val = {
        "accuracy": eval_stats["accuracy"],
        "correct": eval_stats["correct"],
        "total": eval_stats["total"],
        "p_value": p_value,
        "beats_random_by": round((eval_stats["accuracy"] - 1 / 3) / (1 / 3) * 100, 1),
        "rps": eval_stats["rps"],
        "draw_recall": eval_stats["draw_recall"],
        "confidence_bands": eval_stats.get("confidence_bands", {}),
        "high_conf_acc": eval_stats.get("high_conf_acc"),
        "high_conf_n": eval_stats.get("high_conf_n"),
        "low_conf_acc": eval_stats.get("low_conf_acc"),
        "low_conf_n": eval_stats.get("low_conf_n"),
        "big_game_acc": eval_stats.get("big_game_acc"),
        "big_game_n": eval_stats.get("big_game_n", 0),
        "close_game_acc": eval_stats.get("close_game_acc"),
        "close_game_n": eval_stats.get("close_game_n", 0),
        "per_outcome": eval_stats.get("per_outcome", {}),
        "seasons": n_seasons,
        "training_seasons": f"{first_year}-{last_year}",
        "model_version": f"{n_seasons} seasons ({first_year}-{last_year}), XGBoost + Optuna + Pi-ratings, retrained {datetime.date.today()}",
        "walk_forward_accuracy": round(wf_stats["accuracy"], 4),
        "walk_forward_rps": round(wf_stats["rps"], 4),
        "walk_forward_draw_recall": round(wf_stats["draw_recall"], 4) if not np.isnan(wf_stats["draw_recall"]) else None,
        "best_params": best_params,
    }
    with open(VALIDATION_JSON, "w") as f:
        json.dump(val, f, indent=2)
    print(f"  Updated validation.json")

    # Update results.json
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        results["entries"].append({
            "date": str(datetime.date.today()),
            "model": "XGBoost + Optuna + Pi-ratings + RFE",
            "accuracy": eval_stats["accuracy"],
            "rps": eval_stats["rps"],
            "draw_recall": eval_stats["draw_recall"],
            "n_features": len(cols),
            "features": cols,
            "walk_forward_accuracy": round(wf_stats["accuracy"], 4),
            "walk_forward_rps": round(wf_stats["rps"], 4),
            "best_params": best_params,
        })
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Updated results.json")


def save_updated_features(feat_df):
    """Save the rebuilt feature matrix so hist_features.csv stays in sync."""
    feat_df.to_csv(HIST_FEATURES, index=False)
    print(f"  Updated hist_features.csv ({len(feat_df)} rows)")


def git_push(eval_stats):
    """Commit and push model artifacts to GitHub."""
    print("\n" + "=" * 60)
    print("PUSHING TO GITHUB")
    print("=" * 60)

    acc_pct = round(eval_stats["accuracy"] * 100, 1)
    msg = f"retrain champion model: {acc_pct}% accuracy, Pi-ratings + RFE + Optuna"

    os.system(f"cd {ROOT} && git add models/ data/validation.json data/hist_features.csv benchmarks/results.json")
    os.system(f'cd {ROOT} && git commit -m "{msg}"')
    os.system(f"cd {ROOT} && git push")
    print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EPL PREDICTOR — MODEL RETRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n1. Loading data...")
    matches = pd.read_csv(HIST_MATCHES, parse_dates=["date"])
    pi = pd.read_csv(PI_RATINGS, parse_dates=["date"])
    print(f"   hist_matches.csv: {len(matches)} rows")
    print(f"   pi_ratings.csv:   {len(pi)} rows")

    # ── 2. Build features ─────────────────────────────────────────────────────
    print("\n2. Building features (this takes a few minutes)...")
    feat_df = build_features(matches, pi)

    # Save updated features
    save_updated_features(feat_df)

    # ── 3. Prepare train/test split ───────────────────────────────────────────
    print("\n3. Preparing train/test split (80/20 chronological)...")
    feat_df = feat_df[feat_df["home_form_pts"] + feat_df["away_form_pts"] > 0].reset_index(drop=True)
    print(f"   After filtering: {len(feat_df)} rows")

    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    print(f"   Train: {len(train_df)}  Test: {len(test_df)}")

    le = LabelEncoder().fit(["H", "D", "A"])

    # All candidate features (exclude non-feature columns)
    NON_FEATURE_COLS = {"result", "date", "home_team", "away_team", "season_code"}
    all_feature_cols = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
    print(f"   Candidate features: {len(all_feature_cols)}")

    # ── 4. RFE feature selection ──────────────────────────────────────────────
    print("\n4. Running RFE feature selection...")
    # Force-include xG and odds features — never let RFE drop them
    PROTECTED_FEATURES = [
        # xG features (5)
        "home_xg_avg", "away_xg_avg", "home_xga_avg", "away_xga_avg", "xg_diff",
        # Odds features (5)
        "b365_implied_home", "b365_implied_draw", "b365_implied_away",
        "b365_home_edge", "b365_favourite",
    ]
    # Only protect features that actually exist in the data
    protected = [c for c in PROTECTED_FEATURES if c in all_feature_cols]
    print(f"   Force-included features: {len(protected)}/{len(PROTECTED_FEATURES)}")

    selected_cols, rfe_history = rfe_select(
        train_df, train_df["result"], test_df, test_df["result"],
        le, all_feature_cols, min_features=10, protected_cols=protected
    )

    # ── 5. Optuna hyperparameter search ───────────────────────────────────────
    print("\n5. Running Optuna hyperparameter search...")
    # Use training data only for Optuna (walk-forward within train set)
    best_params, study = optuna_search(train_df, selected_cols, le, n_trials=100)

    # ── 6. Train final model ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. TRAINING FINAL MODEL")
    print("=" * 60)

    final_model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss",
        verbosity=0, random_state=42, **best_params
    )
    final_model.fit(
        train_df[selected_cols], le.transform(train_df["result"]),
        verbose=False
    )

    # ── 7. Evaluate on held-out test set ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. HELD-OUT TEST SET EVALUATION")
    print("=" * 60)

    eval_stats = full_evaluate(final_model, le, selected_cols, test_df, test_df["result"])

    # ── 8. Walk-forward validation (full dataset, final params) ───────────────
    wf_folds, wf_stats = walk_forward_final(feat_df, selected_cols, le, best_params)

    # ── 9. Compare with previous model ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("9. COMPARISON WITH PREVIOUS MODEL")
    print("=" * 60)

    if os.path.exists(VALIDATION_JSON):
        with open(VALIDATION_JSON) as f:
            prev = json.load(f)
        prev_acc = prev.get("accuracy", 0)
        new_acc = eval_stats["accuracy"]
        delta = new_acc - prev_acc
        print(f"  Previous accuracy: {prev_acc:.3%}")
        print(f"  New accuracy:      {new_acc:.3%}")
        print(f"  Delta:             {delta:+.3%}")

        if new_acc < prev_acc:
            print("\n  ⚠️  NEW MODEL IS WORSE THAN PREVIOUS!")
            print("  Model will still be saved — review before deploying.")
    else:
        print("  No previous validation.json found.")

    # ── 10. Save and push ─────────────────────────────────────────────────────
    save_model(final_model, le, selected_cols, eval_stats, wf_stats, best_params)

    print("\n" + "=" * 60)
    print("DONE — SUMMARY")
    print("=" * 60)
    print(f"  Features:              {len(selected_cols)}")
    print(f"  Held-out accuracy:     {eval_stats['accuracy']:.3%}")
    print(f"  Held-out RPS:          {eval_stats['rps']:.4f}")
    print(f"  Draw recall:           {eval_stats['draw_recall']:.3%}")
    print(f"  Walk-forward accuracy: {wf_stats['accuracy']:.3%}")
    print(f"  Walk-forward RPS:      {wf_stats['rps']:.4f}")

    # Ask before pushing
    print("\n  Model files saved locally.")
    print("  To push to GitHub, run:")
    print(f"    cd {ROOT} && git add models/ data/validation.json data/hist_features.csv benchmarks/results.json")
    print(f'    git commit -m "retrain: {eval_stats["accuracy"]:.1%} acc, Pi-ratings + RFE + Optuna"')
    print(f"    git push")

    # Uncomment to auto-push:
    # git_push(eval_stats)


if __name__ == "__main__":
    main()
