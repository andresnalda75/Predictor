"""
experiment_covid.py — Compare model accuracy with/without COVID season data.

Three experiments using the same feature set (47 existing + 5 xG = 52 candidates,
with 10 force-included: 5 xG + 5 odds):

    A) Baseline: all data (current)
    B) Exclude 1920 only (COVID season, behind closed doors)
    C) Exclude 1920 + 2021 (both COVID-affected seasons)

Uses 50 Optuna trials each (fast). Comparison only — no deployment.

Run in Google Colab:
    !git clone https://github.com/andresnalda75/Predictor.git
    %cd Predictor
    !pip install xgboost optuna scikit-learn pandas numpy
    !python notebooks/experiment_covid.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import shared functions from retrain_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrain_model import (
    build_features, rfe_select, optuna_search, walk_forward_final,
    full_evaluate, predict_proba_ordered, rps_batch, draw_recall,
    OUTCOME_ORDER, HIST_MATCHES, PI_RATINGS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# Force-included features (same as retrain_model.py)
PROTECTED_FEATURES = [
    "home_xg_avg", "away_xg_avg", "home_xga_avg", "away_xga_avg", "xg_diff",
    "b365_implied_home", "b365_implied_draw", "b365_implied_away",
    "b365_home_edge", "b365_favourite",
]

NON_FEATURE_COLS = {"result", "date", "home_team", "away_team", "season_code"}

OPTUNA_TRIALS = 50


def run_experiment(name, feat_df):
    """Run a single experiment: RFE -> Optuna -> evaluate. Returns stats dict."""
    print("\n" + "#" * 70)
    print(f"# EXPERIMENT: {name}")
    print("#" * 70)

    # Filter out rows with no form
    feat_df = feat_df[feat_df["home_form_pts"] + feat_df["away_form_pts"] > 0].reset_index(drop=True)
    print(f"  Usable rows: {len(feat_df)}")

    # 80/20 chronological split
    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    print(f"  Train: {len(train_df)}  Test: {len(test_df)}")

    le = LabelEncoder().fit(["H", "D", "A"])

    all_feature_cols = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]
    protected = [c for c in PROTECTED_FEATURES if c in all_feature_cols]
    print(f"  Candidate features: {len(all_feature_cols)}, protected: {len(protected)}")

    # RFE
    selected_cols, _ = rfe_select(
        train_df, train_df["result"], test_df, test_df["result"],
        le, all_feature_cols, min_features=10, protected_cols=protected
    )

    # Optuna (50 trials)
    best_params, _ = optuna_search(train_df, selected_cols, le, n_trials=OPTUNA_TRIALS)

    # Train final model
    final_model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss",
        verbosity=0, random_state=42, **best_params
    )
    final_model.fit(train_df[selected_cols], le.transform(train_df["result"]), verbose=False)

    # Evaluate on held-out test set
    print(f"\n  HELD-OUT TEST SET ({name}):")
    eval_stats = full_evaluate(final_model, le, selected_cols, test_df, test_df["result"])

    # Walk-forward
    _, wf_stats = walk_forward_final(feat_df, selected_cols, le, best_params)

    return {
        "name": name,
        "total_rows": len(feat_df),
        "train": len(train_df),
        "test": len(test_df),
        "n_features": len(selected_cols),
        "features": selected_cols,
        "accuracy": eval_stats["accuracy"],
        "rps": eval_stats["rps"],
        "draw_recall": eval_stats["draw_recall"],
        "wf_accuracy": round(wf_stats["accuracy"], 4),
        "wf_rps": round(wf_stats["rps"], 4),
        "best_params": best_params,
    }


def main():
    print("=" * 70)
    print("COVID EXCLUSION EXPERIMENT")
    print("=" * 70)

    # -- Load data ----------------------------------------------------------
    print("\nLoading data...")
    matches = pd.read_csv(HIST_MATCHES, parse_dates=["date"])
    pi = pd.read_csv(PI_RATINGS, parse_dates=["date"])
    print(f"  hist_matches.csv: {len(matches)} rows")
    print(f"  pi_ratings.csv:   {len(pi)} rows")

    seasons = sorted(matches["season_code"].unique())
    print(f"  Seasons: {seasons}")
    for sc in seasons:
        n = (matches["season_code"] == sc).sum()
        print(f"    {sc}: {n} matches")

    # -- Build features on full dataset -------------------------------------
    # build_features() only uses PRIOR matches for each row, so removing a
    # season's rows after building doesn't affect other seasons' features.
    print("\nBuilding features on full dataset (this takes a few minutes)...")
    feat_all = build_features(matches, pi)

    # Reconstruct season_code in feat_df for filtering
    season_map = matches[["date", "home_team", "away_team", "season_code"]].copy()
    season_map["date"] = season_map["date"].dt.strftime("%Y-%m-%d")
    feat_all["date_str"] = pd.to_datetime(feat_all["date"]).dt.strftime("%Y-%m-%d")
    feat_all = feat_all.merge(
        season_map.rename(columns={"date": "date_str"}),
        on=["date_str", "home_team", "away_team"],
        how="left",
    )
    feat_all = feat_all.drop(columns=["date_str"])
    print(f"  Total feature rows: {len(feat_all)}")
    print(f"  Season codes in features: {sorted(feat_all['season_code'].dropna().unique())}")

    # -- Experiment A: Baseline (all data) ----------------------------------
    feat_a = feat_all.drop(columns=["season_code"]).reset_index(drop=True)
    result_a = run_experiment("A: Baseline (all data)", feat_a)

    # -- Experiment B: Exclude 1920 -----------------------------------------
    feat_b = feat_all[feat_all["season_code"] != 1920].drop(columns=["season_code"]).reset_index(drop=True)
    result_b = run_experiment("B: Exclude 1920", feat_b)

    # -- Experiment C: Exclude 1920 + 2021 ----------------------------------
    feat_c = feat_all[~feat_all["season_code"].isin([1920, 2021])].drop(columns=["season_code"]).reset_index(drop=True)
    result_c = run_experiment("C: Exclude 1920 + 2021", feat_c)

    # -- Summary ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Rows':>5} {'Feat':>4} {'Acc':>8} {'RPS':>8} {'Draw':>8} {'WF Acc':>8} {'WF RPS':>8}")
    print("-" * 100)
    for r in [result_a, result_b, result_c]:
        print(f"{r['name']:<30} {r['total_rows']:>5} {r['n_features']:>4} "
              f"{r['accuracy']:>7.3%} {r['rps']:>8.4f} {r['draw_recall']:>7.3%} "
              f"{r['wf_accuracy']:>7.3%} {r['wf_rps']:>8.4f}")

    # Delta vs baseline
    print("\n  Delta vs baseline:")
    for r in [result_b, result_c]:
        da = r["accuracy"] - result_a["accuracy"]
        dr = r["rps"] - result_a["rps"]
        dd = r["draw_recall"] - result_a["draw_recall"]
        print(f"    {r['name']}: acc={da:+.3%}  rps={dr:+.4f}  draw={dd:+.3%}")

    print("\nDone! No models were saved -- comparison only.")


if __name__ == "__main__":
    main()
