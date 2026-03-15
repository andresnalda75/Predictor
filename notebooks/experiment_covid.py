"""
experiment_covid.py — COVID season exclusion experiment.

Compares three training scenarios:
  A) Baseline: all data (current behaviour)
  B) Exclude 1920 season (COVID behind-closed-doors)
  C) Exclude 1920 + 2021 seasons (both COVID-affected)

Does NOT deploy anything — prints a comparison table.

Run in Google Colab:
    !git clone https://github.com/andresnalda75/Predictor.git
    %cd Predictor
    !pip install xgboost optuna scikit-learn pandas numpy
    !python notebooks/experiment_covid.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import shared functions from retrain_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrain_model import (
    build_features,
    rps_batch,
    draw_recall,
    predict_proba_ordered,
    walk_forward_score,
    full_evaluate,
    walk_forward_final,
    FOLD_SIZE,
    MIN_TRAIN,
    OUTCOME_ORDER,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
HIST_MATCHES = os.path.join(DATA_DIR, "hist_matches.csv")
PI_RATINGS = os.path.join(DATA_DIR, "pi_ratings.csv")

OPTUNA_TRIALS = 50  # Fewer than full retrain — this is an experiment


def run_scenario(name, matches, pi, le, n_trials=OPTUNA_TRIALS):
    """Run a full train/evaluate cycle for one scenario. Returns dict of results."""
    print(f"\n{'=' * 70}")
    print(f"  SCENARIO: {name}")
    print(f"  Matches: {len(matches)} | Seasons: {sorted(matches['season_code'].unique())}")
    print(f"{'=' * 70}")

    # Build features
    print("  Building features...")
    feat_df = build_features(matches, pi)
    feat_df = feat_df[feat_df["home_form_pts"] + feat_df["away_form_pts"] > 0].reset_index(drop=True)
    print(f"  Feature rows: {len(feat_df)}")

    # 80/20 chronological split
    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Candidate features
    NON_FEATURE_COLS = {"result", "date", "home_team", "away_team", "season_code"}
    all_cols = [c for c in feat_df.columns if c not in NON_FEATURE_COLS]

    # Skip RFE for speed — use all features (RFE is deterministic given same data,
    # and we want to isolate the effect of COVID exclusion, not feature selection)
    selected_cols = all_cols

    # Optuna search (walk-forward on training set)
    print(f"  Optuna search ({n_trials} trials)...")

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
        return walk_forward_score(train_df, selected_cols, le, params)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    print(f"  Best walk-forward acc: {study.best_value:.3%}")

    # Train final model
    model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss",
        verbosity=0, random_state=42, **best_params
    )
    model.fit(train_df[selected_cols], le.transform(train_df["result"]), verbose=False)

    # Evaluate on holdout
    print(f"\n  Holdout evaluation:")
    eval_stats = full_evaluate(model, le, selected_cols, test_df, test_df["result"])

    # Walk-forward on full dataset
    wf_folds, wf_stats = walk_forward_final(feat_df, selected_cols, le, best_params)

    return {
        "name": name,
        "total_matches": len(matches),
        "feature_rows": len(feat_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "n_features": len(selected_cols),
        "holdout_accuracy": eval_stats["accuracy"],
        "holdout_rps": eval_stats["rps"],
        "holdout_draw_recall": eval_stats["draw_recall"],
        "wf_accuracy": round(wf_stats["accuracy"], 4),
        "wf_rps": round(wf_stats["rps"], 4),
        "high_conf_acc": eval_stats.get("high_conf_acc"),
        "high_conf_n": eval_stats.get("high_conf_n"),
        "best_params": best_params,
    }


def main():
    print("=" * 70)
    print("  COVID EXCLUSION EXPERIMENT")
    print("  Comparing: all data vs exclude 1920 vs exclude 1920+2021")
    print("=" * 70)

    # Load data
    matches = pd.read_csv(HIST_MATCHES, parse_dates=["date"])
    pi = pd.read_csv(PI_RATINGS, parse_dates=["date"])
    le = LabelEncoder().fit(["H", "D", "A"])

    print(f"\nFull dataset: {len(matches)} matches")
    print(f"Seasons: {sorted(matches['season_code'].unique())}")
    covid_1920 = (matches["season_code"] == 1920).sum()
    covid_2021 = (matches["season_code"] == 2021).sum()
    print(f"COVID rows: 1920={covid_1920}, 2021={covid_2021}, total={covid_1920 + covid_2021}")

    # Define scenarios
    scenarios = [
        ("A) Baseline (all data)", None),
        ("B) Exclude 1920", [1920]),
        ("C) Exclude 1920 + 2021", [1920, 2021]),
    ]

    results = []
    for name, exclude_seasons in scenarios:
        if exclude_seasons:
            mask = ~matches["season_code"].isin(exclude_seasons)
            m = matches[mask].reset_index(drop=True)
            p = pi[mask].reset_index(drop=True)
        else:
            m = matches.copy()
            p = pi.copy()
        result = run_scenario(name, m, p, le)
        results.append(result)

    # Comparison table
    print("\n")
    print("=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Scenario':<30} {'Matches':>8} {'Train':>6} {'Test':>5} "
          f"{'Holdout':>8} {'RPS':>7} {'Draw%':>7} {'WF Acc':>7}")
    print("-" * 90)

    baseline_acc = results[0]["holdout_accuracy"]
    for r in results:
        delta = r["holdout_accuracy"] - baseline_acc
        delta_str = f" ({delta:+.2%})" if delta != 0 else ""
        print(f"{r['name']:<30} {r['total_matches']:>8} {r['train_rows']:>6} {r['test_rows']:>5} "
              f"{r['holdout_accuracy']:>7.2%}{delta_str:>9} {r['holdout_rps']:>7.4f} "
              f"{r['holdout_draw_recall']:>6.1%} {r['wf_accuracy']:>7.2%}")

    # High confidence comparison
    print(f"\n{'Scenario':<30} {'HighConf Acc':>12} {'HighConf N':>11}")
    print("-" * 55)
    for r in results:
        hca = r.get("high_conf_acc")
        hcn = r.get("high_conf_n", 0)
        print(f"{r['name']:<30} {hca:>11.2%} {hcn:>11}")

    # Recommendation
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)
    best = max(results, key=lambda r: r["holdout_accuracy"])
    print(f"  Best holdout accuracy: {best['name']} ({best['holdout_accuracy']:.2%})")
    best_wf = max(results, key=lambda r: r["wf_accuracy"])
    print(f"  Best walk-forward:     {best_wf['name']} ({best_wf['wf_accuracy']:.2%})")
    best_rps = min(results, key=lambda r: r["holdout_rps"])
    print(f"  Best RPS:              {best_rps['name']} ({best_rps['holdout_rps']:.4f})")

    if best["name"] == results[0]["name"]:
        print("\n  → Baseline wins. COVID data does not hurt — keep it.")
    else:
        print(f"\n  → {best['name']} wins. Consider excluding COVID data in next retrain.")
        print(f"    Delta vs baseline: {best['holdout_accuracy'] - baseline_acc:+.2%}")

    # Save results
    out_path = os.path.join(DATA_DIR, "covid_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
