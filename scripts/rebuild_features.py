#!/usr/bin/env python3
"""
Rebuild hist_features.csv from hist_matches.csv + pi_ratings.csv.

Runs only the feature-building portion of retrain_model.py — no XGBoost,
no model training, no Optuna. Safe to run locally (no libomp needed).

Usage:
    python scripts/rebuild_features.py
"""

import os
import sys
import pandas as pd

# Add notebooks/ to path so we can import build_features
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks"))

# Import only the feature engineering functions (no xgboost dependency)
# We can't import retrain_model directly because it imports xgboost at top level.
# Instead, re-implement the minimal loader + call build_features via exec.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
HIST_MATCHES = os.path.join(DATA_DIR, "hist_matches.csv")
PI_RATINGS = os.path.join(DATA_DIR, "pi_ratings.csv")
HIST_FEATURES = os.path.join(DATA_DIR, "hist_features.csv")


def main():
    print("=" * 60)
    print("REBUILD HIST_FEATURES.CSV")
    print("=" * 60)

    # We need to import build_features without importing xgboost.
    # Extract just the feature engineering code.
    import importlib.util
    import types

    # Create a mock xgboost module so the import doesn't fail
    mock_xgb = types.ModuleType("xgboost")
    mock_xgb.XGBClassifier = None
    sys.modules["xgboost"] = mock_xgb

    mock_optuna = types.ModuleType("optuna")
    mock_optuna.logging = types.ModuleType("optuna.logging")
    mock_optuna.logging.set_verbosity = lambda x: None
    mock_optuna.logging.WARNING = 30
    mock_optuna.create_study = None
    sys.modules["optuna"] = mock_optuna
    sys.modules["optuna.logging"] = mock_optuna.logging

    from retrain_model import build_features, HIST_MATCHES, PI_RATINGS, HIST_FEATURES

    # Load data
    print("\n1. Loading data...")
    matches = pd.read_csv(HIST_MATCHES, parse_dates=["date"])
    pi = pd.read_csv(PI_RATINGS, parse_dates=["date"])
    print(f"   hist_matches.csv: {len(matches)} rows")
    print(f"   pi_ratings.csv:   {len(pi)} rows")

    # Build features
    print("\n2. Building features (this takes a few minutes)...")
    feat_df = build_features(matches, pi)

    # Save
    print(f"\n3. Saving hist_features.csv...")
    feat_df.to_csv(HIST_FEATURES, index=False)
    print(f"   Saved {len(feat_df)} rows, {len(feat_df.columns)} columns")

    # Report new columns
    print(f"\n4. Columns ({len(feat_df.columns)}):")
    for c in feat_df.columns:
        print(f"   {c}")

    # Sample of new features
    new_cols = ["home_days_rest", "away_days_rest", "home_momentum", "away_momentum",
                "home_xg_avg", "away_xg_avg", "home_xga_avg", "away_xga_avg", "xg_diff"]
    existing = [c for c in new_cols if c in feat_df.columns]
    if existing:
        print(f"\n5. Sample of new features:")
        sample = feat_df[feat_df[existing[0]].notna()].tail(10)
        print(sample[["date", "home_team", "away_team"] + existing].to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
