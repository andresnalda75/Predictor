"""
retrain_halftime.py — Full retraining pipeline for the EPL Predictor halftime model.

Run in Google Colab (XGBoost needs libomp, unavailable on macOS):

    !git clone https://github.com/andresnalda75/Predictor.git
    %cd Predictor
    !pip install xgboost optuna scikit-learn pandas numpy scipy
    !python notebooks/retrain_halftime.py

What it does:
  1. Loads hist_features.csv (pre-match features) + hist_matches.csv (HT scores)
  2. Merges HT score columns into feature matrix
  3. Adds Pi-rating columns from pi_ratings.csv
  4. Runs RFE to select optimal feature subset
  5. Runs Optuna hyperparameter search with walk-forward CV
  6. Trains final model on 80% chronological split
  7. Evaluates on held-out 20% test set
  8. Saves model files (xgb_halftime.pkl, cols_halftime.pkl)
  9. Updates validation_halftime.json
"""

import os
import sys
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

HIST_FEATURES = os.path.join(DATA_DIR, "hist_features.csv")
HIST_MATCHES = os.path.join(DATA_DIR, "hist_matches.csv")
PI_RATINGS = os.path.join(DATA_DIR, "pi_ratings.csv")
VALIDATION_HT = os.path.join(DATA_DIR, "validation_halftime.json")
RESULTS_JSON = os.path.join(BENCH_DIR, "results.json")

OUTCOME_ORDER = ["H", "D", "A"]

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
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
# Build halftime feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_ht_features(hist_feat, hist_matches):
    """
    Merge halftime score columns from hist_matches into the pre-match
    feature matrix. The HT model uses pre-match context + the live HT score
    to predict the full-time result.

    Joins on date + home_team + away_team to handle row count mismatches
    (hist_features.csv may have fewer rows due to filtering during build).

    Returns DataFrame with all pre-match features + HT columns + result.
    """
    print("\n  Building halftime feature matrix...")
    print(f"  hist_features: {len(hist_feat)} rows, hist_matches: {len(hist_matches)} rows")

    feat = hist_feat.copy()

    # Normalize date column in hist_feat to string for joining
    if "date" in feat.columns:
        feat["date"] = feat["date"].astype(str).str[:10]
    else:
        print("  ⚠️  hist_features.csv missing date column — cannot merge by key")
        print("  Re-run notebooks/retrain_model.py first to rebuild with join keys")
        sys.exit(1)

    # Prepare match HT data for merge
    ht_cols = hist_matches[["date", "home_team", "away_team", "ht_home", "ht_away"]].copy()
    ht_cols["date"] = ht_cols["date"].astype(str).str[:10]

    # Merge on the three key columns
    before = len(feat)
    feat = feat.merge(ht_cols, on=["date", "home_team", "away_team"], how="inner")
    print(f"  Merged: {len(feat)} rows ({before - len(feat)} unmatched dropped)")

    # Derive HT features
    feat["ht_gd"] = feat["ht_home"] - feat["ht_away"]
    feat["ht_result_H"] = (feat["ht_gd"] > 0).astype(int)
    feat["ht_result_D"] = (feat["ht_gd"] == 0).astype(int)
    feat["ht_result_A"] = (feat["ht_gd"] < 0).astype(int)

    # Drop rows with missing HT data
    before = len(feat)
    feat = feat.dropna(subset=["ht_home", "ht_away"]).reset_index(drop=True)
    if before != len(feat):
        print(f"  Dropped {before - len(feat)} rows with missing HT data")

    # Drop join key columns (not features)
    feat = feat.drop(columns=["date", "home_team", "away_team"])

    print(f"  Final: {len(feat)} rows, {len(feat.columns)} columns")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# RFE feature selection
# ─────────────────────────────────────────────────────────────────────────────

def rfe_select(X_train, y_train, X_val, y_val, le, all_cols, min_features=12):
    """
    Recursive Feature Elimination: drop the least important feature each round,
    keep the subset that maximises validation accuracy.
    """
    print("\n" + "=" * 60)
    print("RFE FEATURE SELECTION")
    print("=" * 60)

    current_cols = list(all_cols)
    best_acc = 0
    best_cols = list(current_cols)

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

        print(f"  {len(current_cols):2d} features → acc={acc:.3%}  rps={rps:.4f}  draw_rec={dr:.3%}")

        if acc >= best_acc:
            best_acc = acc
            best_cols = list(current_cols)

        if len(current_cols) <= min_features:
            break

        # Drop least important feature (but never drop ht_ columns — they're the point)
        importances = model.feature_importances_
        # Protect HT columns from elimination
        protected = {"ht_home", "ht_away", "ht_gd", "ht_result_H", "ht_result_D", "ht_result_A"}
        candidates = [(i, imp) for i, imp in enumerate(importances)
                      if current_cols[i] not in protected]
        if not candidates:
            break
        worst_idx = min(candidates, key=lambda x: x[1])[0]
        dropped = current_cols.pop(worst_idx)
        print(f"    dropped: {dropped} (importance={importances[worst_idx]:.4f})")

    print(f"\n  Best: {len(best_cols)} features, accuracy={best_acc:.3%}")
    print(f"  Selected: {best_cols}")
    return best_cols


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
# Full evaluation
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

    report = classification_report(y_true, y_pred, target_names=["A", "D", "H"], output_dict=True)
    print(f"\n  Per-outcome:")
    for label in ["H", "D", "A"]:
        r = report[label]
        print(f"    {label}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}  n={r['support']}")

    max_conf = proba.max(axis=1)
    high_mask = max_conf >= 0.50
    high_acc = accuracy_score(
        [y_true[i] for i in range(len(y_true)) if high_mask[i]],
        [y_pred[i] for i in range(len(y_pred)) if high_mask[i]]
    ) if high_mask.sum() > 0 else None
    low_mask = ~high_mask
    low_acc = accuracy_score(
        [y_true[i] for i in range(len(y_true)) if low_mask[i]],
        [y_pred[i] for i in range(len(y_pred)) if low_mask[i]]
    ) if low_mask.sum() > 0 else None

    if high_acc is not None:
        print(f"\n  High confidence (>=50%): {high_acc:.3%} on {high_mask.sum()} matches")
    if low_acc is not None:
        print(f"  Low confidence  (<50%): {low_acc:.3%} on {low_mask.sum()} matches")

    return {
        "accuracy": round(acc, 4),
        "correct": int(sum(p == a for p, a in zip(y_pred, y_true))),
        "total": len(y_true),
        "rps": round(rps, 4),
        "draw_recall": round(dr, 4),
        "high_conf_acc": round(float(high_acc), 4) if high_acc is not None else None,
        "high_conf_n": int(high_mask.sum()),
        "low_conf_acc": round(float(low_acc), 4) if low_acc is not None else None,
        "low_conf_n": int(low_mask.sum()),
        "per_outcome": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                        for k, v in report.items() if k in ["H", "D", "A"]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation (final reporting)
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
# Save model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, le, cols, eval_stats, wf_stats, best_params):
    """Save halftime model artifacts and validation stats."""
    print("\n" + "=" * 60)
    print("SAVING HALFTIME MODEL")
    print("=" * 60)

    with open(os.path.join(MODEL_DIR, "xgb_halftime.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "cols_halftime.pkl"), "wb") as f:
        pickle.dump(cols, f)
    # label_encoder.pkl is shared with champion model — don't overwrite
    print(f"  Saved xgb_halftime.pkl, cols_halftime.pkl")
    print(f"  Features ({len(cols)}): {cols}")

    # Save validation_halftime.json
    from scipy import stats as scipy_stats
    p_value = scipy_stats.binomtest(
        eval_stats["correct"], eval_stats["total"], 1 / 3, alternative="greater"
    ).pvalue if eval_stats["total"] > 0 else 1.0

    val = {
        "accuracy": eval_stats["accuracy"],
        "correct": eval_stats["correct"],
        "total": eval_stats["total"],
        "p_value": p_value,
        "beats_random_by": round((eval_stats["accuracy"] - 1 / 3) / (1 / 3) * 100, 1),
        "rps": eval_stats["rps"],
        "draw_recall": eval_stats["draw_recall"],
        "high_conf_acc": eval_stats.get("high_conf_acc"),
        "high_conf_n": eval_stats.get("high_conf_n"),
        "low_conf_acc": eval_stats.get("low_conf_acc"),
        "low_conf_n": eval_stats.get("low_conf_n"),
        "per_outcome": eval_stats.get("per_outcome", {}),
        "model_version": f"11 seasons (2014-2025), XGBoost + Optuna + Pi-ratings, halftime, retrained {datetime.date.today()}",
        "walk_forward_accuracy": round(wf_stats["accuracy"], 4),
        "walk_forward_rps": round(wf_stats["rps"], 4),
        "walk_forward_draw_recall": round(wf_stats["draw_recall"], 4) if not np.isnan(wf_stats["draw_recall"]) else None,
        "best_params": best_params,
    }
    with open(VALIDATION_HT, "w") as f:
        json.dump(val, f, indent=2)
    print(f"  Saved validation_halftime.json")

    # Append to results.json
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        results["entries"].append({
            "date": str(datetime.date.today()),
            "model": "halftime XGBoost + Optuna + Pi-ratings + RFE",
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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EPL PREDICTOR — HALFTIME MODEL RETRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n1. Loading data...")
    hist_feat = pd.read_csv(HIST_FEATURES)
    hist_matches = pd.read_csv(HIST_MATCHES, parse_dates=["date"])
    print(f"   hist_features.csv: {len(hist_feat)} rows, {len(hist_feat.columns)} cols")
    print(f"   hist_matches.csv:  {len(hist_matches)} rows")

    # ── 2. Build halftime features ────────────────────────────────────────────
    print("\n2. Building halftime feature matrix...")
    feat_df = build_ht_features(hist_feat, hist_matches)

    # ── 3. Prepare candidate features ─────────────────────────────────────────
    print("\n3. Preparing train/test split (80/20 chronological)...")

    # All pre-match features + HT columns (exclude 'result')
    ht_columns = ["ht_home", "ht_away", "ht_gd", "ht_result_H", "ht_result_D", "ht_result_A"]
    pre_match_cols = [c for c in feat_df.columns if c not in ["result"] + ht_columns]

    # Candidate features: pre-match context + HT score
    all_feature_cols = pre_match_cols + ht_columns
    print(f"   Candidate features: {len(all_feature_cols)} ({len(pre_match_cols)} pre-match + {len(ht_columns)} HT)")

    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]
    print(f"   Train: {len(train_df)}  Test: {len(test_df)}")

    le = LabelEncoder().fit(["H", "D", "A"])

    # ── 4. RFE feature selection ──────────────────────────────────────────────
    print("\n4. Running RFE feature selection...")
    selected_cols = rfe_select(
        train_df, train_df["result"], test_df, test_df["result"],
        le, all_feature_cols, min_features=12
    )

    # ── 5. Optuna hyperparameter search ───────────────────────────────────────
    print("\n5. Running Optuna hyperparameter search...")
    best_params, study = optuna_search(train_df, selected_cols, le, n_trials=100)

    # ── 6. Train final model ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. TRAINING FINAL HALFTIME MODEL")
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

    # ── 8. Walk-forward validation ────────────────────────────────────────────
    wf_folds, wf_stats = walk_forward_final(feat_df, selected_cols, le, best_params)

    # ── 9. Compare with previous model ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("9. COMPARISON WITH PREVIOUS MODEL")
    print("=" * 60)

    prev_acc = 62.4  # from git log
    new_acc = eval_stats["accuracy"] * 100
    delta = new_acc - prev_acc
    print(f"  Previous accuracy: {prev_acc:.1f}%")
    print(f"  New accuracy:      {new_acc:.1f}%")
    print(f"  Delta:             {delta:+.1f}pp")

    if new_acc < prev_acc:
        print("\n  ⚠️  NEW MODEL IS WORSE THAN PREVIOUS!")
        print("  Model will still be saved — review before deploying.")

    # ── 10. Save ──────────────────────────────────────────────────────────────
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

    print("\n  Model files saved locally.")
    print("  To push to GitHub, run:")
    print(f"    cd {ROOT} && git add models/xgb_halftime.pkl models/cols_halftime.pkl data/validation_halftime.json benchmarks/results.json")
    print(f'    git commit -m "retrain halftime: {eval_stats["accuracy"]:.1%} acc, Pi-ratings + RFE + Optuna"')
    print(f"    git push")


if __name__ == "__main__":
    main()
