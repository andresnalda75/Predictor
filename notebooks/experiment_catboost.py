"""
experiment_catboost.py — CatBoost vs XGBoost comparison experiment.

Run in Google Colab:

    !git clone https://github.com/andresnalda75/Predictor.git
    %cd Predictor
    !pip install catboost xgboost optuna scikit-learn pandas numpy scipy
    !python notebooks/experiment_catboost.py

Trains CatBoost and LightGBM on the same 36 features as the deployed
XGBoost model. Compares all three using:
  - 80/20 held-out accuracy
  - Walk-forward validation (380-match folds)
  - RPS (Ranked Probability Score)
  - Draw recall
  - Per-outcome precision/recall

Saves results to benchmarks/results.json.
Does NOT overwrite the deployed model — this is a comparison only.
"""

import os
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

try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False
    print("⚠️  CatBoost not installed — run: pip install catboost")

try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False
    print("⚠️  LightGBM not installed — run: pip install lightgbm")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
BENCH_DIR = os.path.join(ROOT, "benchmarks")

HIST_FEATURES = os.path.join(DATA_DIR, "hist_features.csv")
RESULTS_JSON = os.path.join(BENCH_DIR, "results.json")

OUTCOME_ORDER = ["H", "D", "A"]

# Same 36 features as the deployed XGBoost champion model
CHAMPION_COLS = [
    "home_form_pts", "home_form_gf", "home_form_gd", "home_form_wins",
    "away_form_gf", "away_form_gd", "away_form_draws",
    "home_home_pts", "home_home_gd", "away_away_pts", "away_away_gd",
    "pts_diff", "gd_diff", "home_position", "position_diff",
    "league_pts_diff", "home_league_gd", "away_league_gd", "matchday",
    "elo_home", "elo_away", "elo_diff",
    "pi_home", "pi_away", "pi_diff",
    "home_shots_avg", "home_shots_against_avg",
    "home_sot_avg", "home_sot_against_avg",
    "away_shots_avg", "away_shots_against_avg",
    "away_sot_avg", "away_sot_against_avg",
    "shots_diff", "sot_diff", "corners_diff",
]

FOLD_SIZE = 380
MIN_TRAIN = 380


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


def proba_ordered(raw_proba, classes):
    """Reorder probability columns to [H, D, A]."""
    idx = [list(classes).index(o) for o in OUTCOME_ORDER]
    return raw_proba[:, idx]


def full_evaluate(y_pred, y_true, probas_hda):
    """Return dict of all metrics."""
    acc = accuracy_score(y_true, y_pred)
    rps = rps_batch(probas_hda, list(y_true))
    dr = draw_recall(y_pred, list(y_true))

    report = classification_report(y_true, y_pred, target_names=["A", "D", "H"],
                                   output_dict=True, zero_division=0)

    max_conf = probas_hda.max(axis=1)
    high_mask = max_conf >= 0.50
    high_acc = accuracy_score(
        [y_true.iloc[i] for i in range(len(y_true)) if high_mask[i]],
        [y_pred[i] for i in range(len(y_pred)) if high_mask[i]]
    ) if high_mask.sum() > 0 else None

    return {
        "accuracy": round(acc, 4),
        "rps": round(rps, 4),
        "draw_recall": round(dr, 4),
        "high_conf_acc": round(float(high_acc), 4) if high_acc else None,
        "high_conf_n": int(high_mask.sum()),
        "per_outcome": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
            }
            for label in ["H", "D", "A"]
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

def make_xgb(params=None):
    defaults = dict(
        n_estimators=735, learning_rate=0.0126, max_depth=8,
        subsample=0.988, colsample_bytree=0.859, min_child_weight=4,
        gamma=3.775, reg_alpha=2.66e-5, reg_lambda=2.12e-6,
        use_label_encoder=False, eval_metric="mlogloss",
        verbosity=0, random_state=42,
    )
    if params:
        defaults.update(params)
    return xgb.XGBClassifier(**defaults)


def make_catboost(params=None):
    defaults = dict(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, random_seed=42,
        verbose=0, eval_metric="MultiClass",
        auto_class_weights="Balanced",  # helps with draw recall
    )
    if params:
        defaults.update(params)
    return CatBoostClassifier(**defaults)


def make_lgbm(params=None):
    defaults = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced",  # helps with draw recall
        random_state=42, verbose=-1,
    )
    if params:
        defaults.update(params)
    return lgb.LGBMClassifier(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna search for CatBoost
# ─────────────────────────────────────────────────────────────────────────────

def optuna_catboost(train_df, cols, le, n_trials=80):
    """Optuna hyperparameter search for CatBoost using walk-forward CV."""
    print(f"\n  Running Optuna for CatBoost ({n_trials} trials)...")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }
        # Walk-forward score
        n = len(train_df)
        split = MIN_TRAIN
        accs = []
        while split < n:
            end = min(split + FOLD_SIZE, n)
            tr = train_df.iloc[:split]
            te = train_df.iloc[split:end]
            model = CatBoostClassifier(
                verbose=0, random_seed=42, eval_metric="MultiClass",
                auto_class_weights="Balanced", **params
            )
            model.fit(tr[cols], le.transform(tr["result"]))
            preds = le.inverse_transform(model.predict(te[cols]).astype(int))
            accs.append(accuracy_score(te["result"], preds))
            split += FOLD_SIZE
        return np.mean(accs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best CatBoost walk-forward accuracy: {study.best_value:.3%}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study


def optuna_lgbm(train_df, cols, le, n_trials=80):
    """Optuna hyperparameter search for LightGBM using walk-forward CV."""
    print(f"\n  Running Optuna for LightGBM ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }
        n = len(train_df)
        split = MIN_TRAIN
        accs = []
        while split < n:
            end = min(split + FOLD_SIZE, n)
            tr = train_df.iloc[:split]
            te = train_df.iloc[split:end]
            model = lgb.LGBMClassifier(
                class_weight="balanced", random_state=42, verbose=-1, **params
            )
            model.fit(tr[cols], le.transform(tr["result"]))
            preds = le.inverse_transform(model.predict(te[cols]))
            accs.append(accuracy_score(te["result"], preds))
            split += FOLD_SIZE
        return np.mean(accs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best LightGBM walk-forward accuracy: {study.best_value:.3%}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(df, cols, le, make_model_fn, model_name):
    """Walk-forward validation with per-fold reporting."""
    print(f"\n  Walk-forward for {model_name}:")
    n = len(df)
    split = MIN_TRAIN
    folds = []
    fold_n = 1

    while split < n:
        end = min(split + FOLD_SIZE, n)
        train = df.iloc[:split]
        test = df.iloc[split:end]

        model = make_model_fn()
        if "CatBoost" in model_name:
            model.fit(train[cols], le.transform(train["result"]))
            raw_pred = model.predict(test[cols]).astype(int)
            y_pred = le.inverse_transform(raw_pred)
            raw_proba = model.predict_proba(test[cols])
        else:
            model.fit(train[cols], le.transform(train["result"]))
            y_pred = le.inverse_transform(model.predict(test[cols]))
            raw_proba = model.predict_proba(test[cols])

        classes = le.classes_
        probas_hda = proba_ordered(raw_proba, classes)
        acc = accuracy_score(test["result"], y_pred)
        rps = rps_batch(probas_hda, list(test["result"]))
        dr = draw_recall(y_pred, list(test["result"]))

        folds.append({"fold": fold_n, "train": len(train), "test": len(test),
                       "accuracy": acc, "rps": rps, "draw_recall": dr})
        print(f"    Fold {fold_n}: train={len(train):4d}  test={len(test):3d}  "
              f"acc={acc:.3%}  rps={rps:.4f}  dr={dr:.3%}")
        split += FOLD_SIZE
        fold_n += 1

    wf_acc = np.mean([f["accuracy"] for f in folds])
    wf_rps = np.mean([f["rps"] for f in folds])
    valid_dr = [f["draw_recall"] for f in folds if not np.isnan(f["draw_recall"])]
    wf_dr = np.mean(valid_dr) if valid_dr else float("nan")

    print(f"    Mean: acc={wf_acc:.3%}  rps={wf_rps:.4f}  dr={wf_dr:.3%}")
    return {"accuracy": wf_acc, "rps": wf_rps, "draw_recall": wf_dr, "folds": folds}


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EPL PREDICTOR — CATBOOST vs XGBOOST vs LIGHTGBM EXPERIMENT")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n1. Loading data...")
    df = pd.read_csv(HIST_FEATURES)
    # Only keep rows where we have the feature columns
    missing = [c for c in CHAMPION_COLS if c not in df.columns]
    if missing:
        print(f"   ⚠️ Missing columns in hist_features.csv: {missing}")
        print("   Re-run notebooks/retrain_model.py first")
        return
    print(f"   {len(df)} rows, {len(CHAMPION_COLS)} features")

    le = LabelEncoder().fit(["H", "D", "A"])

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"   Train: {len(train_df)}  Test: {len(test_df)}")

    results = {}

    # ── XGBoost (deployed baseline) ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. XGBOOST (deployed model params)")
    print("=" * 70)

    xgb_model = make_xgb()
    xgb_model.fit(train_df[CHAMPION_COLS], le.transform(train_df["result"]), verbose=False)

    xgb_pred = le.inverse_transform(xgb_model.predict(test_df[CHAMPION_COLS]))
    xgb_proba = proba_ordered(xgb_model.predict_proba(test_df[CHAMPION_COLS]), le.classes_)
    xgb_eval = full_evaluate(xgb_pred, test_df["result"], xgb_proba)
    xgb_wf = walk_forward(df, CHAMPION_COLS, le, make_xgb, "XGBoost")

    results["xgboost"] = {"held_out": xgb_eval, "walk_forward": xgb_wf, "params": "deployed"}
    print(f"\n  XGBoost held-out: {xgb_eval['accuracy']:.3%}  rps={xgb_eval['rps']:.4f}  "
          f"dr={xgb_eval['draw_recall']:.3%}")

    # ── CatBoost ──────────────────────────────────────────────────────────────
    if HAVE_CATBOOST:
        print("\n" + "=" * 70)
        print("3. CATBOOST (Optuna tuned, balanced class weights)")
        print("=" * 70)

        cb_params, cb_study = optuna_catboost(train_df, CHAMPION_COLS, le, n_trials=80)

        cb_model = CatBoostClassifier(
            verbose=0, random_seed=42, eval_metric="MultiClass",
            auto_class_weights="Balanced", **cb_params
        )
        cb_model.fit(train_df[CHAMPION_COLS], le.transform(train_df["result"]))

        cb_pred = le.inverse_transform(cb_model.predict(test_df[CHAMPION_COLS]).astype(int))
        cb_proba = proba_ordered(cb_model.predict_proba(test_df[CHAMPION_COLS]), le.classes_)
        cb_eval = full_evaluate(cb_pred, test_df["result"], cb_proba)
        cb_wf = walk_forward(
            df, CHAMPION_COLS, le,
            lambda: CatBoostClassifier(verbose=0, random_seed=42,
                                        eval_metric="MultiClass",
                                        auto_class_weights="Balanced", **cb_params),
            "CatBoost"
        )

        results["catboost"] = {"held_out": cb_eval, "walk_forward": cb_wf,
                                "best_params": cb_params}
        print(f"\n  CatBoost held-out: {cb_eval['accuracy']:.3%}  rps={cb_eval['rps']:.4f}  "
              f"dr={cb_eval['draw_recall']:.3%}")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    if HAVE_LGBM:
        print("\n" + "=" * 70)
        print("4. LIGHTGBM (Optuna tuned, balanced class weights)")
        print("=" * 70)

        lgbm_params, lgbm_study = optuna_lgbm(train_df, CHAMPION_COLS, le, n_trials=80)

        lgbm_model = lgb.LGBMClassifier(
            class_weight="balanced", random_state=42, verbose=-1, **lgbm_params
        )
        lgbm_model.fit(train_df[CHAMPION_COLS], le.transform(train_df["result"]))

        lgbm_pred = le.inverse_transform(lgbm_model.predict(test_df[CHAMPION_COLS]))
        lgbm_proba = proba_ordered(lgbm_model.predict_proba(test_df[CHAMPION_COLS]), le.classes_)
        lgbm_eval = full_evaluate(lgbm_pred, test_df["result"], lgbm_proba)
        lgbm_wf = walk_forward(
            df, CHAMPION_COLS, le,
            lambda: lgb.LGBMClassifier(class_weight="balanced", random_state=42,
                                        verbose=-1, **lgbm_params),
            "LightGBM"
        )

        results["lightgbm"] = {"held_out": lgbm_eval, "walk_forward": lgbm_wf,
                                "best_params": lgbm_params}
        print(f"\n  LightGBM held-out: {lgbm_eval['accuracy']:.3%}  rps={lgbm_eval['rps']:.4f}  "
              f"dr={lgbm_eval['draw_recall']:.3%}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n  {'Model':<15} {'Held-Out':>10} {'WF Acc':>10} {'RPS':>8} "
          f"{'WF RPS':>8} {'Draw Rec':>10} {'Hi-Conf':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for name, r in results.items():
        h = r["held_out"]
        w = r["walk_forward"]
        hc = f"{h['high_conf_acc']:.3%}" if h.get("high_conf_acc") else "n/a"
        print(f"  {name:<15} {h['accuracy']:>10.3%} {w['accuracy']:>10.3%} "
              f"{h['rps']:>8.4f} {w['rps']:>8.4f} "
              f"{h['draw_recall']:>10.3%} {hc:>10}")

    # Per-outcome comparison
    print(f"\n  Per-outcome precision / recall:")
    for name, r in results.items():
        h = r["held_out"]
        print(f"\n  {name}:")
        for label in ["H", "D", "A"]:
            o = h["per_outcome"][label]
            print(f"    {label}: precision={o['precision']:.3f}  recall={o['recall']:.3f}  f1={o['f1']:.3f}")

    # ── Winner ────────────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["walk_forward"]["accuracy"])
    best_wf = results[best_name]["walk_forward"]["accuracy"]
    print(f"\n  🏆 Best walk-forward accuracy: {best_name} ({best_wf:.3%})")

    best_rps_name = min(results, key=lambda k: results[k]["walk_forward"]["rps"])
    best_rps = results[best_rps_name]["walk_forward"]["rps"]
    print(f"  🏆 Best walk-forward RPS: {best_rps_name} ({best_rps:.4f})")

    # ── Save to results.json ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("6. SAVING TO results.json")
    print("=" * 70)

    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            data = json.load(f)
    else:
        data = {"entries": []}

    for name, r in results.items():
        if name == "xgboost":
            continue  # already in results.json
        entry = {
            "date": str(datetime.date.today()),
            "model": f"{name} (experiment, same 36 features)",
            "accuracy": r["held_out"]["accuracy"],
            "rps": r["held_out"]["rps"],
            "draw_recall": r["held_out"]["draw_recall"],
            "n_features": len(CHAMPION_COLS),
            "walk_forward_accuracy": round(r["walk_forward"]["accuracy"], 4),
            "walk_forward_rps": round(r["walk_forward"]["rps"], 4),
            "walk_forward_draw_recall": round(r["walk_forward"]["draw_recall"], 4)
                if not np.isnan(r["walk_forward"]["draw_recall"]) else None,
            "high_conf_acc": r["held_out"].get("high_conf_acc"),
            "high_conf_n": r["held_out"].get("high_conf_n"),
            "per_outcome": r["held_out"]["per_outcome"],
            "best_params": r.get("best_params"),
            "notes": f"Experiment: {name} vs XGBoost on identical 36 features. "
                     f"Class weights balanced to improve draw recall.",
        }
        data["entries"].append(entry)

    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(results) - 1} new entries to results.json")

    # ── Recommendation ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    xgb_wf_acc = results["xgboost"]["walk_forward"]["accuracy"]
    for name, r in results.items():
        if name == "xgboost":
            continue
        wf_acc = r["walk_forward"]["accuracy"]
        delta = wf_acc - xgb_wf_acc
        dr = r["held_out"]["draw_recall"]
        xgb_dr = results["xgboost"]["held_out"]["draw_recall"]
        print(f"\n  {name} vs XGBoost:")
        print(f"    WF accuracy: {delta:+.2%} ({'better' if delta > 0 else 'worse'})")
        print(f"    Draw recall: {dr:.3%} vs {xgb_dr:.3%} "
              f"({'better' if dr > xgb_dr else 'worse'})")
        if delta > 0.005:
            print(f"    → Consider deploying {name}")
        elif delta > -0.005:
            print(f"    → Comparable to XGBoost — check draw recall and RPS")
        else:
            print(f"    → XGBoost remains better")

    print("\n  To deploy a new model, update retrain_model.py to use the winning algorithm.")
    print("  Done.")


if __name__ == "__main__":
    main()
