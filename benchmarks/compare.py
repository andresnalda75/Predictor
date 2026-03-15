"""
compare.py — Walk-forward validation for the EPL Predictor champion model.

Runs two evaluations and prints a comparison against academic benchmarks:

  1. 80/20 held-out split  — reproduces the app.py baseline (accuracy, RPS, draw recall)
  2. Walk-forward validation — expanding window, ~380-match folds (one per season)

XGBoost usage:
  - In Google Colab / Railway:  xgboost is available → uses the real trained model
    for section 1, and retrains XGBoost for each walk-forward fold.
  - Locally (macOS without libomp):  falls back to sklearn's
    HistGradientBoostingClassifier.  Numbers are directionally correct but
    will differ from the deployed XGBoost model.

Usage (from repo root):
    python3 benchmarks/compare.py
    python3 benchmarks/compare.py --update-results   # write stats back to results.json

Colab quick-start:
    !git clone https://github.com/andresnalda75/Predictor.git
    %cd Predictor
    !python benchmarks/compare.py --update-results
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── XGBoost detection ────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    import pickle
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

if not HAVE_XGB:
    from sklearn.ensemble import HistGradientBoostingClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
BENCH_DIR = os.path.join(ROOT, "benchmarks")

HIST_FEATURES = os.path.join(DATA_DIR,  "hist_features.csv")
VALIDATION    = os.path.join(DATA_DIR,  "validation.json")
RESULTS_JSON  = os.path.join(BENCH_DIR, "results.json")

OUTCOME_ORDER = ["H", "D", "A"]   # ordered H→D→A for RPS cumulative sums

# ── RPS (Ranked Probability Score) ────────────────────────────────────────────
# Lower is better.  Perfect = 0.  Random baseline ≈ 0.222 for 3 outcomes.

def rps_single(proba: np.ndarray, actual: str) -> float:
    """RPS for one match.  proba must be ordered [P(H), P(D), P(A)]."""
    actual_vec = np.zeros(3)
    actual_vec[OUTCOME_ORDER.index(actual)] = 1.0
    return float(np.mean((np.cumsum(proba[:-1]) - np.cumsum(actual_vec[:-1])) ** 2))

def rps_batch(probas: np.ndarray, actuals) -> float:
    return float(np.mean([rps_single(p, a) for p, a in zip(probas, actuals)]))

def draw_recall(predicted, actual) -> float:
    pred, act = np.array(predicted), np.array(actual)
    mask = act == "D"
    return float((pred[mask] == "D").mean()) if mask.sum() else float("nan")


# ── Model wrappers ────────────────────────────────────────────────────────────

def load_xgb_model():
    """Load the trained champion model and return (model, le, champ_cols)."""
    with open(os.path.join(MODEL_DIR, "xgb_champion.pkl"),  "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "cols_champion.pkl"), "rb") as f:
        cols = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return model, le, cols


def new_model(use_xgb: bool):
    """Return a fresh unfitted model (for walk-forward retraining)."""
    if use_xgb:
        return xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            verbosity=0, random_state=42,
        )
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=5, random_state=42
    )


def predict_proba_ordered(model, le, X, cols):
    """Return proba array ordered [H, D, A] regardless of model type."""
    raw = model.predict_proba(X[cols])
    if hasattr(le, "classes_"):
        classes = list(le.classes_)          # LabelEncoder: ['A', 'D', 'H']
    else:
        classes = OUTCOME_ORDER              # fallback
    idx = [classes.index(o) for o in OUTCOME_ORDER]
    return raw[:, idx]


def evaluate(model, le, cols, X: pd.DataFrame, y_true) -> dict:
    proba  = predict_proba_ordered(model, le, X, cols)
    y_pred = le.inverse_transform(model.predict(X[cols]))
    y_true = list(y_true)
    return {
        "n":           len(y_true),
        "accuracy":    float(np.mean([p == a for p, a in zip(y_pred, y_true)])),
        "rps":         rps_batch(proba, y_true),
        "draw_recall": draw_recall(y_pred, y_true),
    }


# ── Walk-forward ──────────────────────────────────────────────────────────────
FOLD_SIZE = 380
MIN_TRAIN = 380

def walk_forward(df: pd.DataFrame, cols: list, use_xgb: bool) -> list:
    le = LabelEncoder().fit(["H", "D", "A"])
    folds, n, split, fold_n = [], len(df), MIN_TRAIN, 1

    while split < n:
        end   = min(split + FOLD_SIZE, n)
        train = df.iloc[:split]
        test  = df.iloc[split:end]

        model = new_model(use_xgb)
        if use_xgb:
            model.fit(
                train[cols], le.transform(train["result"]),
                verbose=False,
            )
        else:
            model.fit(train[cols], le.transform(train["result"]))

        m = evaluate(model, le, cols, test, test["result"])
        m.update({"fold": fold_n, "train_rows": len(train), "test_rows": len(test)})
        folds.append(m)
        split += FOLD_SIZE
        fold_n += 1

    return folds


# ── Academic benchmark table ──────────────────────────────────────────────────

def print_academic_table():
    print("\n" + "=" * 60)
    print("ACADEMIC / INDUSTRY BENCHMARKS")
    print("=" * 60)
    benchmarks = [
        ("Bookmaker implied odds",          "~65%",  "Market ceiling"),
        ("LightGBM + xG (academic)",        "~67%",  "Post-match xG, not pre-match"),
        ("CatBoost + Pi-ratings",           "55.8%", "Hvattum & Arntzen-style"),
        ("Our model (XGBoost + Optuna)",    "55.6%", "11 seasons (2014–2025), pre-match only"),
        ("Random baseline",                 "33.3%", "Uniform H/D/A prior"),
    ]
    print(f"  {'Model':<35} {'Accuracy':>9}  Notes")
    print(f"  {'-'*35} {'-'*9}  {'-'*30}")
    for name, acc, note in benchmarks:
        print(f"  {name:<35} {acc:>9}  {note}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-results", action="store_true",
                        help="Write computed stats back to benchmarks/results.json")
    args = parser.parse_args()

    print(f"XGBoost available: {HAVE_XGB}  "
          f"({'using real model' if HAVE_XGB else 'using sklearn proxy — run in Colab for XGBoost results'})")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data …")
    df = pd.read_csv(HIST_FEATURES)
    df = df[df["home_form_pts"] + df["away_form_pts"] > 0].reset_index(drop=True)

    # Load authoritative held-out stats from validation.json
    with open(VALIDATION) as fh:
        val = json.load(fh)

    # Determine feature columns
    if HAVE_XGB:
        champ_model, champ_le, CHAMP_COLS = load_xgb_model()
        print(f"Loaded XGBoost champion model — {len(CHAMP_COLS)} features")
    else:
        # Hardcoded fallback: cols_champion.pkl as of March 2026 retrain
        CHAMP_COLS = [
            "gd_diff", "home_league_pts", "away_league_pts", "league_pts_diff",
            "home_league_gd", "away_league_gd", "matchday",
            "elo_home", "elo_away", "elo_diff",
            "home_shots_avg", "home_shots_against_avg",
            "away_shots_avg", "away_shots_against_avg",
            "shots_diff",
        ]
        # Only keep cols that exist in the dataset
        CHAMP_COLS = [c for c in CHAMP_COLS if c in df.columns]
        print(f"sklearn proxy mode — {len(CHAMP_COLS)} features")

    print(f"Dataset: {len(df)} matches")

    # ── 1. 80/20 held-out split ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1.  80/20 HELD-OUT SPLIT")
    print("=" * 60)

    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    if HAVE_XGB:
        # Evaluate the saved champion model directly (no refit)
        held = evaluate(champ_model, champ_le, CHAMP_COLS, test_df, test_df["result"])
        print(f"  Source        : xgb_champion.pkl (no refit)")
    else:
        # Retrain proxy on same split
        le_proxy = LabelEncoder().fit(["H", "D", "A"])
        proxy = new_model(False)
        proxy.fit(train_df[CHAMP_COLS], le_proxy.transform(train_df["result"]))
        held = evaluate(proxy, le_proxy, CHAMP_COLS, test_df, test_df["result"])
        print(f"  Source        : sklearn proxy (XGBoost unavailable locally)")

    print(f"  Test matches  : {held['n']}")
    print(f"  Accuracy      : {held['accuracy']:.3%}  "
          f"(validation.json: {val['accuracy']:.3%})")
    print(f"  RPS           : {held['rps']:.4f}  (random ≈ 0.222)")
    print(f"  Draw recall   : {held['draw_recall']:.3%}  "
          f"(validation.json: {val['draw_recall']:.3%})")

    # ── 2. Walk-forward ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"2.  WALK-FORWARD  (~{FOLD_SIZE}-match folds, expanding window)")
    print("=" * 60)

    folds = walk_forward(df, CHAMP_COLS, HAVE_XGB)

    hdr = (f"  {'Fold':>4}  {'Train':>6}  {'Test':>5}  "
           f"{'Accuracy':>9}  {'RPS':>7}  {'DrawRec':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in folds:
        dr = f"{f['draw_recall']:.3%}" if not np.isnan(f["draw_recall"]) else "    n/a"
        print(f"  {f['fold']:>4}  {f['train_rows']:>6}  {f['test_rows']:>5}"
              f"  {f['accuracy']:>9.3%}  {f['rps']:>7.4f}  {dr:>8}")

    valid = [f for f in folds if not np.isnan(f["draw_recall"])]
    wf_acc = np.mean([f["accuracy"]    for f in folds])
    wf_rps = np.mean([f["rps"]         for f in folds])
    wf_dr  = np.mean([f["draw_recall"] for f in valid])
    print(f"\n  Mean accuracy  : {wf_acc:.3%}")
    print(f"  Mean RPS       : {wf_rps:.4f}")
    print(f"  Mean draw rec  : {wf_dr:.3%}")

    # ── 3. Academic comparison ─────────────────────────────────────────────────
    print_academic_table()

    # ── 4. Optionally write back to results.json ───────────────────────────────
    if args.update_results:
        if not HAVE_XGB:
            print("\n  ⚠  --update-results skipped: run in Colab for authoritative XGBoost RPS.")
            return

        with open(RESULTS_JSON) as fh:
            data = json.load(fh)

        latest = data["entries"][-1]
        latest["rps"]                      = round(held["rps"],   4)
        latest["walk_forward_accuracy"]    = round(wf_acc,        4)
        latest["walk_forward_rps"]         = round(wf_rps,        4)
        latest["walk_forward_draw_recall"] = round(float(wf_dr),  4)

        with open(RESULTS_JSON, "w") as fh:
            json.dump(data, fh, indent=2)

        print(f"\n  results.json updated  (RPS = {held['rps']:.4f})")


if __name__ == "__main__":
    main()
