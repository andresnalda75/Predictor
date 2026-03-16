# Changelog

All model and data changes are recorded here. Each deployed model gets a version, name, and accuracy record.

---

## 2026-03-15 — Prediction Logging + Predictions Tab + Model Versioning

### Prediction Logging System
- **New:** `data/predictions.db` — SQLite database storing every prediction with outcome tracking
- **New:** `POST /api/log_prediction` — log a prediction (match_date, teams, probabilities, confidence, model version)
- **New:** `GET /api/track_record` — all predictions + summary stats (total, resolved, correct, accuracy %, breakdown by model_version)
- **New:** `DELETE /api/track_record` — clear all predictions
- **New:** `scripts/reconcile_predictions.py` — reconcile pending predictions against actual results from football-data.org API
- **Duplicate check:** unique on `match_date + home_team + away_team + model_version` — same model predicts once per match, new model version can log fresh prediction alongside old ones

### Predictions Tab (replaces old Predict + Track Record tabs)
- Fixture cards grouped by matchday (GW 30/31/32/33 — next 4 matchdays shown)
- 3 card states: **unpredicted** (Predict button) → **awaiting result** (probabilities + confidence badge) → **resolved** (result + ✅ or ❌)
- Per-matchday "Predict All" button + individual Predict buttons per fixture
- Team names shown everywhere — no more generic Home/Away labels
- Tab state persists when navigating away and back
- Auto-logs every prediction to `predictions.db` on click
- Design doc: `docs/PREDICT_TAB_REDESIGN.md`

### Reconciliation GitHub Action
- `.github/workflows/reconcile.yml` + manual `workflow_dispatch`
- Schedule: Sat/Sun 15:00, 18:00, 21:00 UTC · Mon 23:00 UTC · Tue/Wed 22:00 UTC
- `FOOTBALL_DATA_API_KEY` added to GitHub Actions secrets
- Fetches finished EPL matches, fills in `actual_outcome` + `correct` for all unresolved predictions (all model versions scored)

### Model Versioning
- v1.0 "Kickoff" (55.6%, early March 2026) — base XGBoost + Pi-ratings + ELO
- v2.0 "Odds On" (57.79%, 2026-03-14) — added B365 bookmaker odds
- v3.0 "Sharp" (59.11%, 2026-03-15) — added xG, RFE to 26 features, draw recall 3.66%. **Current champion.**
- Every prediction tagged with model version for per-version accuracy tracking

---

## v3.0 "Sharp" — 2026-03-15 (Current Champion)

### Pre-Match Model (xgb_champion)
- **Accuracy:** 59.11% on 834 holdout matches
- **RPS:** 0.1895 (random ≈ 0.222)
- **Draw recall:** 3.66% (up from 0%)
- **Walk-forward:** 55.70%
- **High confidence (≥60%):** 70.1% accuracy
- **Features:** 26 (selected by RFE from 47)
- **Key features:** B365 implied odds, xG (home_xg_avg, away_xg_avg, home_xga_avg, away_xga_avg, xg_diff), Pi-ratings (pi_home, pi_away, pi_diff), ELO, form (exp. decay)
- **Optuna:** 300 trials, best at trial 247. n_estimators=443, lr=0.196, max_depth=7, subsample=0.766, colsample=0.920, min_child_weight=10, gamma=3.076, reg_alpha=4.888, reg_lambda=2.676
- **Training data:** 11 EPL seasons (2014–2025), 3,332 training matches
- **What changed from v2.0:** Added 5 xG features from Understat. RFE pruned 36 → 26 features (leaner model). 300 Optuna trials (vs 100 in v2.0). days_rest, momentum, and most shot stats dropped by RFE.

### Halftime Model (xgb_halftime)
- **Accuracy:** 60.6% on 834 holdout matches
- **Walk-forward:** 63.6%
- **Features:** 35 (HT score, HT goal difference, HT result flags, form, ELO, Pi-ratings)
- **Notes:** 43.4% of EPL games flip from HT result to FT result — model accounts for this.

---

## v2.0 "Odds On" — 2026-03-14

### Pre-Match Model (xgb_champion)
- **Accuracy:** 57.79% on 834 holdout matches
- **Features:** 36
- **What changed from v1.0:** Added 11 B365 implied probability features (implied probs, home edge, favourite, overround). First model with bookmaker odds. Pi-ratings integrated.
- **Known limitation:** Trained on real B365 historical odds, but live predictions used ELO-derived implied probabilities as a proxy.

---

## v1.0 "Kickoff" — Early March 2026

### Pre-Match Model (xgb_champion)
- **Accuracy:** 55.6% on holdout
- **Features:** 36 (form pts/GF/GA/GD/wins/draws, home/away form, league position, league points, GD, ELO, Pi-ratings, rolling shots/SOT, position diff, matchday)
- **Algorithm:** Base XGBoost + Optuna hyperparameter tuning
- **Draw recall:** 0%
- **Notes:** Baseline model. Pi-ratings + ELO + form (exp. decay). Platt scaling applied for confidence calibration. p-value 5×10⁻²⁶ vs random baseline.

---

## Data

- `hist_matches.csv` — 11 seasons of EPL results (2014–2025), 4,180 rows
- `hist_features.csv` — engineered features aligned to hist_matches.csv, 4,166 rows
- `validation.json` — held-out test set metrics for current champion
- `pi_ratings.csv` — Pi-ratings for all historical matches
- `pi_team_ratings.csv` — current Pi-ratings per team (for live predictions)

---

_Add a new versioned entry above the Data section whenever a new model is deployed._
