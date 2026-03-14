# EPL Predictor — Master Plan

## Project Overview

Flask web app deployed on Railway that predicts EPL match outcomes using XGBoost models trained on 8 seasons of historical EPL data (2017–2025, no COVID). Live data is fetched from the football-data.org API v4.

**Stack:** Flask · XGBoost · Pandas · Gunicorn · Railway · football-data.org API

**Key files:**
- `app.py` — Flask backend, feature engineering, all API routes
- `templates/index.html` — full frontend (8 tabs, vanilla JS)
- `live_data.py` — football-data.org integration (results, standings, fixtures)
- `injury_data.py` — API-Football integration (pre-match injury counts)
- `scripts/calculate_pi_ratings.py` — Pi-ratings calculation from hist_matches.csv
- `scripts/add_odds_to_hist.py` — Download B365 odds from football-data.co.uk into hist_matches.csv
- `notebooks/retrain_model.py` — Colab retraining pipeline (champion model)
- `notebooks/retrain_halftime.py` — Colab retraining pipeline (halftime model)
- `notebooks/experiment_catboost.py` — CatBoost vs XGBoost vs LightGBM comparison
- `models/` — xgb_champion, xgb_halftime, cols_champion, cols_halftime, label_encoder
- `data/` — hist_matches.csv, hist_features.csv, validation.json, pi_ratings.csv, pi_team_ratings.csv

**Environment variables:**
- `FOOTBALL_DATA_API_KEY` — football-data.org API v4 key (match results, standings, fixtures)
- `APIFOOTBALL_KEY` — API-Football key (injury data). Optional; app degrades gracefully without it. Free tier: 100 req/day. Sign up at api-sports.io

---

## Current Baseline

| Model | Accuracy | Features | Test Matches | Notes |
|---|---|---|---|---|
| Pre-match (deployed) | 57.07% | 36 | 834 | XGBoost + Optuna + Pi-ratings + exp. decay |
| Halftime in-game | 60.6% | 35 | 834 | Uses HT score + form + ELO + Pi-ratings |
| Random baseline | 33.3% | — | — | 3-outcome coin flip |
| Pro benchmark | 54–56% | — | — | Industry standard |

**Pre-match model details (retrained 2026-03-14):**
- RPS: 0.1951 (random ≈ 0.222)
- Draw recall: 1.57% (critical weakness, draws are 23% of outcomes)
- High confidence (>=50%): 63.5% accuracy on 499 matches
- Big game (8+ pos gap): 56.5% on 278 matches
- Close game (<4 pos gap): 54.8% on 554 matches
- Walk-forward accuracy: 55.01%
- Key features: Pi-ratings, ELO, form (exp. decay), league position, shots

---

## Model Improvement Roadmap

### TIER 1 — HIGH IMPACT, DO NEXT

| # | Improvement | Data Source | Est. Impact | Status |
|---|---|---|---|---|
| 1 | **Bookmaker odds as features** — football-data.co.uk CSV has B365H/B365D/B365A columns. Add implied probability features from odds. | football-data.co.uk (free CSV) | +2-3% accuracy | IN PROGRESS — B365 odds merged into hist_matches.csv, 5 derived features (implied probs, home edge, favourite) added to retrain pipeline. Awaiting retrain. |
| 2 | **xG from Understat** — free historical xG for EPL teams. Better than shots on target for measuring chance quality. | understat.com (free, scrape) | +1-2% accuracy | Not started |
| 3 | **Ensemble voting** — combine XGBoost + CatBoost + LightGBM predictions via meta-learner (stacking). | Internal | +1-2% accuracy | CatBoost experiment running |
| 4 | **Auto-retrain weekly** — accuracy improves +0.32% per week as season data accumulates. Set up GitHub Action to retrain on schedule. | GitHub Actions | +0.3%/week cumulative | Not started |

### TIER 2 — MEDIUM IMPACT, MORE WORK

| # | Improvement | Notes |
|---|---|---|
| 5 | **Player market values from Transfermarkt** | Free, proven predictive of team quality in academic literature |
| 6 | **Poisson regression model** | Model goals directly, derive draw probability from goal distribution. Better for draws than classification |
| 7 | **Ordinal classification** | Treat H/D/A as ordered classes (A < D < H), not independent categories |
| 8 | **Separate draw model** | Binary classifier for draw vs decisive, then combine with main model |
| 9 | **SHAP explainability** | Show users WHY each prediction was made (feature contributions) |
| 10 | **Siamese network** | Encode matchup directly as a team pair rather than two separate feature vectors |

### TIER 3 — LONGER TERM

| # | Improvement | Notes |
|---|---|---|
| 11 | **StatsBomb event data** | Passes, pressures, dribbles. Expensive but powerful |
| 12 | **Tracking data** | Player positions. Not publicly available yet |

### WHAT DIDN'T WORK — DO NOT RETRY WITHOUT NEW APPROACH

| Attempt | Result | Root Cause |
|---|---|---|
| ELO hardcoded to 1500 for current season | Flat home bias predictions | Fixed by using hist_df for ELO lookup |
| Retraining with 2025/26 season data without real ELO | Degraded 57.07% → 56.47% | ELO signal lost when all teams start at 1500 |
| Reducing features below 36 via aggressive RFE | Accuracy drops below 54% | Features are complementary, not redundant |
| Optuna stochasticity | Best run 57.07%, subsequent runs 56.4-56.5% | Optuna is stochastic — don't assume best run is repeatable |
| CatBoost without balanced weights | Similar to XGBoost, no draw improvement | Need `auto_class_weights="Balanced"` or custom loss |
| CatBoost with balanced weights | Walk-forward peaked at 52-53%, well below XGBoost 55% | Not worth running again without new features |
| LightGBM experiment | Cancelled — deprioritised in favour of bookmaker odds features | Odds features expected to have higher impact than algorithm swap |
| Halftime retrain | 60.9% vs 60.6% deployed, marginal gain | Walk-forward 63.6% suggests promise, but held-out gain too small to deploy |

### TESTING PROTOCOL — REQUIRED BEFORE DEPLOYING ANY NEW MODEL

1. Must beat **57.07%** on held-out test set (80/20 chronological split)
2. Must have walk-forward mean accuracy **> 54%**
3. Must not increase home bias (check H recall stays **below 85%**)
4. Run at least **3 Optuna trials**, take the best
5. Compare RPS score (lower is better, target **< 0.195**)
6. Check draw recall — any improvement over 1.57% is a bonus
7. Update `validation.json`, `results.json`, and `CHANGELOG.md` before pushing

---

## Agent Status (as of 2026-03-14)

### Agent 1 — `frontend-improvements` ✅ DONE
**All tasks complete:**
- Mobile responsiveness (hamburger nav, touch-friendly tabs)
- Team search/filter on fixtures tab
- Form dots (last 5 results W/D/L)
- PWA with service worker
- Injury indicators
- Dark mode toggle
- H2H summary in predictions

### Agent 2 — `data-pipeline` ✅ DONE
**All tasks complete:**
- Pi-ratings calculated and integrated (scripts/calculate_pi_ratings.py)
- Exponential decay weighting in get_form()
- hist_features.csv rebuilt with 36 features + join keys (date, home_team, away_team)
- Pi-team-ratings.csv for live predictions

### Agent 3 — `model-improvements` (this agent)
**Branch:** `model-improvements`
**Goal:** Improve model accuracy beyond 57.07%
**Completed:**
- Retrained champion model: 55.6% → 57.07% (+1.47pp)
- Pi-ratings integrated (pi_home, pi_away, pi_diff)
- RFE feature selection (41 candidates → 36 selected)
- Optuna hyperparameter search with walk-forward CV
- Fixed cols_champion.pkl mismatch (15 → 36 features)
- Fixed away_form_ga bug (was using a_gf instead of a_ga)
- Fixed validation.json accuracy source (now reads from file, not recalculated)
- Created retrain_halftime.py pipeline
- Fixed halftime merge (date+team join keys instead of index alignment)
- Fixed halftime endpoint 500 error (feat_dict missing 16 features)
- Created CatBoost vs XGBoost vs LightGBM experiment script

### Agent 4 — `benchmarking` (not started)
**Goal:** Value betting / Odds API integration
**Tasks:**
- Odds API integration for value bet detection
- Prediction logging (store every prediction with actual result)
- Weekly accuracy report (rolling 10-match accuracy)
- Calibration curve for confidence scores

### Agent 5 — `deployment` ✅ DONE
**Completed:**
- /health endpoint
- Caching for live API calls
- Basic logging for prediction requests
- Competitive analysis complete

---

## Dormant Features (in app.py, ready for next retrain)

These features are computed at prediction time but NOT yet in the trained model's feature set. Include them as candidates in the next retrain:

| Feature | Function | Description |
|---|---|---|
| `home_days_rest` / `away_days_rest` | `get_days_rest()` | Days since last match |
| `home_momentum` / `away_momentum` | `get_momentum()` | PPG delta (recent 5 vs prior 5) |
| `h2h_home_wins` / `h2h_draws` / `h2h_away_wins` | `get_h2h()` | Head-to-head record |
| `home_injuries` / `away_injuries` | `get_injury_count()` | Pre-match injury count (API-Football) |

---

## Next Session Priorities

1. **A3:** Bookmaker odds features (Tier 1 #1) — biggest expected accuracy gain
2. **A3:** xG from Understat (Tier 1 #2)
3. **A3:** Ensemble stacking with CatBoost/LightGBM results (Tier 1 #3)
4. **A4:** Value betting / Odds API integration (not started)
5. **A3:** Halftime retrain with draw-weighted objective (address 1.57% draw recall)
6. **A5:** Run `benchmarks/compare.py` in Colab for authoritative RPS benchmarks
7. **A3:** Include dormant features (rest, momentum, H2H, injuries) in next retrain

---

## Phase Sequence

1. ~~**Stabilise** — fix data pipeline (ELO, shots for live matches), keep deployed model working~~ ✅
2. **Benchmark** — set up prediction logging so we can track live accuracy week by week
3. ~~**Improve model** — retrain with better features, target >56% pre-match~~ ✅ (57.07%)
4. ~~**Frontend polish** — mobile, form indicators, search~~ ✅
5. **Monetise** — see MARKETING.md

---

## Agent Rules

- Never retrain and push models without first validating accuracy on a held-out test set
- Never push directly to main without testing locally first (except Colab model pushes)
- All model changes must come with an updated CHANGELOG.md entry
- Never hardcode API keys in source code — use Railway environment variables
- Keep requirements.txt unpinned unless a version conflict arises
- The halftime model (`xgb_halftime.pkl`) and champion model (`xgb_champion.pkl`) are independent — changes to one do not require retraining the other
- Before any data pipeline change, verify that hist_matches.csv and hist_features.csv stay in sync (same row count and date alignment)
- validation.json is the source of truth for displayed accuracy — app.py reads VALIDATED_ACCURACY from it at startup
- Follow the Testing Protocol before deploying any new model (see Model Improvement Roadmap)
