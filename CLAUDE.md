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
- `notebooks/retrain_model.py` — Colab retraining pipeline (champion model)
- `notebooks/retrain_halftime.py` — Colab retraining pipeline (halftime model)
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
| Halftime in-game | 60.6% | 19 | 834 | Uses HT score + form + ELO |
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

1. **A4:** Value betting / Odds API integration (not started)
2. **A3:** Try CatBoost and LightGBM as alternatives to XGBoost
3. **A3:** Halftime retrain with draw-weighted objective (address 1.57% draw recall)
4. **A5:** Run `benchmarks/compare.py` in Colab for authoritative RPS benchmarks
5. **A3:** Include dormant features (rest, momentum, H2H, injuries) in next retrain

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
