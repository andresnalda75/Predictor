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
- `models/` — xgb_champion, xgb_halftime, cols_champion, cols_halftime, label_encoder
- `data/` — hist_matches.csv, hist_features.csv, validation.json, pi_ratings.csv, pi_team_ratings.csv

**Environment variables:**
- `FOOTBALL_DATA_API_KEY` — football-data.org API v4 key (match results, standings, fixtures)
- `APIFOOTBALL_KEY` — API-Football key (injury data). Optional; app degrades gracefully without it. Free tier: 100 req/day. Sign up at api-sports.io

---

## Current Baseline

| Model | Accuracy | Notes |
|---|---|---|
| Pre-match (deployed) | 48.1% | Shown in Predict tab |
| Pre-match (Colab best) | 55.6% | 8 seasons, XGBoost + Optuna |
| Halftime in-game | 55.3% (UI) / 62.4% (git) | Uses HT score + form + ELO |
| Random baseline | 33.3% | 3-outcome coin flip |
| Pro benchmark | 54–56% | Industry standard |

---

## Agent Definitions

### Agent 1 — `model-trainer` (Google Colab)
**Branch:** _(work done in Colab, push directly to main)_
**Goal:** Improve pre-match model accuracy beyond 55.6%
**Tasks:**
- Experiment with additional features (xG, referee, weather, travel distance)
- Try ensemble methods (XGBoost + LightGBM stacking)
- Tune Optuna hyperparameter search budget
- Evaluate on held-out 2025/26 season data
- Export updated pkl files and push to GitHub

### Agent 2 — `data-pipeline`
**Branch:** `data-pipeline`
**Goal:** Keep training data fresh and expand feature set
**Tasks:**
- Script to fetch completed 2025/26 matches from football-data.org and append to hist_matches.csv
- Recalculate ELO ratings dynamically (currently static at 1500 for live matches)
- Add shots data for live matches (currently zeroed out in fetch_current_season)
- Add xG data source if available
- Automate seasonal data updates

### Agent 3 — `frontend-improvements`
**Branch:** `frontend-improvements`
**Goal:** Improve UX and add useful features for users
**Tasks:**
- Mobile responsiveness improvements (nav tabs overflow on small screens)
- Add team search/filter to fixtures tab
- Show form indicators (last 5 results) alongside predictions
- Add a "How accurate have we been this week?" banner on overview
- Dark mode toggle

### Agent 4 — `benchmarking`
**Branch:** `benchmarking`
**Goal:** Track model performance over time with rigorous metrics
**Tasks:**
- Build a prediction log — store every fixture prediction with actual result
- Weekly accuracy report (rolling 10-match accuracy)
- Compare live accuracy vs validation accuracy (detect drift)
- Calibration curve for confidence scores
- Update validation.json automatically after each matchday

### Agent 5 — `deployment`
**Branch:** _(deploy from main only)_
**Goal:** Keep Railway deployment stable and fast
**Tasks:**
- Add health check endpoint (`/health`)
- Cache live API calls (standings, fixtures) with TTL to avoid rate limits
- Environment variable audit (API key should only come from Railway env, not hardcoded)
- Add basic logging for prediction requests
- Monitor memory usage (large CSVs loaded at startup)

---

## Phase Sequence

1. **Stabilise** — fix data pipeline (ELO, shots for live matches), keep deployed model working
2. **Benchmark** — set up prediction logging so we can track live accuracy week by week
3. **Improve model** — retrain with better features, target >56% pre-match
4. **Frontend polish** — mobile, form indicators, search
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
