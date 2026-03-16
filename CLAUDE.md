# EPL Predictor — Master Plan

## Project Overview

Flask web app deployed on Railway that predicts EPL match outcomes using XGBoost models trained on 11 seasons of historical EPL data (2014–2025). Live data is fetched from the football-data.org API v4.

**Stack:** Flask · XGBoost · Pandas · Gunicorn · Railway · football-data.org API

**Key files:**
- `app.py` — Flask backend, feature engineering, all API routes, prediction logging
- `templates/index.html` — full frontend (8 tabs, vanilla JS)
- `live_data.py` — football-data.org integration (results, standings, fixtures)
- `injury_data.py` — API-Football integration (pre-match injury counts)
- `scripts/calculate_pi_ratings.py` — Pi-ratings calculation from hist_matches.csv
- `scripts/add_odds_to_hist.py` — Download B365 odds from football-data.co.uk into hist_matches.csv
- `scripts/reconcile_predictions.py` — match prediction outcomes against actual results
- `notebooks/retrain_model.py` — Colab retraining pipeline (champion model)
- `notebooks/retrain_halftime.py` — Colab retraining pipeline (halftime model)
- `notebooks/experiment_catboost.py` — CatBoost vs XGBoost vs LightGBM comparison
- `models/` — xgb_champion, xgb_halftime, cols_champion, cols_halftime, label_encoder
- `data/` — hist_matches.csv, hist_features.csv, validation.json, pi_ratings.csv, pi_team_ratings.csv, predictions.db

**Environment variables:**
- `FOOTBALL_DATA_API_KEY` — football-data.org API v4 key (match results, standings, fixtures)
- `APIFOOTBALL_KEY` — API-Football key (injury data). Optional; app degrades gracefully without it. Free tier: 100 req/day. Sign up at api-sports.io
- `ODDS_API_KEY` — The Odds API key (real bookmaker odds for live predictions). Optional; falls back to ELO-derived proxy without it. Free tier: 500 req/month. Sign up at the-odds-api.com

---

## Model Version History

Every deployed model gets a version number, name, and accuracy record. Prediction logging tags every prediction with its model version so we can track live accuracy per model.

| Version | Name | Accuracy | RPS | Features | Deploy Date | Key Changes |
|---|---|---|---|---|---|---|
| **v3.0** | **Sharp** | **59.11%** | **0.1895** | 26 | 2026-03-15 | Added 5 xG features (Understat). RFE pruned 36 → 26 features. 300 Optuna trials (best at trial 247). Draw recall 3.66%. **Current champion.** |
| v2.0 | Odds On | 57.79% | — | 36 | 2026-03-14 | Added 11 B365 implied probability features (home edge, favourite, overround). First model with bookmaker odds. Pi-ratings integrated. |
| v1.0 | Kickoff | 55.6% | — | 36 | early March 2026 | Base XGBoost + Optuna. Pi-ratings, ELO, form (exp. decay), shots, position diff. Draw recall 0%. |

**Versioning rules for future models:**
- Increment major version (v4.0, v5.0…) for each new champion deployment
- Every model must record: version, name, accuracy, RPS, feature count, deploy date, key changes
- Prediction logging tags every prediction with the model version that produced it (stored in `predictions.db`)
- Retired models stay in this table for reference — never delete rows
- When deploying a new model, update the defaults in `_init_predictions_db()` and `api_log_prediction()` in app.py

---

## Prediction Logging

SQLite-based system that logs every prediction the app makes so we can track live accuracy over time. This is separate from the holdout validation — it measures real-world performance on future matches.

**Database:** `data/predictions.db` (created automatically on app startup)

**Schema (`predictions` table):**

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `match_date` | TEXT | Match date (YYYY-MM-DD) |
| `home_team` | TEXT | Home team name |
| `away_team` | TEXT | Away team name |
| `predicted_outcome` | TEXT | H, D, or A |
| `prob_home` | REAL | Model probability for home win |
| `prob_draw` | REAL | Model probability for draw |
| `prob_away` | REAL | Model probability for away win |
| `confidence` | REAL | Max probability (confidence score) |
| `actual_outcome` | TEXT | H, D, or A (NULL until reconciled) |
| `correct` | INTEGER | 1 if correct, 0 if wrong (NULL until reconciled) |
| `model_version` | TEXT | e.g. "v3.0" — links to Model Version History |
| `model_name` | TEXT | e.g. "Sharp" |
| `model_deployed` | TEXT | Deploy date of the model that made this prediction |
| `created_at` | TEXT | ISO timestamp when prediction was logged |

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/api/log_prediction` | POST | Log a prediction. Required JSON: `match_date`, `home_team`, `away_team`, `predicted_outcome`, `prob_home`, `prob_draw`, `prob_away`, `confidence`. Optional: `model_version`, `model_name`, `model_deployed` (default to current champion). Returns `{id, status}`. |
| `/api/track_record` | GET | Returns all logged predictions + summary stats: total, resolved, correct, accuracy %, and breakdown by `model_version`. |

**Reconciliation:** `scripts/reconcile_predictions.py`
- Finds predictions where `actual_outcome IS NULL` and `match_date < today`
- Fetches finished EPL matches from football-data.org API
- Matches by (date, home_team, away_team) and fills in `actual_outcome` + `correct`
- Run manually or via cron: `python scripts/reconcile_predictions.py`
- Requires `FOOTBALL_DATA_API_KEY` environment variable
- Includes team name mapping (API names → short names used in predictions)

**Duplicate logic:** unique on `match_date + home_team + away_team + model_version`. Same model version can only predict a match once. When a new model is deployed (e.g. v4.0), it can log a fresh prediction for the same match alongside the existing v3.0 entry.

**Workflow:**
1. User requests a prediction → app checks if (match_date, home_team, away_team, current model_version) exists. If yes, return existing prediction. If no, run model and log.
2. After matchday, run `reconcile_predictions.py` to fill in actual results for ALL prediction rows matching that match (all model versions get scored).
3. `/api/track_record` shows running accuracy — default view shows most recent prediction per match, advanced view shows all model versions.
4. When a new model is deployed, update defaults in app.py — old predictions keep their original model tags.

---

## Current Baseline

| Model | Accuracy | Features | Test Matches | Notes |
|---|---|---|---|---|
| Pre-match (deployed) | 59.11% | 26 | 834 | XGBoost + Optuna + Pi-ratings + B365 odds + xG + exp. decay, 11 seasons (2014–2025) |
| Halftime in-game | 60.6% | 35 | 834 | Uses HT score + form + ELO + Pi-ratings |
| Random baseline | 33.3% | — | — | 3-outcome coin flip |
| Pro benchmark | 54–56% | — | — | Industry standard |

**Pre-match model details (retrained 2026-03-15):**
- RPS: 0.1895 (random ≈ 0.222)
- Draw recall: 3.66% (up from 0%, still weak but improving)
- High confidence (>=60%): 70.1% accuracy
- Close game (<4 pos gap): 55.2%
- Walk-forward accuracy: 55.70%
- 26 features selected by RFE (down from 47 — leaner model is better)
- Key features: B365 implied odds, xG (5 features), Pi-ratings, ELO, form (exp. decay)
- Best params: n_estimators=443, lr=0.196, max_depth=7, subsample=0.766, colsample=0.920, min_child_weight=10, gamma=3.076, reg_alpha=4.888, reg_lambda=2.676

---

## Model Improvement Roadmap

### TIER 1 — HIGH IMPACT, DO NEXT

| # | Improvement | Data Source | Est. Impact | Status |
|---|---|---|---|---|
| 1 | **Bookmaker odds as features** — football-data.co.uk CSV has B365H/B365D/B365A columns. Add implied probability features from odds. | football-data.co.uk (free CSV) | +2-3% accuracy | ✅ DONE — 57.07% → 57.79% (+0.72pp). 11 new features from B365 odds (implied probs, home edge, favourite, overround). **Known limitation:** trained on real B365 historical odds, but live predictions use ELO-derived implied probabilities as a proxy (`get_implied_odds()` in app.py). Live accuracy may be slightly below 57.79%. To fix: implement The Odds API (A4 — value betting) for real pre-match odds. |
| 2 | **xG from Understat** — free historical xG for EPL teams. Better than shots on target for measuring chance quality. | understat.com (free, scrape) | +1-2% accuracy | ✅ DONE — 5 xG features added, all survived RFE. Part of 59.11% champion. |
| 3 | **FIFA/EA FC player ratings** — aggregate by role (GK/DEF/MID/FWD) per team. 9 features: home/away att, def, mid, overall + fifa_overall_diff. FC 25 real ratings from Kaggle (nyagami dataset). | Kaggle (free) | +1-2% accuracy | ❌ FAILED — 57.31% holdout (vs 59.11% champion). 6 of 9 FIFA features survived RFE (home_fifa_att, home_fifa_overall, away_fifa_def, away_fifa_mid, away_fifa_overall, fifa_overall_diff) but didn't improve accuracy. Draw recall regressed to 0.0%. WF 56.23%. Pipeline built (`scripts/fetch_fifa_ratings.py`) but features not deployed. |
| 4 | **Transfermarkt squad market values** — strong proxy for team quality and depth. Proven predictive in academic literature. | transfermarkt.com (free, scrape) | +1-2% accuracy | Next after FIFA results confirmed |
| 5 | **Team formation data** — 4-3-3 vs 5-4-1 etc affects defensive/offensive patterns. Available for current season. | football-data.org API | +0.5-1% accuracy | Not started |
| 6 | **Ensemble voting** — combine XGBoost + CatBoost + LightGBM predictions via meta-learner (stacking). | Internal | +1-2% accuracy | Deprioritised — CatBoost underperformed (52-53% WF), LightGBM cancelled |
| 7 | **Auto-retrain weekly** — accuracy improves +0.32% per week as season data accumulates. Set up GitHub Action to retrain on schedule. | GitHub Actions | +0.3%/week cumulative | Not started |

### TIER 2 — MEDIUM IMPACT, MORE WORK

| # | Improvement | Notes |
|---|---|---|
| 8 | **Poisson regression model** | Model goals directly, derive draw probability from goal distribution. Better for draws than classification |
| 9 | **Ordinal classification** | Treat H/D/A as ordered classes (A < D < H), not independent categories |
| 10 | **Separate draw model** | Binary classifier for draw vs decisive, then combine with main model |
| 11 | **SHAP explainability** | Show users WHY each prediction was made (feature contributions) |
| 12 | **Siamese network** | Encode matchup directly as a team pair rather than two separate feature vectors |
| 13 | **Manager tenure** | New manager bounce is statistically proven. Scrape from Wikipedia or football-data.org |
| 14 | **Weather data** | Precipitation and temperature at match location. Free historical via OpenWeatherMap API |

### DRAW IMPROVEMENT STRATEGIES (ordered by priority)

Current draw recall is 0% — the model never predicts draws. This is the single biggest weakness.

**Why draws are hard:**
- Draws are 23% of EPL matches but the model predicts almost none
- XGBoost maximises overall accuracy — predicting never-draw costs only 23% but gains H/A precision
- Bookmaker odds features made it worse — the market also rarely prices draws as favourite
- Our `shots_against` features already capture defensive strength partially but lack relative context
- Our current 47 features include `home_shots_against_avg` and `home_sot_against_avg` (defensive shots allowed) but these are absolute rolling averages, not relative strength ratings

| Priority | Strategy | Est. Impact | Complexity |
|---|---|---|---|
| 1 | **Dixon-Coles P(draw) as hybrid feature** — run a Dixon-Coles Poisson model alongside XGBoost, use its P(draw) output as an additional input feature. Keeps XGBoost accuracy gains while adding draw intelligence. | Draw recall +10-15% | Medium — need to implement Dixon-Coles in Python, compute per-match P(draw) for all historical matches |
| 2 | **Custom draw threshold** — instead of `argmax(H,D,A)`, predict draw when `P(draw) > 0.28`. 5 lines of code. Test immediately after next retrain. | Draw recall +5-10% | Trivial — post-processing only, no retrain needed |
| 3 | **Draw-specific features** — add features that correlate specifically with draws: `elo_gap` (absolute ELO difference, small gap = more likely draw), `b365_draw_odds` (bookmaker draw odds < 3.5 signals likely draw), `h2h_draw_rate` (historical draw rate between these two teams), `both_low_scoring` (both teams average under 1.2 goals per game) | Draw recall +5-8% | Low — derived from existing data |
| 4 | **Dixon-Coles attack/defense ratings** — add 4 new features: `home_attack_rating` (goals scored relative to opposition defense strength), `home_defense_rating` (goals conceded relative to opposition attack strength), `away_attack_rating`, `away_defense_rating`. These are relative strength ratings (like Pi-ratings but split attack/defense). Different from rolling averages — Arsenal scoring 3 vs Ipswich means less than scoring 3 vs Man City. | Accuracy +1-2%, draw recall +5% | Medium — need to implement iterative rating system |
| 5 | **Full Dixon-Coles parallel model** — replace XGBoost for draw predictions only, use XGBoost for H/A. Ensemble the two. | Draw recall +15-20% | High — two separate models, need calibration |
| 6 | **SMOTE oversampling for draws** — synthetically oversample draw examples in training data. Last resort — risk to overall accuracy. | Draw recall +5-10% | Low — but may degrade H/A accuracy |
| 7 | **Referee features** — referee ID already in football-data.co.uk CSV. Some referees produce more cards/chaos = more draws. Zero effort to add. | Draw recall +2-5% | Trivial — data already available |
| 8 | **Match stakes features** — points gap to relegation/title/CL spots. Low stakes late season = more draws. Already have standings data to derive this. | Draw recall +3-5% | Low — derived from existing features |
| 9 | **Travel/fatigue proxy** — `home_days_rest` and `away_days_rest` are ALREADY BUILT as dormant features in app.py (`get_days_rest()`). Just need to activate in next retrain. Zero development cost. | Draw recall +2-3% | Trivial — already implemented, just activate |

### TIER 3 — LONGER TERM

| # | Improvement | Notes |
|---|---|---|
| 15 | **StatsBomb event data** | Passes, pressures, dribbles. Expensive but powerful |
| 16 | **Tracking data** | Player positions. Not publicly available yet |
| 17 | **Social media sentiment** | Twitter/X pre-match sentiment. Noisy signal but novel |
| 18 | **VAEP player ratings** | Event-based value metric, requires StatsBomb data |
| 19 | **Weather live** | Current weather forecast for upcoming fixtures. Free APIs available |

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
| xG alone without odds | 57.67% holdout — did not beat 57.79% champion | xG adds value but not enough to displace B365 odds features alone |
| COVID exclusion (1920 and/or 2021) | Hurts accuracy −1.2% to −1.5% | Model learns from all data including COVID anomalies; removing seasons shrinks training set. Keep all 11 seasons. |
| CatBoost balanced class weights | Walk-forward peaked at 52–53% | Well below XGBoost 55% — ruled out without new features |
| days_rest / momentum features | Dropped by RFE (2026-03-15) | Not predictive enough — xG and odds features dominate |
| 100 Optuna trials | Missed best params | Best found at trial 247 of 300 — always run 300+ trials |
| FIFA ratings alone (31 features) | 57.31% holdout, −1.80pp vs 59.11% champion | 6 of 9 FIFA features survived RFE but Optuna found shallow trees (max_depth=3) vs champion's deep trees (max_depth=7). Walk-forward CV Optuna finds conservative params. |
| Combined FIFA+Transfermarkt+match stakes+referee (37 features) | 58.27% via direct 300-trial Optuna, −0.84pp vs 59.11% champion | New features survive RFE but Optuna converges at trial 71, suggesting feature set ceiling. Referee features dropped by RFE entirely. |
| 5-season training window | Not tried, deprioritised | ELO/Pi-ratings already encode recency. GitHub Action compound gains are better ROI than windowing experiments. |

### TESTING PROTOCOL — REQUIRED BEFORE DEPLOYING ANY NEW MODEL

1. Must beat **59.11%** on held-out test set (80/20 chronological split)
2. Must have walk-forward mean accuracy **> 55%**
3. Must not increase home bias (check H recall stays **below 85%**)
4. Run at least **300 Optuna trials** (best params found at trial 247 — 100 is not enough)
5. Compare RPS score (lower is better, target **< 0.190**)
6. Check draw recall — any improvement over 3.66% is a bonus
7. Update `validation.json`, `results.json`, and `CHANGELOG.md` before pushing
8. **Live odds now available:** The Odds API integrated. Live predictions use real odds when available, ELO-derived proxy as fallback.

---

## Agent Status (as of 2026-03-15)

### Agent 1 — `frontend-improvements` ✅ DONE
**All tasks complete:**
- Mobile responsiveness (hamburger nav, touch-friendly tabs)
- Team search/filter on fixtures tab
- Form dots (last 5 results W/D/L)
- PWA with service worker
- Injury indicators
- Dark mode toggle
- H2H summary in predictions
- Nav reordered: Fixtures | Predict | Performance | Teams | Live | H2H | Table | Methodology
- Performance tab consolidated from 4 tabs (Stats, Accuracy, Confidence, About)
- Methodology tab created from About page
- Tab persistence on refresh via localStorage
- Fixtures pre-cached at startup
- Live odds badges on fixtures (⚡ icon, white text)
- Table tab: form dots, sortable columns, column order Team | Form | P | PTS | W | D | L | GF | GA | GD
- Loading skeletons on all tabs

### Agent 2 — `data-pipeline` ✅ DONE
**All tasks complete:**
- Pi-ratings calculated and integrated (scripts/calculate_pi_ratings.py)
- Exponential decay weighting in get_form()
- hist_features.csv rebuilt with 36 features + join keys (date, home_team, away_team)
- Pi-team-ratings.csv for live predictions

### Agent 3 — `model-improvements` ✅ CHAMPION DEPLOYED
**Goal:** Improve model accuracy beyond 57.79%
**Result:** 59.11% accuracy (+1.32pp), deployed 2026-03-15
**Completed:**
- Retrained champion model: 55.6% → 57.07% → 57.79% → **59.11%** (+3.51pp total)
- xG features added and survived RFE: home_xg_avg, away_xg_avg, home_xga_avg, away_xga_avg, xg_diff
- RFE found 26 optimal features (down from 47 — leaner model is better)
- 300 Optuna trials (best params at trial 247 — 100 would have missed it)
- Draw recall improved: 0.0% → 3.66%
- Pi-ratings integrated (pi_home, pi_away, pi_diff)
- Fixed cols_champion.pkl mismatch (15 → 36 features)
- Fixed away_form_ga bug (was using a_gf instead of a_ga)
- Fixed validation.json accuracy source (now reads from file, not recalculated)
- Created retrain_halftime.py pipeline
- Fixed halftime merge (date+team join keys instead of index alignment)
- Fixed halftime endpoint 500 error (feat_dict missing 16 features)
- Created CatBoost vs XGBoost vs LightGBM experiment script
- Created COVID exclusion experiment script
- **Dropped by RFE:** days_rest, momentum, most shot stats, most form features

### Agent 4 — `benchmarking` (in progress)
**Goal:** Value betting / Odds API integration
**Completed:**
- The Odds API integrated — 19–20 matches live, 476/500 requests remaining
- `/api/performance` endpoint live
- All hardcoded accuracy stats replaced with dynamic values
- Prediction logging system: SQLite `predictions.db`, `/api/log_prediction`, `/api/track_record`, `scripts/reconcile_predictions.py`
**Remaining:**
- Value bet detection logic
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

These features are computed at prediction time but NOT yet in the trained model's feature set:

| Feature | Function | Description | Status |
|---|---|---|---|
| `home_days_rest` / `away_days_rest` | `get_days_rest()` | Days since last match | ❌ Tested 2026-03-15, dropped by RFE |
| `home_momentum` / `away_momentum` | `get_momentum()` | PPG delta (recent 5 vs prior 5) | ❌ Tested 2026-03-15, dropped by RFE |
| `h2h_home_wins` / `h2h_draws` / `h2h_away_wins` | `get_h2h()` | Head-to-head record | Not yet tested |
| `home_injuries` / `away_injuries` | `get_injury_count()` | Pre-match injury count (API-Football) | Not yet tested |

---

## Next Session Priorities

1. **A1+A2:** Predict Tab Redesign — replace free-form dropdowns with real fixture cards, auto-log predictions. See `docs/PREDICT_TAB_REDESIGN.md`
2. **A3:** GitHub Action — weekly auto-retrain, +0.32%/week compound gain
3. **A4:** Value betting — `ODDS_API_KEY` ready, The Odds API already integrated
4. **A3:** Dixon-Coles P(draw) hybrid feature — est. +10–15% draw recall
5. **A3:** Custom draw threshold — 5 lines of code, test post-retrain
6. **A3:** Try time-weighted training — weight recent seasons more heavily
7. **A1:** Teams table mobile column fix

---

## Phase Sequence

1. ~~**Stabilise** — fix data pipeline (ELO, shots for live matches), keep deployed model working~~ ✅
2. **Benchmark** — set up prediction logging so we can track live accuracy week by week
3. ~~**Improve model** — retrain with better features, target >56% pre-match~~ ✅ (59.11%)
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

---

## DATA AUDIT & DISCLOSURE PLAN

**Disclosure principle:** Every number shown to users must have a clear answer to: *'accuracy on what data, over how many seasons, for which teams?'* If we can't answer that clearly, we shouldn't show the number. This was flagged by the Reddit community and is critical for credibility before any marketing push.

### AUDIT TASKS

#### 1. Map every stat displayed in the app to its data source

| UI Element | Data Source | Answer |
|---|---|---|
| Validation tab stats | `validation.json` → `/api/validation` | ✅ All from 20% chronological holdout (834 matches, last ~2 seasons of 8) |
| Team accuracy table | `test_df` (holdout only) → `/api/teams` | ✅ Holdout only — ~83 games/team is correct (834 total ÷ 20 teams ≈ 83). Not a bug. |
| Hero section accuracy | `VALIDATED_ACCURACY` from `validation.json` → `/api/overview` | ✅ Dynamic, reads holdout accuracy. Now labelled "Holdout accuracy" |
| Halftime model stats | Calculated at startup: `xgb_halftime` evaluated on same holdout split | ✅ Same 834-match holdout, computed live at app startup |
| Form dots on fixtures | `get_form_list(team, n=5)` → last 5 matches from `live_df` | ✅ Last 5 matches, no decay. Different from form features (which use exp. decay) |
| ELO ratings | Pre-computed in `hist_matches.csv` (2014–2025). Live uses last known value, fallback 1500. | ✅ Covers all 11 seasons. Current-season teams default to last known historical ELO. |

#### 2. 3-season discrepancy — RESOLVED

- **Footer**: Was showing hardcoded '57.1%' → ✅ Fixed to '57.8%' (already done in prior commit)
- **834 matches**: ✅ Confirmed — 20% chronological holdout of 4,166 usable matches (after form filter) = 834 test matches
- **~83 games per team**: ✅ Not a bug — 834 holdout matches ÷ 20 teams ≈ 83 appearances. Teams appear in both home/away, so each team gets ~83 matches in the holdout
- **Leeds shows 7 games**: ✅ Correct — Leeds were relegated early in the holdout window, so only 7 of their matches fall in the test period. They have more in training data.
- **The "3-season" confusion**: The holdout covers ~2 seasons' worth of matches (the most recent 20%). Teams with fewer EPL seasons overall (promoted/relegated) have fewer test matches.

#### 3. Data completeness per team

- **Full 8+ seasons**: Arsenal, Chelsea, Everton, Liverpool, Man City, Man Utd, Tottenham, Crystal Palace, West Ham, Brighton, Bournemouth, Southampton, Wolves, Leicester, Newcastle, Burnley
- **Promoted/relegated**: Leeds (3 EPL seasons in data), Ipswich (1), Luton (1), Nott'm Forest (3), Sheffield United (3), Sunderland (1), Brentford (4)
- **Reliability**: Teams with <3 seasons in training data have fewer examples for the model to learn from. Flagged with `*` badge in UI.
- Season counts now use `season_code` column (EPL seasons, not calendar years) for accuracy.

#### 4. UI disclosures — DONE

| Fix | Description | Status |
|---|---|---|
| Dataset source labels | Overview: "20% holdout · 11 seasons", "on held-out test data" | ✅ DONE |
| Team accuracy table | "Seasons" column present, sorted high→low, `*` promoted badge for limited-data teams | ✅ DONE (was already present, enhanced footnote) |
| Validation tab | Methodology note explains holdout split, walk-forward, and promoted team caveat | ✅ DONE |
| Hero section | Model descriptions now say "Holdout accuracy" instead of generic descriptions | ✅ DONE |
| Methodology footnote | Added to overview tab and validation tab explaining three datasets | ✅ DONE |
| Teams table footnote | Enhanced to explain what "Games" and "Seasons" columns mean | ✅ DONE |

#### 5. Footer — DONE

- ✅ Footer already shows '57.8%' (fixed in prior commit)

#### 6. COVID Season Audit (2026-03-15) — DATA INCLUDES COVID

**Finding: the "no COVID" claim is FALSE.** COVID season data was never excluded at the data level. The label in `retrain_model.py` line 618 and CLAUDE.md header is aspirational, not enforced.

| File | Total Rows | 1920 (COVID) Rows | 2021 (partial COVID) Rows | Status |
|---|---|---|---|---|
| `data/hist_matches.csv` | 4,180 | 380 | 380 | ⚠️ PRESENT |
| `data/hist_features.csv` | 4,166 | 380 | 379 | ⚠️ PRESENT |
| `data/pi_ratings.csv` | 4,180 | 380 | 380 | ⚠️ PRESENT |
| `data/validation.json` test set (834) | 834 | 0 | 0 | ✅ CLEAN |
| xG coverage for 1920 | — | 380/380 have xG | 380/380 | ⚠️ PRESENT |

**Details:**
- Data actually has **11 seasons** (1415–2425), not 8. The "8 seasons" count is also wrong.
- Training set (first 3,332 rows) includes 759 COVID-era rows (380 from 1920 + 379 from 2021)
- Test set (last 834 rows, 2023-04-21 to 2025-05-25) has zero COVID rows — evaluation is clean
- `retrain_model.py` `build_features()` has no season exclusion filter — it processes all rows
- 2020/21 was also affected: matches played behind closed doors, home advantage significantly reduced

**Decision needed before next retrain:**
- **Option A:** Exclude 1920 from training (and optionally 2021). May improve accuracy since home advantage was distorted. Requires re-running Pi-ratings and rebuilding features.
- **Option B:** Keep data, fix all labels to say "11 seasons (2014–2025, includes COVID)" — honest but may hurt accuracy
- **Option C:** Keep data, add a `covid_flag` feature (1 for 1920/2021, 0 otherwise) so the model can learn the COVID effect rather than being confused by it
- ✅ Added tooltip on hover explaining: "Accuracy on a 20% chronological holdout test set (834 matches) from 8 EPL seasons (2017–2025)"

---

## COMMUNITY & MARKETING RULES

### Reddit Posting Rules — lessons learned 15 March 2026

**Subreddit status:**
- ❌ **r/sportsanalytics** — BANNED, removed by mods. Wrong sub for project/self-promotion posts.
- ⏳ **r/PremierLeague** — requires mod approval, too slow. Not worth the wait.
- ✅ **r/datascience** — ML projects welcome, large audience
- ✅ **r/learnmachinelearning** — supportive community, loves validation methodology
- ✅ **r/footballanalytics** — niche but perfect audience (check rules first)
- ✅ **r/soccer** — huge audience, keep post casual not technical
- ✅ **r/sportsbook** — betting/analytics crowd, good fit for value betting feature

**ALWAYS read subreddit rules before posting** — check if self-promotion is allowed.

### Post Checklist (before submitting)

1. Read pinned posts and sidebar rules
2. Check if self-promotion / project posts are allowed
3. Check minimum karma requirements
4. Have screenshots ready — visual posts get more traction
5. Reply in your own voice — AI-sounding replies hurt credibility
6. Don't post same content to multiple subs on the same day

### WHAT WORKED

- Honest title with real numbers — "57.8% accuracy on 834 matches"
- Admitting limitations upfront — 0% draw recall
- Responding quickly to technical criticism
- Short human replies beat long AI-drafted ones
- 2.4k views and 22 comments before removal — content resonates
