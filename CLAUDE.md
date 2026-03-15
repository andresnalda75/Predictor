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
| Pre-match (deployed) | 57.79% | 47 | 834 | XGBoost + Optuna + Pi-ratings + B365 odds + exp. decay |
| Halftime in-game | 60.6% | 35 | 834 | Uses HT score + form + ELO + Pi-ratings |
| Random baseline | 33.3% | — | — | 3-outcome coin flip |
| Pro benchmark | 54–56% | — | — | Industry standard |

**Pre-match model details (retrained 2026-03-14):**
- RPS: 0.1894 (random ≈ 0.222)
- Draw recall: 0% (critical weakness, draws are 23% of outcomes)
- High confidence (>=50%): 66.1% accuracy on 487 matches
- Big game (8+ pos gap): 62.5% on 304 matches
- Close game (<4 pos gap): 52.6% on 270 matches
- Walk-forward accuracy: 56.4%
- Key features: B365 implied odds, Pi-ratings, ELO, form (exp. decay), league position, shots

---

## Model Improvement Roadmap

### TIER 1 — HIGH IMPACT, DO NEXT

| # | Improvement | Data Source | Est. Impact | Status |
|---|---|---|---|---|
| 1 | **Bookmaker odds as features** — football-data.co.uk CSV has B365H/B365D/B365A columns. Add implied probability features from odds. | football-data.co.uk (free CSV) | +2-3% accuracy | ✅ DONE — 57.07% → 57.79% (+0.72pp). 11 new features from B365 odds (implied probs, home edge, favourite, overround). **Known limitation:** trained on real B365 historical odds, but live predictions use ELO-derived implied probabilities as a proxy (`get_implied_odds()` in app.py). Live accuracy may be slightly below 57.79%. To fix: implement The Odds API (A4 — value betting) for real pre-match odds. |
| 2 | **xG from Understat** — free historical xG for EPL teams. Better than shots on target for measuring chance quality. | understat.com (free, scrape) | +1-2% accuracy | Not started |
| 3 | **Ensemble voting** — combine XGBoost + CatBoost + LightGBM predictions via meta-learner (stacking). | Internal | +1-2% accuracy | Deprioritised — CatBoost underperformed (52-53% WF), LightGBM cancelled |
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

1. Must beat **57.79%** on held-out test set (80/20 chronological split)
2. Must have walk-forward mean accuracy **> 55%**
3. Must not increase home bias (check H recall stays **below 85%**)
4. Run at least **3 Optuna trials**, take the best
5. Compare RPS score (lower is better, target **< 0.195**)
6. Check draw recall — any improvement over 1.57% is a bonus
7. Update `validation.json`, `results.json`, and `CHANGELOG.md` before pushing
8. **Odds feature caveat:** the 57.79% holdout was trained on real B365 odds, but live predictions use ELO-derived proxies. Live accuracy will be lower until The Odds API is integrated (A4). When comparing models, note whether odds features used real or proxy values.

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

### Agent 3 — `model-improvements`
**Branch:** `model-improvements`
**Goal:** Improve model accuracy beyond 57.79%
**Completed:**
- Retrained champion model: 55.6% → 57.07% → 57.79% (+2.19pp total)
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

---

## DATA AUDIT & DISCLOSURE PLAN

**Disclosure principle:** Every number shown to users must have a clear answer to: *'accuracy on what data, over how many seasons, for which teams?'* If we can't answer that clearly, we shouldn't show the number. This was flagged by the Reddit community and is critical for credibility before any marketing push.

### AUDIT TASKS

#### 1. Map every stat displayed in the app to its data source

| UI Element | Question | Status |
|---|---|---|
| Validation tab stats | Which dataset? Holdout? All 8 seasons? | ❓ Investigate |
| Team accuracy table | Which dataset? Why does Wolves show ~83 not 300+? | ❓ Investigate |
| Hero section accuracy numbers | Holdout or walk-forward? | ❓ Investigate |
| Halftime model stats | Same question — holdout or walk-forward? | ❓ Investigate |
| Form dots on fixtures | How many games lookback? | ❓ Investigate |
| ELO ratings | Calculated from which seasons? | ❓ Investigate |

#### 2. Investigate the 3-season discrepancy

- Footer says 'Model accuracy 57.1%' — is this the right number? (Should be 57.79%)
- Validation tab says 834 matches — confirm this is 20% holdout of 8 seasons
- Team accuracy table shows ~83 games per team — this suggests only ~2 seasons not 8, why?
- Leeds shows 7 games — should be ~29 current season + historical EPL games

#### 3. Check data completeness per team

- Which teams have full 8 seasons?
- Which teams were promoted/relegated and when?
- Leeds, Ipswich, Sunderland, Burnley, Sheffield United — how many seasons each?
- Are promoted team predictions less reliable due to limited data?

#### 4. Fix all UI disclosures

| Fix | Description | Status |
|---|---|---|
| Dataset source labels | Add dataset source label to every stat | ❌ TODO |
| Team accuracy table | Add "Seasons" column, sort high→low, add "promoted" badge for <3 seasons | ❌ TODO |
| Validation tab | Clarify "834 matches = holdout test set from 8 seasons of training data" | ❌ TODO |
| Hero section | Clarify what the accuracy number represents | ❌ TODO |
| Methodology footnote | Add a footnote explaining the three datasets (training, holdout, walk-forward) | ❌ TODO |

#### 5. Fix the footer

- Currently shows '57.1%' — should show '57.8%' (from validation.json VALIDATED_ACCURACY)
- Add tooltip explaining what the accuracy number means
