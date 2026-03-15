# EPL Predictor — Session Journal

---

## Session — March 15, 2026

### Champion Model Deployed — 59.11% accuracy

Biggest single accuracy jump in project history (+1.32pp).

| Metric | Previous | New | Delta |
|---|---|---|---|
| Holdout accuracy | 57.79% | 59.11% | +1.32pp |
| Walk-forward | 56.38% | 55.70% | -0.68pp |
| Draw recall | 0.0% | 3.66% | +3.66pp |
| Above 60% conf | 66.1% | 70.1% | +4pp |
| Close games | 52.6% | 55.2% | +2.6pp |
| Features | 47 | 26 | leaner |
| RPS | 0.1894 | 0.1895 | ~same |

### What Worked

- 300 Optuna trials found best params at trial 247 — 100 trials would have missed it
- xG features all survived RFE — confirmed useful signal
- Force-including xG + odds features in RFE was the right call

### Best Params

- `n_estimators`: 443
- `learning_rate`: 0.196
- `max_depth`: 7
- `subsample`: 0.766
- `colsample_bytree`: 0.920
- `min_child_weight`: 10
- `gamma`: 3.076
- `reg_alpha`: 4.888
- `reg_lambda`: 2.676

### What Didn't Work / Dropped by RFE

- `home_days_rest`, `away_days_rest` — dropped by RFE despite being new
- `home_momentum`, `away_momentum` — dropped by RFE
- Most shot statistics — xG replaced them
- Most form features — odds features dominate

### Infrastructure

- The Odds API integrated — 19–20 matches live, 476/500 requests remaining this month
- xG features now supplied for live predictions via `get_rolling_xg()` in `app.py`
- `validation.json` fully dynamic — all stats auto-update after retrain
- `/api/performance` endpoint live
- All hardcoded accuracy stats replaced with dynamic values
- Footer, Methodology tab, Performance tab all read from API

### Frontend Changes

- Nav reordered and renamed: Fixtures | Predict | Performance | Teams | Live | H2H | Table | Methodology
- Performance tab consolidated from 4 tabs (Stats, Accuracy, Confidence, About)
- Methodology tab created from About page
- Tab persistence on refresh via localStorage
- Fixtures pre-cached at startup
- Live odds badges on fixtures (⚡ icon, white text)
- Table tab: column order Team | Form | P | PTS | W | D | L | GF | GA | GD
- Form dots added to Table tab
- Sortable columns on Table tab
- Loading skeletons on all tabs

### Documentation

- All '8 seasons' references updated to '11 seasons (2014–2025)' across 6 files
- HISTORY.md left unchanged (historical record)
- Roadmap updated with new features: FIFA ratings, Transfermarkt, formation data, referee, match stakes, travel/fatigue, manager tenure, weather
- COVID audit completed — data includes COVID, exclusion hurts accuracy, keep all 11 seasons

### Odds API Usage

476/500 requests remaining this month.

### FIFA/EA FC Player Ratings Pipeline (late session)

- Built `scripts/fetch_fifa_ratings.py` — complete pipeline
- FC 25 real ratings sourced from Kaggle (nyagami dataset) — replaced FC 24 carry-forward
- All 20 current EPL teams confirmed including Leeds, Sunderland, Ipswich
- 9 new FIFA features in `hist_features.csv`: home/away att, def, mid, overall + `fifa_overall_diff`
- `hist_features.csv` now 71 columns, 4,166 rows, zero NaN
- Retrain completed in Colab — **57.31% holdout, did NOT beat 59.11% champion**

### FIFA Retrain Results (FAILED — do not deploy)

| Metric | Champion (59.11%) | FIFA retrain |
|---|---|---|
| Holdout accuracy | 59.11% | 57.31% (−1.80pp) |
| Features | 26 | 31 |
| Draw recall | 3.66% | 0.0% (regression) |
| Walk-forward | 55.70% | 56.23% |
| RPS | 0.1895 | 0.1905 |
| max_depth | 7 | 3 |

**Key insight:** Walk-forward CV Optuna found shallow trees (max_depth=3) — different feature sets need different hyperparameters. The champion's 300-trial direct holdout Optuna found max_depth=7 which works better. FIFA features survived RFE (6 of 9) but the overall model was worse. Squad quality is already captured by odds + ELO + Pi-ratings.

**Plan:** Try again in a combined retrain with FIFA + Transfermarkt + referee features, using 300-trial direct holdout Optuna (not walk-forward CV).

### Combined Retrain Results (FAILED — do not deploy)

- FIFA + Transfermarkt + match stakes + referee = 37 features
- 58.27% via direct 300-trial Optuna — still below 59.11% champion (−0.84pp)
- Optuna converges at trial 71, suggesting feature set ceiling
- Referee features dropped by RFE entirely
- New features survive RFE but add noise for Optuna
- 5-season training window deprioritised — ELO/Pi-ratings already encode recency

### New Pipelines Built (ready for future retrains)

| Script | Data | Features |
|---|---|---|
| `scripts/fetch_referees.py` | Referee stats, 51 unique refs | 3 features |
| `scripts/fetch_transfermarkt.py` | Squad market values, 20 teams × 11 seasons | 3 features |
| `scripts/fetch_fifa_ratings.py` | FC 25 real ratings from Kaggle | 9 features |
| Match stakes (derived) | Points gap to relegation/CL/title | 7 features |

### Documentation Created

- `docs/MARKETING_STRATEGY.md` — full channel playbook (Reddit, Substack, Twitter, podcasts)
- `docs/ARCHITECTURE_ROADMAP.md` — technical roadmap (GitHub Action, API tier, multi-league)
- `docs/MODEL_RESEARCH.md` — research-backed feature pipeline and what won't work

### Next Session Priorities

1. GitHub Action — weekly auto-retrain, +0.32%/week compound
2. A4 value betting — `ODDS_API_KEY` ready
3. Dixon-Coles P(draw) hybrid feature — est. +10–15% draw recall
4. Custom draw threshold — 5 lines of code
5. Try time-weighted training — weight recent seasons more
6. Teams table mobile column fix
