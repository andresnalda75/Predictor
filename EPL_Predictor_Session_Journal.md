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

### Next Session Priorities

1. Push new champion to Railway (pending)
2. GitHub Action — weekly auto-retrain
3. A4 value betting — `ODDS_API_KEY` available
4. FIFA player ratings from fifaindex.com
5. Transfermarkt market values
6. Custom draw threshold — 5 lines of code
7. Referee data pipeline — football-data.co.uk
8. Teams table column order fix on mobile
9. Tab refresh fixes on web
