# Changelog

All model and data changes are recorded here.

---

## [Current] — 2026-03-14

### Pre-Match Model (xgb_champion)
- **Accuracy:** 55.6% (validation) / 48.1% (displayed in app)
- **Algorithm:** XGBoost + Optuna hyperparameter tuning
- **Training data:** 8 EPL seasons, 2017–2025, COVID seasons excluded
- **Test set:** 20% holdout (chronological split)
- **Features:** last-5 form (pts, GF, GA, GD, wins, draws), home/away-specific form, league position, league points, GD, ELO ratings, rolling shots and SOT averages, position diff, matchday
- **Notes:** Platt scaling applied for confidence calibration. p-value 5×10⁻²⁶ vs random baseline.

### Halftime Model (xgb_halftime)
- **Accuracy:** 62.4% (git log) / 55.3% (displayed in app)
- **Algorithm:** XGBoost
- **Features:** HT score, HT goal difference, HT result flags, form pts, form GD, home/away form, position diff, league pts diff, ELO
- **Notes:** 43.4% of EPL games flip from HT result to FT result — model accounts for this.

### Data
- `hist_matches.csv` — 11 seasons of EPL results (2014–2025)
- `hist_features.csv` — engineered features aligned to hist_matches.csv
- `validation.json` — held-out test set metrics

---

_Add a new entry above this line whenever models or data are retrained/updated._
