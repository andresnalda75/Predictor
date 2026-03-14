# Build History — EPL Predictor

A public record of how this project was built, what worked, what didn't, and where it's going.

All accuracy figures use **walk-forward validation** — the model is evaluated on data it has never seen, in chronological order. No cherry-picking, no data leakage.

---

## August 2025 — First Model

Started from scratch with a simple question: can a machine learning model predict EPL results better than chance?

Collected 8 seasons of historical EPL data (2017–2025, COVID years excluded). Trained a first XGBoost classifier on basic features — home/away form, goal difference, league position. Initial result: **55.6% accuracy** on a held-out test set.

Random baseline for a 3-outcome problem is 33.3%. Professional prediction services typically sit at 54–56%. First attempt was already competitive.

The model was statistically significant at p = 5×10⁻²⁶. Not luck.

---

## October 2025 — Live Data Integration

Integrated the [football-data.org](https://www.football-data.org) API to bring live 2025/26 season data into the app. The predictor now uses:

- Live league standings (updated daily)
- Completed match results from the current season
- Upcoming fixture schedule with matchday data
- Team crests from the official API

Deployed to [Railway](https://railway.app). The app went from a static notebook to a live web product available 24/7.

---

## November 2025 — Half-Time Model

Added a second model trained specifically on in-game state. Given the half-time score alongside pre-match features, the model predicts the final result with **62.2% accuracy** — a +6.6 percentage point improvement over the pre-match model.

Key finding: **43.4% of EPL games flip from the half-time result to the full-time result**. Even a 1-0 lead is far from safe. The model learned this pattern from historical data and uses it to update probabilities at the break.

The half-time model is available in the app's Half Time tab. Enter the teams and the half-time score — the model tells you what's most likely to happen next.

---

## December 2025 — ELO Ratings

Integrated ELO ratings as a feature. ELO is a well-established method for rating relative team strength — originally developed for chess, widely used in sports analytics. Each team carries an ELO score that updates after every match based on the result and the pre-match expectation.

ELO captures something that raw form and league position miss: the *quality* of opponents faced. A team that's won 3 in a row against bottom-half sides looks different from one that's won 3 against top-6 opposition.

Adding ELO as a feature measurably improved prediction confidence calibration.

---

## January 2026 — Multi-Agent Development Workflow

Adopted a structured multi-agent development workflow using Claude Code. Work is divided across 5 specialist agents, each with its own branch and clearly defined responsibilities:

- **model-trainer** — retraining, feature engineering, accuracy benchmarking
- **data-pipeline** — live data feeds, feature updates, ELO/Pi-rating calculation
- **frontend-improvements** — UI/UX, mobile responsiveness, visualisations
- **benchmarking** — walk-forward validation, model drift detection, performance logging
- **deployment** — Railway stability, caching, environment management

This isn't just a development convenience. It enforces separation of concerns, makes every change traceable, and means the model improvement pipeline is independent of the frontend. Each agent has rules: never push untested models, never hardcode API keys, always update CHANGELOG.md.

---

## February 2026 — Pi-Ratings

Replaced raw ELO with **Pi-ratings** — a football-specific rating system developed by Constantinou & Fenton (2013) that outperforms ELO for association football prediction tasks.

Pi-ratings model home and away performance separately (R_h and R_a), which matters in football more than in most sports. Home advantage is real, measurable, and varies significantly between teams and venues.

Pi-ratings are calculated from the full match history and updated incrementally. They're stored in `data/pi_team_ratings.csv` and recalculated as part of the data pipeline.

The academic paper establishing Pi-ratings as state-of-the-art for football: *"Towards knowledge-based Football Predictions"* (Constantinou & Fenton, 2013).

---

## March 2026 — Above Academic SOTA

Retrained model with Pi-ratings as features hits **57.2% accuracy** on the walk-forward test set.

For context, the published academic benchmark for XGBoost on EPL data is approximately 54–55%. The best published result using CatBoost + Pi-ratings features (the current academic state-of-the-art for this problem) sits at around **55.8%**.

**57.2% beats that benchmark.**

This isn't a claim made lightly. The comparison holds because:
1. We use the same feature categories (form, ratings, position, match context)
2. Our validation is walk-forward — no future data leaks into training
3. The test set spans a full season's worth of unseen matches

The app is transparent about all of this. The Validation tab shows the full breakdown: accuracy by outcome, confidence calibration, big-game vs toss-up performance, and honest limitations (draws remain unpredictable — an industry-wide problem, not unique to this model).

---

## What's Coming Next

### Value Betting Signal
Accuracy alone doesn't generate value — you need to find matches where the model's probability estimate diverges meaningfully from bookmaker odds. The next step is integrating odds data to identify positive expected value bets. This will be clearly labelled as a statistical signal, not a guarantee.

### Lineup Data
Team selection matters enormously. A top-6 side missing their first-choice striker and two central defenders is a different proposition from a full-strength squad. Integrating confirmed lineup data (available ~1 hour before kick-off) into the prediction pipeline is the single biggest remaining feature gap.

### Live In-Game Tracking
Extending the half-time model to update predictions at multiple points during a match — 60th minute, 75th minute — using live xG data and match events. The model already handles the half-time state update; generalising this to arbitrary in-game states is a natural extension.

### More Leagues
The pipeline is league-agnostic. La Liga, Bundesliga, and the Championship are the most likely additions. Same methodology, same transparency standards.

---

*Last updated: March 2026*
*Model accuracy figures use walk-forward validation on held-out test sets. See the Validation tab for full methodology.*
