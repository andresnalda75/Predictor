# Predict Tab Redesign

## Problem

The current Predict tab has two dropdowns where users manually pick any two teams. This is artificial — the matchups aren't tied to real fixtures. Users can predict Man City vs Man City or pick teams that don't play each other for months. Worse, predictions made this way aren't automatically logged, so the Track Record tab stays empty.

## New Design

Replace the free-form team picker with **real upcoming EPL fixtures** from football-data.org (same data source as the Fixtures tab).

### User Flow

1. User opens Predict tab
2. Tab shows upcoming gameweek fixtures as cards (e.g. "GW 30 — Sat 22 Mar")
3. Each fixture card shows: home team, away team, kickoff time, venue
4. User clicks **Predict** button on a fixture card
5. App runs the model, logs the prediction to `predictions.db` automatically
6. Button state changes to show results: **Home 52% · Draw 24% · Away 24%** with a confidence badge
7. A **"Predict All Gameweek"** button at the top runs all upcoming fixtures at once and logs them all

### What Changes

| Component | Current | New |
|---|---|---|
| Predict tab UI | Two team dropdowns + Submit button | Fixture cards with individual Predict buttons + Predict All |
| Data source | User picks any two teams | Real fixtures from `/api/fixtures` |
| Prediction logging | Manual / not connected | Automatic — every prediction logged to `predictions.db` with correct match_date, home_team, away_team |
| Track Record tab | Empty until manual logging | Populated automatically as users make predictions |
| Old team picker | Primary UI | Retired or moved to advanced/debug section |

## Why This Matters

- **Predictions tied to reality** — only real scheduled matches, no artificial matchups
- **Automatic logging** — every prediction stored with correct match_date and teams, no manual step
- **Track Record becomes useful immediately** — users see their prediction history fill up week by week
- **Weekly engagement loop** — users return each gameweek to predict new fixtures
- **Reconciliation works seamlessly** — `reconcile_predictions.py` matches on (date, home_team, away_team), which are guaranteed correct when pulled from the fixture schedule

## Agent Responsibilities

### A1 — Frontend (Predict tab UI)
- Replace dropdown team pickers with fixture cards pulled from `/api/fixtures`
- Each card has a Predict button that calls `/api/predict` with fixture data
- On prediction response, update button to show probabilities + confidence badge
- Add "Predict All Gameweek" button that iterates all fixtures and logs each
- Group fixtures by gameweek/date
- Show already-predicted fixtures with their stored probabilities (check `predictions.db` via `/api/track_record`)
- Handle loading states, errors, already-kicked-off matches (disable Predict button)

### A2 — Backend (API changes)
- Update `/api/predict` to accept a `fixture_id` or `match_date` + `home_team` + `away_team` from the schedule (instead of free-form team selection)
- After running the model, automatically call the prediction logging logic (no separate `/api/log_prediction` call needed from frontend)
- Return prediction result including `prediction_id` from the database
- Add endpoint or param to check if a fixture has already been predicted (avoid duplicates)

### A4 — Prediction Logging (no changes needed)
- `predictions.db` schema already handles all required fields
- `/api/log_prediction` and `/api/track_record` work as-is
- `reconcile_predictions.py` matches on (date, home_team, away_team) — compatible with fixture-sourced predictions

### Old Team Picker
- Can be retired entirely or moved to an "Advanced" / "Debug" section for ad-hoc what-if predictions
- Not a priority — real fixture predictions are the primary workflow now

## Technical Notes

- Fixture data already available via `/api/fixtures` (uses football-data.org API, cached)
- Prediction logging schema in `predictions.db` already has all needed columns (match_date, home_team, away_team, model_version, etc.)
- Team names must match between fixture API response and prediction logging — use the same normalisation as `live_data.py`
- Gameweek grouping available from football-data.org API `matchday` field
- Already-predicted fixtures should show stored results, not re-run the model
