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
- **Re-predict button:** shown on fixture cards that already have a prediction. Clicking it calls `/api/predict` — backend decides whether to return existing prediction (same model version) or run a fresh one (new model version). Frontend doesn't need to know which model is current — backend handles it.

### A2 — Backend (API changes)
- Update `/api/predict` to accept a `fixture_id` or `match_date` + `home_team` + `away_team` from the schedule (instead of free-form team selection)
- After running the model, automatically call the prediction logging logic (no separate `/api/log_prediction` call needed from frontend)
- Return prediction result including `prediction_id` from the database
- **Duplicate check:** unique on `match_date + home_team + away_team + model_version` (NOT just match_date + teams). This means v3.0 Sharp logs once per match, but when v4.0 deploys it can log a fresh prediction for the same match alongside the v3.0 entry.
- Before inserting, check if (match_date, home_team, away_team, current model_version) already exists. If yes, return the existing prediction — no new log entry. If no (e.g. new model version deployed), run the model and log a fresh entry tagged to the new version.

### A4 — Prediction Logging (minor update)
- `predictions.db` schema already handles all required fields including `model_version`
- `/api/log_prediction` works as-is — multiple entries per match are allowed (one per model version)
- `reconcile_predictions.py` needs a minor update: when reconciling, update ALL prediction rows for a given (date, home_team, away_team) — not just one. Multiple model versions may have predicted the same match.
- `/api/track_record` display logic:
  - **Default view:** show the most recent prediction per match (latest `model_version`)
  - **Advanced view:** show full prediction history per match including all model versions, so users can compare how v3.0 vs v4.0 performed on the same fixture

### Old Team Picker
- Can be retired entirely or moved to an "Advanced" / "Debug" section for ad-hoc what-if predictions
- Not a priority — real fixture predictions are the primary workflow now

## Technical Notes

- Fixture data already available via `/api/fixtures` (uses football-data.org API, cached)
- Prediction logging schema in `predictions.db` already has all needed columns (match_date, home_team, away_team, model_version, etc.)
- Team names must match between fixture API response and prediction logging — use the same normalisation as `live_data.py`
- Gameweek grouping available from football-data.org API `matchday` field
- Already-predicted fixtures should show stored results, not re-run the model (unless a new model version is deployed)
- **Uniqueness constraint:** `(match_date, home_team, away_team, model_version)` — enforced at the application level before INSERT. One prediction per match per model version. A UNIQUE index on these four columns is recommended.
- **Reconciliation with multiple versions:** `reconcile_predictions.py` must update all rows matching (date, home_team, away_team) with the actual result, not just the first match. Each model version's prediction gets scored independently.
