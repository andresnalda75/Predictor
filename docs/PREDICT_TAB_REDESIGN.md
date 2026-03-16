# Predictions Tab — Final Design

## Overview

A single **Predictions** tab replaces both the old Predict tab (free-form dropdowns) and the Track Record tab. It shows real upcoming fixtures as cards, lets users predict them with one click, and displays results after matches are played — all in one place.

**Nav order after redesign:** Fixtures | **Predictions** | Performance | Teams | Live | H2H | Table | Methodology

## Problem

The current Predict tab has two dropdowns where users manually pick any two teams. This is artificial — users can predict Man City vs Man City or pick teams that don't play each other for months. A separate Track Record tab exists but stays empty because predictions aren't automatically logged.

## User Flow

1. User opens **Predictions** tab
2. Tab shows upcoming gameweek fixtures as cards (e.g. "GW 30 — Sat 22 Mar")
3. Each fixture card shows: home team badge, away team badge, kickoff time, date
4. User clicks **Predict** on a fixture card → model runs, prediction logged to `predictions.db` automatically
5. Card transitions to **predicted/pending** state showing probabilities and confidence
6. A **"Predict All Gameweek"** button at the top runs all unpredicted fixtures at once
7. After matches are played, `reconcile_predictions.py` fills in actual results (daily GitHub Action)
8. Card transitions to **predicted/resolved** state showing prediction vs actual result (correct/incorrect)
9. Summary stats at the top: total predictions, accuracy %, streak

## Three Card States

Every fixture card exists in one of three states:

### 1. Unpredicted

Match is upcoming, no prediction logged for this fixture + current model version.

```
┌─────────────────────────────────────┐
│  Arsenal    vs    Chelsea           │
│  Sat 22 Mar · 15:00 · Emirates     │
│                                     │
│          [ Predict ]                │
└─────────────────────────────────────┘
```

- **Predict** button is primary CTA
- Clicking it calls `/api/predict` with match_date + home_team + away_team
- Backend runs model, logs to `predictions.db`, returns probabilities

### 2. Predicted / Pending

Prediction exists but match hasn't been played yet (actual_outcome IS NULL).

```
┌─────────────────────────────────────┐
│  Arsenal    vs    Chelsea           │
│  Sat 22 Mar · 15:00 · Emirates     │
│                                     │
│  H 52%  ·  D 24%  ·  A 24%        │
│  ◆ 52% confidence · v3.0 Sharp     │
│                                     │
│        [ Re-predict ]               │
└─────────────────────────────────────┘
```

- Probabilities displayed with the predicted outcome highlighted
- Confidence badge + model version tag
- **Re-predict** button: if current model version already predicted this match → does nothing (shows existing). If a newer model version is deployed → runs new model and logs fresh entry.

### 3. Predicted / Resolved

Match has been played, `reconcile_predictions.py` has filled in the actual result.

```
┌─────────────────────────────────────┐
│  Arsenal  2 - 1  Chelsea      ✅   │
│  Sat 22 Mar · Emirates              │
│                                     │
│  Predicted: H (52%) — Actual: H     │
│  ◆ v3.0 Sharp                       │
└─────────────────────────────────────┘
```

- Shows final score + actual outcome
- Green checkmark (✅) if correct, red cross (❌) if wrong
- No Predict/Re-predict button — match is settled
- Predicted outcome and actual outcome shown side by side

## Reconciliation — Daily GitHub Action

`scripts/reconcile_predictions.py` runs automatically via GitHub Actions on a daily schedule.

**Schedule:** Daily at 08:00 UTC (after overnight/late matches finish)

**GitHub Action workflow:** `.github/workflows/reconcile.yml`
```yaml
name: Reconcile Predictions
on:
  schedule:
    - cron: '0 8 * * *'    # Daily at 08:00 UTC
  workflow_dispatch:         # Manual trigger
```

**What it does:**
1. Finds predictions where `actual_outcome IS NULL` and `match_date < today`
2. Fetches finished EPL matches from football-data.org API
3. Matches by (date, home_team, away_team) and fills in `actual_outcome` + `correct` for ALL prediction rows (all model versions scored independently)
4. Requires `FOOTBALL_DATA_API_KEY` as a GitHub Actions secret

**Manual run:** `python scripts/reconcile_predictions.py` (requires `FOOTBALL_DATA_API_KEY` env var)

## What Changes

| Component | Current | New |
|---|---|---|
| Tab structure | Predict tab + Track Record tab (separate) | Single **Predictions** tab |
| Predict UI | Two team dropdowns + Submit button | Fixture cards with Predict buttons + Predict All |
| Track Record UI | Separate tab, empty until manual logging | Integrated into same tab — resolved cards show results inline |
| Data source | User picks any two teams | Real fixtures from `/api/fixtures` |
| Prediction logging | Manual / not connected | Automatic on every Predict click |
| Result reconciliation | Manual script run | Daily GitHub Action + manual fallback |
| Old team picker | Primary UI | Retired (or hidden in debug section) |

## Agent Responsibilities

### A1 — Frontend (Predictions tab UI)
- Create single Predictions tab replacing Predict + Track Record tabs
- Render fixture cards in three states: unpredicted, predicted/pending, predicted/resolved
- Each card's Predict button calls `/api/predict` with fixture data
- On response, transition card to predicted/pending state with probabilities + confidence badge
- "Predict All Gameweek" button iterates all unpredicted fixtures
- Group cards by gameweek/date
- On page load, fetch existing predictions via `/api/track_record` and merge with fixtures to determine card states
- Resolved cards show prediction vs actual with correct/incorrect indicator
- Summary stats bar at top: total predictions, resolved, correct, accuracy %
- Handle loading states, errors, already-kicked-off matches (disable Predict button)
- **Re-predict button:** calls `/api/predict` — backend returns existing prediction (same model) or runs fresh (new model). Frontend doesn't track model versions.
- Update nav order: Fixtures | **Predictions** | Performance | Teams | Live | H2H | Table | Methodology

### A2 — Backend (API changes)
- Update `/api/predict` to accept `match_date` + `home_team` + `away_team` from schedule
- After running the model, auto-log to `predictions.db` (no separate `/api/log_prediction` call from frontend)
- Return prediction result including `prediction_id` from database
- **Duplicate check:** unique on `match_date + home_team + away_team + model_version`. Same model returns existing; new model logs fresh entry.
- Add `/api/predictions_status` endpoint (or extend `/api/track_record`): given a list of fixture (date, home, away) tuples, return which have predictions and their state (pending/resolved)

### A4 — Prediction Logging (minor updates)
- `predictions.db` schema unchanged — already has all required fields
- `reconcile_predictions.py` must update ALL rows for a match (all model versions scored)
- `/api/track_record` returns all predictions; frontend handles display logic (latest per match by default)

### A5 — Deployment
- Add `.github/workflows/reconcile.yml` GitHub Action for daily reconciliation
- Add `FOOTBALL_DATA_API_KEY` as GitHub Actions repository secret
- Ensure `predictions.db` persists across Railway deployments (Railway persistent volume or external DB)

## Technical Notes

- Fixture data from `/api/fixtures` (football-data.org API, cached)
- Team name normalisation must match between fixture API and prediction logging — use `live_data.py` mapping
- Gameweek grouping from football-data.org `matchday` field
- **Uniqueness constraint:** `(match_date, home_team, away_team, model_version)` — UNIQUE index recommended
- **Reconciliation with multiple versions:** all rows matching (date, home_team, away_team) get updated with actual result
- **Persistence:** `predictions.db` on Railway needs a persistent volume — SQLite file will be lost on redeploy without one. Alternative: migrate to PostgreSQL (Railway provides it free).
