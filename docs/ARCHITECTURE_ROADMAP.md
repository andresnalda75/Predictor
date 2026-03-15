# Architecture Roadmap

## Current Stack

Flask + XGBoost + Gunicorn on Railway. Single `app.py` monolith with vanilla JS frontend. football-data.org API for live data, The Odds API for live odds, API-Football for injuries.

---

## Near Term (next 4 weeks)

### 1. GitHub Action — Weekly Auto-Retrain

**Priority:** HIGH — compound accuracy gain (+0.32%/week as season data accumulates)

**Design:**
- Scheduled GitHub Action runs every Monday at 06:00 UTC
- Pulls latest results from football-data.org API
- Appends to hist_matches.csv
- Rebuilds hist_features.csv
- Runs retrain_model.py (300 Optuna trials)
- Compares accuracy to current champion
- If better: commits new model, opens PR for review
- If worse: logs result to benchmarks/results.json, no deploy

**Requirements:**
- GitHub-hosted runner with XGBoost (Ubuntu, pip install)
- `FOOTBALL_DATA_API_KEY` as repository secret
- Colab is no longer needed once this works

### 2. sklearn Version Fix

**Issue:** Current `requirements.txt` has unpinned sklearn. Railway may install a different version than Colab used for training, causing pickle deserialization errors.

**Fix:** Pin `scikit-learn==1.4.2` (or whatever version Colab uses) in requirements.txt. Verify with `python -c "import sklearn; print(sklearn.__version__)"` in Colab before pinning.

### 3. PWA Improvements

**Already done:** manifest.json, service worker, icon, apple-mobile-web-app tags.

**Remaining:**
- Offline fallback page (show cached predictions when offline)
- Background sync for prediction updates
- Push notifications for matchday predictions (requires backend)

### 4. Custom Domain

- Register `eplpredictor.com` or `epl-predictor.co.uk`
- Point DNS to Railway deployment
- Update PWA manifest, meta tags, social sharing URLs
- Add SSL (Railway handles this automatically)

---

## Medium Term (1–3 months)

### 5. Email Digest

**Ties to:** Substack marketing strategy.

**If self-hosted:**
- Weekly cron job generates prediction summary
- SendGrid or Resend for delivery
- Unsubscribe handling
- Simple subscriber table in SQLite or Supabase

**If Substack:** No backend work needed — manually paste predictions weekly.

**Recommendation:** Start with Substack, migrate to self-hosted only if subscriber count justifies it (500+).

### 6. API Tier

Expose predictions as a REST API for third-party consumers.

**Endpoints:**
- `GET /api/v1/predictions` — upcoming match predictions
- `GET /api/v1/predictions/:match_id` — single match detail
- `GET /api/v1/track-record` — cumulative accuracy

**Auth:** API key per user, rate-limited (100 req/day free, paid tier for more).

**Monetisation:** Free tier drives awareness, paid tier generates revenue.

### 7. Live Scores Integration

- WebSocket or polling for live match scores during games
- Auto-trigger halftime model when HT scores arrive
- Show live probability updates as match progresses
- Source: football-data.org API (already have key) or free alternatives

### 8. Prediction Logging

**Critical for credibility.** Store every prediction with:
- Match details (teams, date, competition)
- Pre-match probabilities (H/D/A)
- Confidence level
- Actual result (filled in after match)
- Correct/incorrect flag

**Storage:** JSON file initially (predictions_log.json), migrate to SQLite if volume grows.

---

## Long Term (3–6 months)

### 9. User Accounts

- Simple auth (email + password or OAuth)
- Save favourite teams
- Personal prediction history
- Custom notification preferences
- Required for: email digest, API keys, personalisation

**Tech:** Flask-Login + SQLite or Supabase.

### 10. Multi-League Expansion

**Order of expansion:**
1. La Liga (next most data-rich)
2. Bundesliga
3. Serie A
4. Ligue 1
5. Championship (English second tier)

**Requirements per league:**
- Historical match data (football-data.co.uk has all major leagues)
- ELO ratings (recalculate per league)
- Pi-ratings (recalculate per league)
- Separate model per league (different playing styles)
- FIFA ratings already cover all leagues

**Architecture change:** Current code assumes EPL everywhere. Need to parameterise by league (season codes, team names, API endpoints).

---

## Technical Debt

| Item | Priority | Notes |
|---|---|---|
| sklearn version pin | HIGH | Pickle compatibility between Colab and Railway |
| Single app.py monolith | MEDIUM | Split into routes/, models/, features/ when >1500 lines |
| No database | LOW | JSON files work for now, migrate when user accounts arrive |
| No CI/CD tests | MEDIUM | Add pytest for feature engineering functions |
| Hardcoded team names | LOW | ALL_TEAMS list in app.py, fine for EPL-only |
