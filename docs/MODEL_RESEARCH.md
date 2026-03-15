# Model Research — Path to 60%+

## Current State

59.11% holdout accuracy with 26 features (XGBoost + Optuna). Key signal comes from B365 implied odds, xG, Pi-ratings, and ELO. Draw recall at 3.66% — still the biggest weakness.

---

## Path to 60%+ Accuracy

### Tier A — Most Likely to Help (do next)

#### 1. Transfermarkt Squad Market Values

**Why:** Market value is a strong proxy for squad quality and depth. Academic literature consistently shows it as one of the best single predictors of match outcomes. Unlike form or standings, it captures off-season squad changes (signings, departures).

**Features:**
- `home_squad_value` / `away_squad_value` — total squad market value
- `squad_value_ratio` — home / away ratio
- `squad_value_diff` — absolute difference

**Data:** Free scrape from transfermarkt.com. Historical values available per season.

**Expected impact:** +0.5–1% accuracy. May overlap with FIFA ratings.

#### 2. Referee Features

**Why:** Some referees produce systematically more cards, fouls, and stoppages — which correlates with draws. Referee assignment is known before kickoff.

**Features:**
- `referee_avg_cards` — average cards per match for this referee
- `referee_draw_rate` — historical draw percentage in this referee's matches
- `referee_id` — categorical (let XGBoost learn patterns)

**Data:** Already in football-data.co.uk CSV. Zero effort to extract.

**Expected impact:** +0.3–0.5% accuracy, primarily through draw recall improvement.

#### 3. Match Stakes Features

**Why:** Late-season matches where one or both teams have nothing to play for produce more draws. A team mathematically safe from relegation and out of European contention plays differently.

**Features:**
- `home_gap_to_relegation` — points above 18th place
- `away_gap_to_relegation`
- `home_gap_to_cl` — points below 4th place
- `away_gap_to_cl`
- `both_safe` — binary flag: both teams >8 points above relegation
- `late_season` — binary flag: matchday > 30

**Data:** Derived from existing standings data. Zero external data needed.

**Expected impact:** +0.3–0.5% accuracy, mainly in late-season matches.

### Tier B — Draw-Specific Improvements

#### 4. Dixon-Coles P(draw) as Hybrid Feature

**Why:** Dixon-Coles is a Poisson-based model specifically designed for football. It models goal-scoring rates per team and derives draw probability from the joint distribution. XGBoost can't learn this structure natively.

**Approach:** Run Dixon-Coles alongside XGBoost. Use its `P(draw)` output as an additional input feature to XGBoost. Best of both worlds — Poisson structure for draws, gradient boosting for everything else.

**Expected impact:** Draw recall +10–15%, overall accuracy +0.5%.

**Complexity:** Medium — need to implement Dixon-Coles in Python, compute per-match P(draw) for all historical matches.

#### 5. Custom Draw Threshold

**Why:** Current model uses `argmax(H, D, A)`. Since P(draw) is almost never the highest probability, draws are never predicted. Lowering the threshold to predict draw when `P(draw) > 0.28` could recover some draw recall at minimal cost to H/A precision.

**Implementation:** 5 lines of code. Test immediately — no retrain needed.

**Expected impact:** Draw recall +5–10%. May slightly reduce H/A precision.

#### 6. Separate Draw Classifier

**Why:** Draws are fundamentally different from decisive results. A two-stage model could work better: (1) binary classifier: draw vs decisive, (2) if decisive, classify H vs A.

**Expected impact:** Draw recall +10–20%, but risk to overall accuracy if stage 1 is miscalibrated.

### Tier C — Advanced / Experimental

#### 7. Expected Possession Value (EPV)

**Why:** EPV measures the value of possessions based on location, type, and outcome. More sophisticated than xG — captures build-up play, not just shots.

**Data:** Requires event-level data (StatsBomb or similar). Expensive.

**Expected impact:** +1–2% if data is available. Unlikely to be free.

#### 8. LSTM Form Sequences

**Why:** Current form features are rolling averages. An LSTM could learn temporal patterns — e.g., "team that lost 3 in a row then won 2 is in a different state than team with 2W 1L 1W 1L."

**Approach:** Encode last 10 match results as a sequence, feed through LSTM, use output as features for XGBoost (hybrid architecture).

**Expected impact:** Unknown — could be +0.5–1% or nothing. Experimental.

#### 9. Ensemble Strategy

**Status:** CatBoost underperformed (52–53% WF), LightGBM cancelled.

**Revised approach:** Instead of algorithm diversity, try:
- Multiple XGBoost models with different random seeds → majority vote
- XGBoost + Dixon-Coles ensemble (one for H/A, one for draws)
- Stacking with logistic regression meta-learner

**Expected impact:** +0.3–0.5% from variance reduction.

---

## What Won't Work — Do Not Pursue

| Approach | Why It Won't Work |
|---|---|
| **Deep learning (neural nets, transformers)** | Not enough data. 4,166 training rows is tiny for deep learning. XGBoost is near-optimal for tabular data at this scale. Academic papers confirm GBMs beat NNs on structured football data. |
| **More historical data (pre-2014)** | Football has changed too much. Tactics, fitness, VAR, financial fair play — a 2010 match is a different sport. More data ≠ better signal when the distribution shifts. |
| **Player-level features without aggregation** | 22 players per match × hundreds of features = curse of dimensionality. Must aggregate to team level (which FIFA ratings already do). |
| **Sentiment analysis alone** | Too noisy. Fan sentiment reflects recent results (which we already capture with form features) plus transfer rumours and media narratives (which don't predict match outcomes). |
| **CatBoost or LightGBM replacing XGBoost** | Tested extensively. CatBoost peaked at 52–53% walk-forward with balanced weights. LightGBM cancelled. Algorithm swap is not the bottleneck — features are. |
| **SMOTE/oversampling for draws** | Synthetically generating draw examples risks teaching the model fake patterns. Better to add real draw-predictive features (referee, stakes, Dixon-Coles). |
| **COVID exclusion** | Tested: removing 1920 and/or 2021 seasons hurts accuracy by 1.2–1.5%. The model learns from all data, including anomalies. |

---

## Research Priority Order

1. **Transfermarkt values** — proven in literature, free data
2. **Referee features** — already in our CSV, trivial to add
3. **Match stakes** — derived from existing data, zero cost
4. **Custom draw threshold** — 5 lines of code
5. **Dixon-Coles hybrid** — medium effort, high potential for draws
6. **LSTM form sequences** — experimental, try after exhausting simpler options
7. **EPV** — only if StatsBomb data becomes accessible
