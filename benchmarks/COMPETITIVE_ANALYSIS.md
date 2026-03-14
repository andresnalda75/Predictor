# Competitive Analysis — EPL Predictor
_Last updated: March 2026_

---

## Competitor Profiles

### 1. Forebet — forebet.com
**Traffic:** ~10.5M/month | **Founded:** 2009 | **App:** iOS + Android (3.96★ Google Play, 9,952 reviews)

**Accuracy claims**
No official accuracy figures published on site. Third-party reviews attribute 75–83% to them; independent analysts estimate actual performance at 65–70%. No Brier scores, no calibration curves, no season-by-season breakdown. ValueTheMarkets explicitly notes the "absence of hit rates across seasons."

**Pricing**
Core site is free, ad-supported. A separate VIP service (likely third-party using the brand) charges €99–€250/month with a 3-month minimum commitment.

**Coverage**
850–1,200+ leagues claimed — the widest of any competitor.

**User complaints (Trustpilot 3.2/5, Reddit, reviews)**
- "Very poor performance" — only Trustpilot review (1-star, Oct 2025)
- App has no dark mode; scrolling in Trends section is broken
- Many international club games missing from app
- Interface overwhelming for casual users
- Forebet has a dedicated "Avoid Scam" page — multiple copycat sites exploit the brand

**Key weaknesses**
- Zero accountability: no published historical track record including losses
- Methodology entirely opaque — inputs, weights, and model type undisclosed
- Mobile app is a stripped-down version of the desktop site
- No community, no discussion, no expert layer

---

### 2. WinDrawWin — windrawwin.com
**Traffic:** ~2.2M/month | **Founded:** 2003 | **App:** None

**Accuracy claims**
No official figure. Third-party promotional reviews estimate 70–80%, unverified. The site itself says tips are "based on analysis, statistics and probabilities, and are never guaranteed."

**Pricing**
Core site is free. A separate windrawwin.win domain (possibly an impersonator) charges €35/week or €105/month.

**Coverage**
140+ leagues.

**User complaints (Trustpilot 2.6/5, soccertipsters.net)**
- Blacklisted by soccertipsters.net: "All are just fake winning rates" / "results are manipulated"
- Trustpilot pattern is suspicious: 5 five-star reviews on two days in April 2023, then one legitimate 1-star in November 2025
- No hotline support
- No injury or lineup data incorporated

**Key weaknesses**
- Lowest trust score of the four; categorised as a scam by tipster aggregators
- No independent accuracy verification
- Partnered with Predictz for distribution — not a standalone reliable source
- No mobile app in 2026 despite being 23 years old
- Static tips with no real-time variable adjustment

---

### 3. Predictz — predictz.com
**Traffic:** ~3.5M/month | **Founded:** 2009 | **App:** None

**Accuracy claims**
None disclosed. The only Trustpilot reviewer (2-star, Sep 2025) wrote: _"Stats are alright but their 'predictions' are shocking. I can fully understand why there's no record of what's come off."_

**Pricing**
Fully free. Revenue via bookmaker affiliate links.

**Coverage**
100+ leagues; stats for 3,000+ teams. Officially licensed to republish WinDrawWin tips.

**User complaints (Trustpilot 3.3/5)**
- Over-2.5 goals and BTTS predictions frequently fail
- Correct score predictions described as unreliable
- No explanation of how predictions are generated
- UI is a data table dump with no insight layer

**Key weaknesses**
- No track record published — reviewer explicitly calls this out
- Predictions depend partly on WinDrawWin tips, inheriting their trust issues
- Covers more leagues than it can model accurately
- No differentiation from raw stats aggregators

---

### 4. SoccerVista — soccervista.com
**Traffic:** Unknown | **Founded:** 1999 (26+ years old) | **App:** Yes (APKPure)

**Accuracy claims**
The site's own FAQ states: _"predictions aren't designed to be particularly accurate because they are not betting tips but a projection or educated guess."_ This is the most damaging self-own in the category. Third-party affiliate sites attribute 70–98% accuracy; the 98% figure is not credible.

**Pricing**
Free. A VIP SMS subscription exists but pricing is not publicly disclosed.

**Coverage**
Major European leagues only (PL, La Liga, Bundesliga, Serie A, Ligue 1, UCL). The site acknowledges: _"Our football predictions won't necessarily cover all leagues."_

**User complaints (soccertipsters.net — blacklisted, "VERY LOW")**
- "You cheaters! soccervista.com is only money grabbing" (2018)
- "Complete spammers site. Avoid this site!" (2017)
- Note: some complaints may confuse soccervista.com with the scam impersonator soccervista.online (flagged as dangerous by Gridinsoft)

**Key weaknesses**
- Self-disclaims accuracy — undermines the entire value proposition
- No retrospective tracking: FAQ says they "display predictions for future games only"
- Design and UX reflects its 1999 origins — not competitive on mobile
- Narrow coverage compared to Forebet
- Brand is being actively exploited by a scam .online domain

---

## Category-Level Frustrations

These apply across all four competitors — structural problems with the prediction site category:

| Problem | Detail |
|---|---|
| **No honest track record** | Sites show wins and bury losses. Researchers investigating betting sites found "99% fraudulent, with 95% caught editing results." |
| **Unverifiable accuracy** | Claims of 70–98% vs. independent estimates of 52–58% for free mathematical models |
| **Stats ≠ predictions** | Common complaint: "Stats are alright but predictions are shocking" — the recommendation layer adds no value over raw data |
| **No real-time variables** | Injuries, lineup reveals, managerial changes, weather — none systematically incorporated |
| **Outdated UX** | Ad-heavy, cluttered, mobile-hostile interfaces dating to the early 2000s |
| **No community layer** | No tipster ratings, no user discussion, no way to see expert disagreement |
| **Scam ecosystem** | Every major brand has at least one impersonator site. This destroys category-level trust. |

---

## Our Differentiators

### vs. Forebet
| Forebet | Us |
|---|---|
| 850+ leagues, thin model quality everywhere | EPL-only, deep model quality |
| No published accuracy | **55.6% validated on 532-match holdout, published methodology** |
| Black-box algorithm | XGBoost + Optuna, features documented in CLAUDE.md |
| 75–83% claimed (unverified) | 55.6% actual (honest — beats industry 54–56% benchmark) |
| Ad-heavy desktop site | Clean single-purpose UI |

**Positioning:** Forebet is wide and shallow. We are narrow and rigorous. For EPL fans who want to trust the numbers, we're the only option with a verifiable track record.

### vs. WinDrawWin
| WinDrawWin | Us |
|---|---|
| No app, no mobile (2026!) | Railway-deployed responsive web app |
| Blacklisted by tipster community | Open methodology, reproducible results |
| Static tips, no feature explanation | Confidence scores with calibration (Platt scaling) |
| 2.6/5 Trustpilot | Accountable — wins and losses both logged in benchmarks/results.json |

**Positioning:** WinDrawWin has a trust problem we don't have. Our entire benchmarking system (results.json, compare.py, walk-forward validation) exists to be the antithesis of "manipulated results."

### vs. Predictz
| Predictz | Us |
|---|---|
| "Predictions are shocking" (own reviewer) | 55.6% accuracy on unseen data, statistically significant (p < 10⁻²⁵) |
| Relies on WinDrawWin tips | Fully independent XGBoost model |
| No track record by design | Full prediction history tracked and published |
| Affiliate-link revenue model | No bookmaker affiliates |

**Positioning:** Predictz proves that data + bad predictions = user frustration. We close the gap between the stats layer and the recommendation layer with a validated model.

### vs. SoccerVista
| SoccerVista | Us |
|---|---|
| "Not designed to be accurate" (their own words) | Accuracy is the core product — published, backtested, benchmarked |
| 1999 UX, desktop-first | Modern responsive UI, form indicators, fixtures tab |
| Future-only predictions, no history | Historical accuracy visible in validation tab + results.json |
| Major leagues only | EPL-focused, but deep: 8 seasons of training data, current live form |

**Positioning:** SoccerVista built 26 years of traffic on brand inertia with a product that openly disclaims being accurate. Our product is the opposite: accuracy is the value proposition, and we prove it.

---

## Our Core Differentiator (Across All Four)

Every competitor shares one fatal flaw: **they ask users to trust them without giving users any reason to.** No published historical accuracy. No methodology. No calibration. No track record including losses.

We are building the opposite:
- **Validated accuracy**: 55.6% on a 532-match holdout (p < 10⁻²⁵ vs. 33.3% random baseline)
- **Honest calibration**: Platt-scaled confidence scores; high-confidence picks achieve 65.2%
- **Open methodology**: XGBoost + Optuna, 8 seasons, features documented
- **Logged track record**: benchmarks/results.json + walk-forward validation in compare.py
- **Transparent weaknesses**: draw recall is 0.79% — we say so publicly

The market is full of black boxes claiming 80%+ accuracy. We claim 55.6% and prove it. That's the differentiator.

---

## Sources
- ValueTheMarkets — Forebet Review
- Forebet, WinDrawWin, Predictz Trustpilot pages
- SoccerVista FAQ and Predictions page
- SoccerTipsters.net blacklist entries (WinDrawWin, SoccerVista)
- WinTips reviews (Forebet, WinDrawWin)
- BetMok — "Is Predictz Accurate?"
- GolSinyali — Most Accurate Soccer Prediction Sites
- BetTipsCompare — Reddit Guide 2025
- Scamadviser (forebet.com, windrawwin.com)
- Predictz/WinDrawWin partnership page
