from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import pickle
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

import threading, time, os
_startup_time = time.time()

# ── TTL cache config ──────────────────────────────────────────────────────────
_CACHE_TTL = 30 * 60  # 30 minutes

# Standings cache — refreshed lazily on prediction requests when stale
_standings_lock = threading.Lock()
_standings_ts   = 0.0   # set to 0 so first call always refreshes

def _refresh_standings():
    """Re-fetch standings if the cache is older than _CACHE_TTL. Thread-safe."""
    global standings_cache, _standings_ts
    if time.time() - _standings_ts < _CACHE_TTL:
        return
    with _standings_lock:
        if time.time() - _standings_ts < _CACHE_TTL:   # double-checked locking
            return
        try:
            standings_cache = fetch_standings()
            _standings_ts   = time.time()
            print(f"[cache] standings refreshed ({len(standings_cache)} teams)")
        except Exception as e:
            print(f"[cache] standings refresh failed: {e}")

# Fixtures cache — avoid hitting the API on every /api/predict_fixtures call
_fixtures_cache = None
_fixtures_ts    = 0.0
_fixtures_lock  = threading.Lock()

def _get_cached_fixtures():
    """Return cached upcoming fixtures, re-fetching if stale."""
    global _fixtures_cache, _fixtures_ts
    if _fixtures_cache is not None and time.time() - _fixtures_ts < _CACHE_TTL:
        return _fixtures_cache
    with _fixtures_lock:
        if _fixtures_cache is not None and time.time() - _fixtures_ts < _CACHE_TTL:
            return _fixtures_cache
        _fixtures_cache = fetch_upcoming()
        _fixtures_ts    = time.time()
        print(f"[cache] fixtures refreshed ({len(_fixtures_cache)} upcoming)")
    return _fixtures_cache
from live_data import fetch_current_season, fetch_standings, fetch_upcoming, CREST_MAP
from injury_data import load_injuries, get_injury_count

# Load live season data
try:
    live_season = fetch_current_season()
    standings_cache = fetch_standings()
    _standings_ts   = time.time()   # mark cache as fresh
    print(f"✅ Live data loaded: {len(live_season)} matches")
except Exception as e:
    print(f"⚠️ Live data failed: {e}")
    live_season = None
    standings_cache = {}

# Load injury data (requires APIFOOTBALL_KEY env var)
injury_cache = load_injuries()
if injury_cache:
    print(f"✅ Injury data loaded: {sum(injury_cache.values())} injuries across {len(injury_cache)} teams")
else:
    print("⚠️ Injury data unavailable (set APIFOOTBALL_KEY to enable)")

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/content/epl_dashboard"

# Load champion model (8 seasons, no COVID, XGBoost + Optuna, 36 features incl. Pi-ratings)
with open(os.path.join(BASE, "models/xgb_champion.pkl"), "rb") as f:
    xgb_champion = pickle.load(f)

with open(os.path.join(BASE, "models/cols_champion.pkl"), "rb") as f:
    CHAMPION_COLS = pickle.load(f)
    # Pi-rating features (pi_home, pi_away, pi_diff) are in feat_dict;
    # they'll be picked up by CHAMPION_COLS after model retrain updates cols_champion.pkl

with open(os.path.join(BASE, "models/xgb_halftime.pkl"), "rb") as f:
    xgb_halftime = pickle.load(f)

with open(os.path.join(BASE, "models/cols_halftime.pkl"), "rb") as f:
    HT_COLS = pickle.load(f)

with open(os.path.join(BASE, "models/label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# Load data
hist_df = pd.read_csv(os.path.join(BASE, "data/hist_matches.csv"), parse_dates=["date"])
hist_feat = pd.read_csv(os.path.join(BASE, "data/hist_features.csv"))

# Load validated accuracy from validation.json (source of truth from Colab retrain)
_val_path = os.path.join(BASE, "data/validation.json")
with open(_val_path) as _vf:
    _validation = json.load(_vf)
VALIDATED_ACCURACY = round(_validation["accuracy"] * 100, 1)
print(f"✅ Validated accuracy from validation.json: {VALIDATED_ACCURACY}%")

# Load Pi-ratings (team-level final ratings for live prediction)
pi_team_path = os.path.join(BASE, "data/pi_team_ratings.csv")
if os.path.exists(pi_team_path):
    _pi_df = pd.read_csv(pi_team_path)
    pi_team_ratings = {r["team"]: {"R_h": r["pi_r_h"], "R_a": r["pi_r_a"], "overall": r["pi_overall"]}
                       for _, r in _pi_df.iterrows()}
    print(f"✅ Pi-ratings loaded for {len(pi_team_ratings)} teams")
else:
    pi_team_ratings = {}
    print("⚠️ Pi-ratings not found — run scripts/calculate_pi_ratings.py")

# Build test set
hist_model = hist_feat[hist_feat["home_form_pts"]+hist_feat["away_form_pts"]>0].copy()

# Inject Pi-ratings and corners_diff if missing from hist_features.csv
# (model was retrained with these features; align hist_df rows to hist_model rows)
missing_cols = [c for c in CHAMPION_COLS if c not in hist_model.columns]
if missing_cols:
    hist_df_aligned = hist_df.iloc[-len(hist_model):].reset_index(drop=True)
    hist_model = hist_model.reset_index(drop=True)
    if "pi_home" in missing_cols:
        hist_model["pi_home"] = hist_df_aligned["home_team"].map(
            lambda t: pi_team_ratings.get(t, {}).get("R_h", 0.0))
    if "pi_away" in missing_cols:
        hist_model["pi_away"] = hist_df_aligned["away_team"].map(
            lambda t: pi_team_ratings.get(t, {}).get("R_a", 0.0))
    if "pi_diff" in missing_cols:
        hist_model["pi_diff"] = hist_model["pi_home"] - hist_model["pi_away"]
    for col in missing_cols:
        if col not in hist_model.columns:
            hist_model[col] = 0.0
    print(f"✅ Injected missing cols into test set: {missing_cols}")

split = int(len(hist_model)*0.8)
X_test = hist_model[CHAMPION_COLS].iloc[split:]
pred = xgb_champion.predict(X_test)
proba = xgb_champion.predict_proba(X_test)

# Align test_df with hist_df
test_df = hist_df.iloc[-len(X_test):].copy().reset_index(drop=True)
test_features = hist_model.iloc[split:].reset_index(drop=True)
test_df["predicted"] = le.inverse_transform(pred)
test_df["correct"] = test_df["predicted"] == test_df["result"]
test_df["confidence"] = (proba.max(axis=1) * 100).round(1)
test_df["home_position"] = test_features["home_position"].values
test_df["away_position"] = test_features["away_position"].values

# Build halftime model test set using the same split
ht_test_feats = hist_model.iloc[split:].copy().reset_index(drop=True)
ht_test_matches = hist_df.iloc[-len(X_test):].copy().reset_index(drop=True)
ht_test_feats["ht_home"] = ht_test_matches["ht_home"].values
ht_test_feats["ht_away"] = ht_test_matches["ht_away"].values
ht_test_feats["ht_gd"] = ht_test_feats["ht_home"] - ht_test_feats["ht_away"]
ht_test_feats["ht_result_H"] = (ht_test_feats["ht_gd"] > 0).astype(int)
ht_test_feats["ht_result_D"] = (ht_test_feats["ht_gd"] == 0).astype(int)
ht_test_feats["ht_result_A"] = (ht_test_feats["ht_gd"] < 0).astype(int)
ht_valid = ht_test_feats.dropna(subset=["ht_home", "ht_away"])
ht_pred = xgb_halftime.predict(ht_valid[HT_COLS])
ht_predicted = le.inverse_transform(ht_pred)
ht_accuracy = round((ht_predicted == ht_test_matches.loc[ht_valid.index, "result"].values).mean() * 100, 1)
print(f"✅ Halftime model test accuracy: {ht_accuracy}%")

# Use most recent season for live predictions - append live API data
import pandas as pd
if live_season is not None:
    try:
        live_df = pd.concat([hist_df, live_season], ignore_index=True).drop_duplicates(
            subset=["date","home_team","away_team"]
        ).sort_values("date").reset_index(drop=True)
        print(f"✅ live_df: {len(live_df)} total matches ({len(hist_df)} historical + {len(live_season)} live)")
    except Exception as e:
        print(f"⚠️ Failed to merge live data: {e}")
        live_df = hist_df.copy()
else:
    live_df = hist_df.copy()
    print("⚠️ Using historical data only")
ALL_TEAMS = sorted(hist_df["home_team"].unique().tolist())

def conf_band(conf):
    if conf < 40:  return "<40%"
    if conf < 50:  return "40-50%"
    if conf < 60:  return "50-60%"
    return ">60%"

def get_form(team, n=5, home_only=False, away_only=False):
    tm = live_df[((live_df["home_team"]==team)|(live_df["away_team"]==team))]
    if home_only: tm = tm[tm["home_team"]==team]
    if away_only: tm = tm[tm["away_team"]==team]
    tm = tm.tail(n)
    count = len(tm)
    if count == 0: return 0,0,0,0,0
    # Exponential decay: most recent = 1.0, oldest of n = 0.5
    decay = 0.5 ** (1.0 / (n - 1)) if n > 1 else 1.0
    pts,gf,ga,wins,draws = 0.0,0.0,0.0,0.0,0.0
    w_sum = 0.0
    for j, (_, r) in enumerate(tm.iterrows()):
        w = decay ** (count - 1 - j)  # j=0 oldest, j=count-1 newest (w=1.0)
        w_sum += w
        ih = r["home_team"]==team
        g_for  = r["home_goals"] if ih else r["away_goals"]
        g_agt  = r["away_goals"] if ih else r["home_goals"]
        gf += w * g_for
        ga += w * g_agt
        if (ih and r["result"]=="H") or (not ih and r["result"]=="A"):
            pts += w * 3; wins += w
        elif r["result"]=="D":
            pts += w; draws += w
    # Normalize so scale matches unweighted sum (0-15 for pts, etc.)
    scale = count / w_sum
    return pts*scale, gf*scale, ga*scale, wins*scale, draws*scale

def get_standing(team):
    s = standings_cache.get(team)
    if s:
        return s["position"], s["points"], s["gd"]
    return 10, 0, 0

def get_form_list(team, n=5):
    tm = live_df[((live_df["home_team"]==team)|(live_df["away_team"]==team))]
    tm = tm.dropna(subset=["result"]).tail(n)
    out = []
    for _, r in tm.iterrows():
        ih = r["home_team"] == team
        if (ih and r["result"] == "H") or (not ih and r["result"] == "A"):
            out.append("W")
        elif r["result"] == "D":
            out.append("D")
        else:
            out.append("L")
    return out  # oldest → newest

def get_rolling_shots(team, n=5):
    all_tm = live_df[((live_df["home_team"]==team)|(live_df["away_team"]==team))]
    # skip current-season rows where shot data is unavailable (zeros from API)
    tm = all_tm[all_tm["hs"] > 0].tail(n)
    if len(tm)==0: return 0,0,0,0
    sh,sha,sot,sota = 0,0,0,0
    for _,r in tm.iterrows():
        ih = r["home_team"]==team
        sh   += r["hs"] if ih else r["as_"] if "as_" in r else r.get("as",0)
        sha  += r["as_"] if "as_" in r else r.get("as",0) if ih else r["hs"]
        sot  += r["hst"] if ih else r["ast"]
        sota += r["ast"] if ih else r["hst"]
    n2=len(tm)
    return sh/n2, sha/n2, sot/n2, sota/n2

def get_pi_rating(team):
    """Return (R_h, R_a, overall) Pi-rating for a team. Defaults to 0.0."""
    r = pi_team_ratings.get(team)
    if r:
        return r["R_h"], r["R_a"], r["overall"]
    return 0.0, 0.0, 0.0

def get_days_rest(team):
    """Days since the team's most recent match in live_df."""
    tm = live_df[((live_df["home_team"]==team)|(live_df["away_team"]==team))]
    if len(tm) == 0:
        return 7
    last_date = tm.iloc[-1]["date"]
    return max((pd.Timestamp.now() - last_date).days, 0)

def get_momentum(team, n=5):
    """Form momentum: PPG in last n matches minus PPG in prior n matches.
    Positive = rising, negative = falling."""
    tm = live_df[((live_df["home_team"]==team)|(live_df["away_team"]==team))]
    if len(tm) < 2:
        return 0.0
    recent = tm.tail(n)
    older  = tm.iloc[max(0, len(tm)-2*n):max(0, len(tm)-n)]
    def _ppg(matches):
        if len(matches) == 0: return 0.0
        pts = 0
        for _, r in matches.iterrows():
            ih = r["home_team"] == team
            if (ih and r["result"]=="H") or (not ih and r["result"]=="A"): pts += 3
            elif r["result"]=="D": pts += 1
        return pts / len(matches)
    return _ppg(recent) - _ppg(older)

def get_h2h_record(home, away, n=10):
    """H2H record from last n meetings (either venue). Returns (home_wins, draws, away_wins)."""
    matches = live_df[
        ((live_df["home_team"]==home)&(live_df["away_team"]==away))|
        ((live_df["home_team"]==away)&(live_df["away_team"]==home))
    ].sort_values("date", ascending=False).head(n)
    if len(matches) == 0:
        return 0, 0, 0
    h_wins = int(((matches["home_team"]==home)&(matches["result"]=="H")).sum()+
                 ((matches["away_team"]==home)&(matches["result"]=="A")).sum())
    a_wins = int(((matches["home_team"]==away)&(matches["result"]=="H")).sum()+
                 ((matches["away_team"]==away)&(matches["result"]=="A")).sum())
    draws  = int((matches["result"]=="D").sum())
    return h_wins, draws, a_wins

@app.route("/health")
def health():
    return jsonify({
        "status":          "ok",
        "uptime_seconds":  int(time.time() - _startup_time),
        "live_data":       live_season is not None,
        "standings":       len(standings_cache) > 0,
        "model":           "xgb_champion",
    })

@app.route("/")
def index():
    return render_template("index.html", teams=ALL_TEAMS)

@app.route("/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")

CURRENT_SEASON_TEAMS = sorted([
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Leeds United", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nott'm Forest", "Sunderland", "Tottenham", "West Ham", "Wolves"
])

@app.route("/api/current_teams")
def api_current_teams():
    if standings_cache:
        names = sorted(standings_cache.keys())
    elif live_season is not None and len(live_season) > 0:
        names = sorted(set(live_season["home_team"].tolist() + live_season["away_team"].tolist()))
    else:
        names = CURRENT_SEASON_TEAMS
    return jsonify([{"name": t, "crest": CREST_MAP.get(t, "")} for t in names])

@app.route("/api/overview")
def api_overview():
    dist = test_df["result"].value_counts().to_dict()
    by_outcome = {}
    for o in ["H","D","A"]:
        sub = test_df[test_df["predicted"]==o]
        by_outcome[o] = {
            "total": len(sub),
            "correct": int(sub["correct"].sum()),
            "accuracy": round(sub["correct"].mean()*100,1) if len(sub) else 0
        }
    return jsonify({"total_matches":len(test_df),"pre_accuracy":VALIDATED_ACCURACY,
                    "ing_accuracy":ht_accuracy,"random_baseline":33.3,
                    "result_distribution":dist,"by_outcome":by_outcome})

@app.route("/api/teams")
def api_teams():
    rows = []
    for team in ALL_TEAMS:
        sub = test_df[(test_df["home_team"]==team)|(test_df["away_team"]==team)]
        if len(sub) >= 3:
            rows.append({
                "team": team,
                "games": len(sub),
                "pre_accuracy": round(sub["correct"].mean()*100,1),
                "ing_accuracy": round(sub["correct"].mean()*100,1),
                "correct": int(sub["correct"].sum())
            })
    rows.sort(key=lambda x: x["pre_accuracy"], reverse=True)
    return jsonify(rows)

@app.route("/api/confidence")
def api_confidence():
    bands = ["<40%","40-50%","50-60%",">60%"]
    result = []
    for band in bands:
        sub = test_df[test_df["confidence"].apply(conf_band)==band]
        result.append({
            "band": band,
            "total": len(sub),
            "correct": int(sub["correct"].sum()),
            "accuracy": round(sub["correct"].mean()*100,1) if len(sub) else 0
        })
    return jsonify(result)

@app.route("/api/head_to_head")
def api_h2h():
    home = request.args.get("home")
    away = request.args.get("away")
    if not home or not away: return jsonify([])
    matches = live_df[
        ((live_df["home_team"]==home)&(live_df["away_team"]==away))|
        ((live_df["home_team"]==away)&(live_df["away_team"]==home))
    ].sort_values("date",ascending=False).head(10)
    rows = []
    for _,r in matches.iterrows():
        rows.append({
            "date": r["date"].strftime("%d %b %Y"),
            "home": r["home_team"],
            "away": r["away_team"],
            "score": f"{int(r['home_goals'])} - {int(r['away_goals'])}",
            "result": r["result"]
        })
    h_wins = int(((matches["home_team"]==home)&(matches["result"]=="H")).sum()+
                 ((matches["away_team"]==home)&(matches["result"]=="A")).sum())
    a_wins = int(((matches["home_team"]==away)&(matches["result"]=="H")).sum()+
                 ((matches["away_team"]==away)&(matches["result"]=="A")).sum())
    draws  = int((matches["result"]=="D").sum())
    return jsonify({"matches":rows,"summary":{"home_wins":h_wins,"away_wins":a_wins,"draws":draws}})

@app.route("/api/predict")
def api_predict():
    home = request.args.get("home")
    away = request.args.get("away")
    if not home or not away: return jsonify({"error":"Select both teams"})
    if home == away: return jsonify({"error":"Teams must be different"})

    _refresh_standings()
    h_pts,h_gf,h_ga,h_wins,h_draws = get_form(home)
    a_pts,a_gf,a_ga,a_wins,a_draws = get_form(away)
    hh_pts,hh_gf,hh_ga,_,_         = get_form(home, home_only=True)
    aa_pts,aa_gf,aa_ga,_,_          = get_form(away, away_only=True)
    h_pos,h_lpts,h_lgd              = get_standing(home)
    a_pos,a_lpts,a_lgd              = get_standing(away)
    h_sh,h_sha,h_sot,h_sota         = get_rolling_shots(home)
    a_sh,a_sha,a_sot,a_sota         = get_rolling_shots(away)

    # ELO (use last known values from hist_df)
    elo_home = hist_df[hist_df["home_team"]==home]["elo_home"].iloc[-1] if len(hist_df[hist_df["home_team"]==home]) else 1500
    elo_away = hist_df[hist_df["away_team"]==away]["elo_away"].iloc[-1] if len(hist_df[hist_df["away_team"]==away]) else 1500
    elo_diff = elo_home - elo_away

    # Pi-ratings
    h_pi_rh, h_pi_ra, _ = get_pi_rating(home)
    a_pi_rh, a_pi_ra, _ = get_pi_rating(away)

    # H2H record
    h2h_hw, h2h_d, h2h_aw = get_h2h_record(home, away)

    feat_dict = {
        "home_form_pts":h_pts,"home_form_gf":h_gf,"home_form_ga":h_ga,
        "home_form_gd":h_gf-h_ga,"home_form_wins":h_wins,"home_form_draws":h_draws,
        "away_form_pts":a_pts,"away_form_gf":a_gf,"away_form_ga":a_ga,
        "away_form_gd":a_gf-a_ga,"away_form_wins":a_wins,"away_form_draws":a_draws,
        "home_home_pts":hh_pts,"home_home_gd":hh_gf-hh_ga,
        "away_away_pts":aa_pts,"away_away_gd":aa_gf-aa_ga,
        "pts_diff":h_pts-a_pts,"gd_diff":(h_gf-h_ga)-(a_gf-a_ga),
        "home_position":h_pos,"away_position":a_pos,"position_diff":a_pos-h_pos,
        "home_league_pts":h_lpts,"away_league_pts":a_lpts,
        "league_pts_diff":h_lpts-a_lpts,
        "home_league_gd":h_lgd,"away_league_gd":a_lgd,
        "matchday":30,
        "elo_home":elo_home,"elo_away":elo_away,"elo_diff":elo_diff,
        "pi_home":h_pi_rh,"pi_away":a_pi_ra,"pi_diff":h_pi_rh-a_pi_ra,
        "home_days_rest":get_days_rest(home),"away_days_rest":get_days_rest(away),
        "home_momentum":get_momentum(home),"away_momentum":get_momentum(away),
        "h2h_home_wins":h2h_hw,"h2h_draws":h2h_d,"h2h_away_wins":h2h_aw,
        "home_injuries":get_injury_count(home),"away_injuries":get_injury_count(away),
        "home_shots_avg":h_sh,"home_shots_against_avg":h_sha,
        "home_sot_avg":h_sot,"home_sot_against_avg":h_sota,
        "away_shots_avg":a_sh,"away_shots_against_avg":a_sha,
        "away_sot_avg":a_sot,"away_sot_against_avg":a_sota,
        "shots_diff":h_sh-a_sh,"sot_diff":h_sot-a_sot,
        "corners_diff":0
    }
    feats = pd.DataFrame([feat_dict])[CHAMPION_COLS]
    proba  = xgb_champion.predict_proba(feats)[0]
    labels = le.classes_
    pred   = labels[np.argmax(proba)]
    conf   = round(float(proba.max()) * 100, 1)
    log.info("predict: %s vs %s → %s (%.1f%% confidence)", home, away, pred, conf)
    return jsonify({
        "home":home,"away":away,"prediction":pred,
        "probabilities":{l:round(float(p)*100,1) for l,p in zip(labels,proba)},
        "home_position":h_pos,"away_position":a_pos
    })

@app.route("/api/predict_halftime")
def api_predict_halftime():
    home = request.args.get("home")
    away = request.args.get("away")
    ht_home = int(request.args.get("ht_home", 0))
    ht_away = int(request.args.get("ht_away", 0))

    if not home or not away: return jsonify({"error":"Select both teams"})
    if home == away: return jsonify({"error":"Teams must be different"})

    h_pts,h_gf,h_ga,h_wins,h_draws = get_form(home)
    a_pts,a_gf,a_ga,a_wins,a_draws = get_form(away)
    hh_pts,hh_gf,hh_ga,_,_         = get_form(home, home_only=True)
    aa_pts,aa_gf,aa_ga,_,_          = get_form(away, away_only=True)
    h_pos,h_lpts,h_lgd              = get_standing(home)
    a_pos,a_lpts,a_lgd              = get_standing(away)
    elo_home = hist_df[hist_df["home_team"]==home]["elo_home"].iloc[-1] if len(hist_df[hist_df["home_team"]==home]) else 1500
    elo_away = hist_df[hist_df["away_team"]==away]["elo_away"].iloc[-1] if len(hist_df[hist_df["away_team"]==away]) else 1500

    ht_gd = ht_home - ht_away

    feat_dict = {
        "home_form_pts":h_pts,"home_form_gd":h_gf-h_ga,
        "away_form_pts":a_pts,"away_form_gd":a_gf-a_ga,
        "home_home_pts":hh_pts,"away_away_pts":aa_pts,
        "pts_diff":h_pts-a_pts,"gd_diff":(h_gf-h_ga)-(a_gf-a_ga),
        "position_diff":a_pos-h_pos,"league_pts_diff":h_lpts-a_lpts,
        "elo_home":elo_home,"elo_away":elo_away,"elo_diff":elo_home-elo_away,
        "ht_home":ht_home,"ht_away":ht_away,"ht_gd":ht_gd,
        "ht_result_H":1 if ht_gd>0 else 0,
        "ht_result_D":1 if ht_gd==0 else 0,
        "ht_result_A":1 if ht_gd<0 else 0
    }
    feats = pd.DataFrame([feat_dict])[HT_COLS]
    proba = xgb_halftime.predict_proba(feats)[0]
    labels = le.classes_
    pred = labels[np.argmax(proba)]

    return jsonify({
        "home":home,"away":away,"prediction":pred,
        "ht_score": f"{ht_home} - {ht_away}",
        "probabilities":{l:round(float(p)*100,1) for l,p in zip(labels,proba)},
        "home_position":h_pos,"away_position":a_pos
    })

@app.route("/api/standings")
def api_standings():
    if not standings_cache:
        return jsonify({"error": "Standings unavailable — live data failed to load"})
    table = sorted(
        [{"team": name, **data} for name, data in standings_cache.items()],
        key=lambda x: x["position"]
    )
    return jsonify(table)

@app.route("/api/validation")
def api_validation():
    with open(os.path.join(BASE, "data", "validation.json")) as f:
        return jsonify(json.load(f))


@app.route("/api/predict_fixtures")
def api_predict_fixtures():
    _refresh_standings()
    try:
        fixtures = _get_cached_fixtures()
    except Exception as e:
        return jsonify({"error": str(e)})

    results = []
    for fix in fixtures:
        home = fix["home_team"]
        away = fix["away_team"]
        try:
            h_pts,h_gf,h_ga,h_wins,h_draws = get_form(home)
            a_pts,a_gf,a_ga,a_wins,a_draws = get_form(away)
            hh_pts,hh_gf,hh_ga,_,_         = get_form(home, home_only=True)
            aa_pts,aa_gf,aa_ga,_,_          = get_form(away, away_only=True)
            h_pos,h_lpts,h_lgd              = get_standing(home)
            a_pos,a_lpts,a_lgd              = get_standing(away)
            h_sh,h_sha,h_sot,h_sota         = get_rolling_shots(home)
            a_sh,a_sha,a_sot,a_sota         = get_rolling_shots(away)

            hist_rows = hist_df[hist_df["home_team"]==home]
            elo_home = hist_rows["elo_home"].iloc[-1] if len(hist_rows) else 1500
            hist_rows = hist_df[hist_df["away_team"]==away]
            elo_away = hist_rows["elo_away"].iloc[-1] if len(hist_rows) else 1500
            elo_diff = elo_home - elo_away

            # Pi-ratings
            h_pi_rh, h_pi_ra, _ = get_pi_rating(home)
            a_pi_rh, a_pi_ra, _ = get_pi_rating(away)

            # H2H record
            h2h_hw, h2h_d, h2h_aw = get_h2h_record(home, away)

            feat_dict = {
                "home_form_pts":h_pts,"home_form_gf":h_gf,"home_form_ga":h_ga,
                "home_form_gd":h_gf-h_ga,"home_form_wins":h_wins,"home_form_draws":h_draws,
                "away_form_pts":a_pts,"away_form_gf":a_gf,"away_form_ga":a_ga,
                "away_form_gd":a_gf-a_ga,"away_form_wins":a_wins,"away_form_draws":a_draws,
                "home_home_pts":hh_pts,"home_home_gd":hh_gf-hh_ga,
                "away_away_pts":aa_pts,"away_away_gd":aa_gf-aa_ga,
                "pts_diff":h_pts-a_pts,"gd_diff":(h_gf-h_ga)-(a_gf-a_ga),
                "home_position":h_pos,"away_position":a_pos,"position_diff":a_pos-h_pos,
                "home_league_pts":h_lpts,"away_league_pts":a_lpts,
                "league_pts_diff":h_lpts-a_lpts,
                "home_league_gd":h_lgd,"away_league_gd":a_lgd,
                "matchday":fix["matchday"],
                "elo_home":elo_home,"elo_away":elo_away,"elo_diff":elo_diff,
                "pi_home":h_pi_rh,"pi_away":a_pi_ra,"pi_diff":h_pi_rh-a_pi_ra,
                "home_days_rest":get_days_rest(home),"away_days_rest":get_days_rest(away),
                "home_momentum":get_momentum(home),"away_momentum":get_momentum(away),
                "h2h_home_wins":h2h_hw,"h2h_draws":h2h_d,"h2h_away_wins":h2h_aw,
                "home_injuries":get_injury_count(home),"away_injuries":get_injury_count(away),
                "home_shots_avg":h_sh,"home_shots_against_avg":h_sha,
                "home_sot_avg":h_sot,"home_sot_against_avg":h_sota,
                "away_shots_avg":a_sh,"away_shots_against_avg":a_sha,
                "away_sot_avg":a_sot,"away_sot_against_avg":a_sota,
                "shots_diff":h_sh-a_sh,"sot_diff":h_sot-a_sot,
                "corners_diff":0
            }
            feats = pd.DataFrame([feat_dict])[CHAMPION_COLS]
            proba  = xgb_champion.predict_proba(feats)[0]
            labels = le.classes_
            pred   = labels[np.argmax(proba)]
            probas = {l:round(float(p)*100,1) for l,p in zip(labels,proba)}

            conf = round(max(probas.values()), 1)
            log.info("fixture: %s vs %s → %s (%.1f%% confidence)", home, away, pred, conf)
            results.append({
                "matchday":   fix["matchday"],
                "date":       fix["date"],
                "home":       home,
                "away":       away,
                "prediction": pred,
                "probabilities": probas,
                "confidence": conf,
                "home_position": h_pos,
                "away_position": a_pos,
                "home_crest": fix.get("home_crest", ""),
                "away_crest": fix.get("away_crest", ""),
                "home_form": get_form_list(home),
                "away_form": get_form_list(away),
            })
        except Exception as e:
            results.append({
                "matchday": fix["matchday"],
                "date":     fix["date"],
                "home":     home,
                "away":     away,
                "prediction": "N/A",
                "error": str(e)
            })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
