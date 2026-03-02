from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import json

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/content/epl_dashboard"

# Load champion model (8 seasons, no COVID, XGBoost + Optuna)
with open(os.path.join(BASE, "models/xgb_champion.pkl"), "rb") as f:
    xgb_champion = pickle.load(f)

with open(os.path.join(BASE, "models/cols_champion.pkl"), "rb") as f:
    CHAMPION_COLS = pickle.load(f)

with open(os.path.join(BASE, "models/xgb_halftime.pkl"), "rb") as f:
    xgb_halftime = pickle.load(f)

with open(os.path.join(BASE, "models/cols_halftime.pkl"), "rb") as f:
    HT_COLS = pickle.load(f)

with open(os.path.join(BASE, "models/label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# Load data
hist_df = pd.read_csv(os.path.join(BASE, "data/hist_matches.csv"), parse_dates=["date"])
hist_feat = pd.read_csv(os.path.join(BASE, "data/hist_features.csv"))

# Build test set
hist_model = hist_feat[hist_feat["home_form_pts"]+hist_feat["away_form_pts"]>0].copy()
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

# Use most recent season for live predictions
live_df = hist_df.copy()
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
    pts,gf,ga,wins,draws = 0,0,0,0,0
    for _,r in tm.iterrows():
        ih = r["home_team"]==team
        gf += r["home_goals"] if ih else r["away_goals"]
        ga += r["away_goals"] if ih else r["home_goals"]
        if (ih and r["result"]=="H") or (not ih and r["result"]=="A"): pts+=3; wins+=1
        elif r["result"]=="D": pts+=1; draws+=1
    return pts,gf,ga,wins,draws

def get_standing(team):
    teams = {}
    for _,r in live_df.iterrows():
        for t,ih in [(r["home_team"],True),(r["away_team"],False)]:
            if t not in teams: teams[t]={"pts":0,"gf":0,"ga":0}
            teams[t]["gf"] += r["home_goals"] if ih else r["away_goals"]
            teams[t]["ga"] += r["away_goals"] if ih else r["home_goals"]
            if (ih and r["result"]=="H") or (not ih and r["result"]=="A"): teams[t]["pts"]+=3
            elif r["result"]=="D": teams[t]["pts"]+=1
    sorted_t = sorted(teams.items(), key=lambda x:(x[1]["pts"],x[1]["gf"]-x[1]["ga"]),reverse=True)
    for i,(t,s) in enumerate(sorted_t):
        if t==team: return i+1, s["pts"], s["gf"]-s["ga"]
    return 10,0,0

def get_rolling_shots(team, n=5):
    tm = live_df[((live_df["home_team"]==team)|(live_df["away_team"]==team))].tail(n)
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

@app.route("/")
def index():
    return render_template("index.html", teams=ALL_TEAMS)

@app.route("/api/overview")
def api_overview():
    acc = round(test_df["correct"].mean()*100,1)
    dist = test_df["result"].value_counts().to_dict()
    by_outcome = {}
    for o in ["H","D","A"]:
        sub = test_df[test_df["predicted"]==o]
        by_outcome[o] = {
            "total": len(sub),
            "correct": int(sub["correct"].sum()),
            "accuracy": round(sub["correct"].mean()*100,1) if len(sub) else 0
        }
    return jsonify({"total_matches":len(test_df),"pre_accuracy":acc,
                    "ing_accuracy":acc,"random_baseline":33.3,
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

    feat_dict = {
        "home_form_pts":h_pts,"home_form_gf":h_gf,"home_form_ga":h_ga,
        "home_form_gd":h_gf-h_ga,"home_form_wins":h_wins,"home_form_draws":h_draws,
        "away_form_pts":a_pts,"away_form_gf":a_gf,"away_form_ga":a_gf,
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

@app.route("/api/validation")
def api_validation():
    with open(os.path.join(BASE, "data", "validation.json")) as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
