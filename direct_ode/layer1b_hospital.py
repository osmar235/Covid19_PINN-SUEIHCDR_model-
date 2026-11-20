#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 14: revised hospital pathway with literature constraints (PHASE14_PROTOCOL.md). No neural network,
no transmission model, no torch.

Structure H4 (per Phase 12 period p, all rates and fractions period specific):
  admissions A(t): fraction q direct to ICU, 1 - q to ward
  ward W     : mean stay L_w; exits ICU (p_wc), death (p_wd), discharge (rest)
  ICU C      : mean stay L_c; exits death (p_cd), survivors to post ICU ward P
  post ICU P : mean stay L_p; exits discharge only
    dW/dt = (1-q) A - W/L_w
    dC/dt = q A + p_wc W/L_w - C/L_c
    dP/dt = (1-p_cd) C/L_c - P/L_p
    hospital deaths/day = p_wd W/L_w + p_cd C/L_c
  observed ward census = W + P; observed ICU census = C; observed deaths = hospital deaths / h_p
Derived per admission quantities (constrained softly by the literature, section 2 of the protocol):
  P(ICU) = q + (1-q) p_wc;  P(death in hospital) = (1-q) p_wd + P(ICU) p_cd
  mean hospital stay = (1-q) L_w + P(ICU) L_c + P(ICU) (1-p_cd) L_p;  mean ICU stay = L_c;
  death among ICU patients = p_cd;  fraction of deaths in hospital = h_p
Models: H4 (with constraints, primary) and H4_noprior (sensitivity).

Usage (project root, .venv Python):  python layer1b_hospital.py --out phase14_layer1b
"""
import argparse, hashlib, json, sys, platform, datetime as dt, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--audit", default="phase12_epi_audit")
ap.add_argument("--hosp-file", default="covid19hospitalbycounty.csv")
ap.add_argument("--cdph-ts", default="data_epi/statewide-covid-19-cases-deaths-tests.csv")
ap.add_argument("--phase10", default="phase10_sandiego_N_stage2/stage2_regime_table.csv,"
                                    "phase10_LA_SF_N_stage2/stage2_regime_table.csv")
ap.add_argument("--phase13", default="phase13_layer1/projections.csv")
ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
ap.add_argument("--models", default="H4,H4_noprior")
ap.add_argument("--origins", default="Winter20,Delta,Omicron,BA5_Waning")
ap.add_argument("--adm-series", default="adm_per_day", choices=["adm_per_day", "adm_per_day_supp2"])
ap.add_argument("--starts", type=int, default=5)
ap.add_argument("--max-nfev", type=int, default=600)
ap.add_argument("--skip-projections", action="store_true")
ap.add_argument("--out", default="phase14_layer1b")
ap.add_argument("--overwrite", action="store_true")
A = ap.parse_args()

WINDOW = ("2020-08-02", "2022-12-31")
PERIODS = [("P1_FirstWave", "2020-03-29", "2020-09-30"), ("P2_Winter20", "2020-10-01", "2021-03-31"),
           ("P3_Delta", "2021-04-01", "2021-11-30"), ("P4_Omicron", "2021-12-01", "2022-04-30"),
           ("P5_BA2_BA5", "2022-05-01", "2022-12-31")]
ORIGIN_DATES = {"Winter20": "2021-01-15", "Delta": "2021-08-15", "Omicron": "2022-01-15", "BA5_Waning": "2022-06-15"}
CRIT = {"icu": 0.15, "ward": 0.15, "deaths": 0.25, "periods_needed": 4, "proj_icu": 0.20, "proj_ward": 0.20,
        "plaus_z": 2.0}
# Protocol section 2. Keys: quantity -> (value for P1/P2, value for P3, value for P4/P5, sd on log scale)
PRIORS = {"mean_hosp_stay": (8.0, 7.6, 5.5, 0.25), "p_icu": (0.182, 0.175, 0.130, 0.30),
          "p_death_hosp": (0.129, 0.123, 0.071, 0.30), "mean_icu_stay": (9.0, 8.1, 4.0, 0.40),
          "p_death_icu": (0.36, 0.35, 0.35, 0.30), "h": (0.70, 0.70, 0.59, 0.15)}
PRIOR_SET = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}           # period index -> column of PRIORS
STAY = (1.0, 60.0); H_BOUNDS = (0.5, 1.0); Q_MAX = 0.9; LOGIT = 8.0; NPP = 8  # raw params per period
CITY = {"San Diego": "SanDiego", "Los Angeles": "LosAngeles", "San Francisco": "SanFrancisco"}

R = Path(A.root); out = Path(A.out)
if out.exists() and any(out.iterdir()) and not A.overwrite:
    raise SystemExit(f"{out} is not empty; use a new --out or --overwrite")
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "run_log.txt", "a", encoding="utf-8")
INPUTS = set()


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


# ------------------------------------------------------------------ data (as Phase 13)
def load_county(county):
    days = pd.date_range(WINDOW[0], WINDOW[1], freq="D")
    p = R / A.audit / f"weekly_{CITY[county]}.csv"; INPUTS.add(str(p))
    wk = pd.read_csv(p, parse_dates=["week_start"]).set_index("week_start")[A.adm_series]
    weekly_idx = pd.date_range(pd.Timestamp(WINDOW[0]) - pd.Timedelta(days=7), WINDOW[1], freq="7D")
    w = wk.reindex(weekly_idx); filled = w.isna()
    w = np.expm1(np.log1p(w).interpolate(limit_direction="both"))
    adm = w.reindex(days, method="ffill"); adm_filled = filled.reindex(days, method="ffill").astype(bool)
    h = pd.read_csv(R / A.hosp_file); INPUTS.add(str(R / A.hosp_file))
    h = h[h["county"] == county].copy()
    h["date"] = pd.to_datetime(h["todays_date"], format="mixed")
    h = h.drop_duplicates("date", keep="last").set_index("date").sort_index()
    icu = h["icu_covid_confirmed_patients"].astype(float).reindex(days)
    ward = (h["hospitalized_covid_confirmed_patients"] - h["icu_covid_confirmed_patients"]).astype(float).reindex(days)
    t = pd.read_csv(R / A.cdph_ts, low_memory=False); INPUTS.add(str(R / A.cdph_ts))
    t = t[t["area"] == county].copy(); t["date"] = pd.to_datetime(t["date"], errors="coerce")
    deaths = pd.to_numeric(t.dropna(subset=["date"]).set_index("date")["deaths"], errors="coerce").reindex(days)
    df = pd.DataFrame({"adm": adm.values, "adm_filled": adm_filled.values, "ward": ward.values,
                       "icu": icu.values, "deaths": deaths.values}, index=days)
    pid = np.zeros(len(days), int)
    for i, (_, a, b) in enumerate(PERIODS):
        pid[(days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))] = i
    df["period"] = pid
    return df


# ------------------------------------------------------------------ model
def unpack_period(r):
    """raw: log L_w, logit p_wc, logit s_wd (share of non ICU ward exits that die), log L_c, logit p_cd,
    log L_p, logit q/Q_MAX, logit h01 (h = H_lo + (H_hi - H_lo) sigmoid)."""
    L_w = math.exp(r[0]); p_wc = sig(r[1]); p_wd = (1.0 - p_wc) * sig(r[2]); L_c = math.exp(r[3])
    p_cd = sig(r[4]); L_p = math.exp(r[5]); q = Q_MAX * sig(r[6])
    h = H_BOUNDS[0] + (H_BOUNDS[1] - H_BOUNDS[0]) * sig(r[7])
    return {"L_w": L_w, "p_wc": p_wc, "p_wd": p_wd, "L_c": L_c, "p_cd": p_cd, "L_p": L_p, "q": q, "h": h}


def derived(p):
    p_icu = p["q"] + (1 - p["q"]) * p["p_wc"]
    return {"mean_hosp_stay": (1 - p["q"]) * p["L_w"] + p_icu * p["L_c"] + p_icu * (1 - p["p_cd"]) * p["L_p"],
            "p_icu": p_icu, "p_death_hosp": (1 - p["q"]) * p["p_wd"] + p_icu * p["p_cd"],
            "mean_icu_stay": p["L_c"], "p_death_icu": p["p_cd"], "h": p["h"]}


def transition(p):
    """Exact one day map for z = [W, C, P, D_cum]: z1 = Phi z0 + Gam a."""
    M = np.zeros((5, 5)); kw, kc, kp = 1 / p["L_w"], 1 / p["L_c"], 1 / p["L_p"]
    M[0, 0] = -kw
    M[1, 0] = p["p_wc"] * kw; M[1, 1] = -kc
    M[2, 1] = (1 - p["p_cd"]) * kc; M[2, 2] = -kp
    M[3, 0] = p["p_wd"] * kw; M[3, 1] = p["p_cd"] * kc
    M[0, 4] = 1 - p["q"]; M[1, 4] = p["q"]
    E = expm(M)
    return E[:4, :4], E[:4, 4]


def steady_split(p):
    """Steady state W:C:P per unit admission, used to split observed census into W and P."""
    W = (1 - p["q"]) * p["L_w"]; C = (p["q"] + p["p_wc"] * W / p["L_w"]) * p["L_c"]
    P = (1 - p["p_cd"]) * C / p["L_c"] * p["L_p"]
    return W, C, P


def init_state(p, ward_obs, icu_obs):
    W, C, P = steady_split(p)
    fw = W / max(W + P, 1e-12)
    return np.array([fw * ward_obs, icu_obs, (1 - fw) * ward_obs, 0.0])


def simulate(theta, n_per, adm, pid, ward0, icu0):
    sets = [unpack_period(theta[j * NPP:(j + 1) * NPP]) for j in range(n_per)]
    maps = [transition(s) for s in sets]
    n = len(adm); Z = np.zeros((n + 1, 4)); Z[0] = init_state(sets[pid[0]], ward0, icu0)
    for d in range(n):
        Phi, Gam = maps[pid[d]]
        Z[d + 1] = Phi @ Z[d] + Gam * adm[d]
    hosp_deaths = np.diff(Z[:, 3])
    h_day = np.array([sets[pid[d]]["h"] for d in range(n)])
    return Z[:n, 0] + Z[:n, 2], Z[:n, 1], hosp_deaths / h_day, sets, Z[n]


def residuals(theta, n_per, df, pid_local, periods_present, use_prior):
    ward_m, icu_m, Dm, sets, _ = simulate(theta, n_per, df["adm"].values, pid_local,
                                          df["ward"].iloc[0], df["icu"].iloc[0])
    rw = np.log1p(np.clip(ward_m, 0, None)) - np.log1p(df["ward"].values)
    rc = np.log1p(np.clip(icu_m, 0, None)) - np.log1p(df["icu"].values)
    nwk = len(df) // 7
    dm = Dm[:nwk * 7].reshape(nwk, 7).sum(1); do = df["deaths"].values[:nwk * 7].reshape(nwk, 7).sum(1)
    rd = np.log1p(np.clip(dm, 0, None)) - np.log1p(do)
    parts = [rw, rc, rd]
    if use_prior:
        rp = []
        for j, pglob in enumerate(periods_present):
            dq = derived(sets[j]); col = PRIOR_SET[pglob]
            for k, v in PRIORS.items():
                rp.append((math.log(dq[k]) - math.log(v[col])) / v[3])
        parts.append(np.array(rp))
    r = np.concatenate(parts)
    return np.where(np.isfinite(r), r, 0.0)


def bounds(n_per):
    lo = [math.log(STAY[0]), -LOGIT, -LOGIT, math.log(STAY[0]), -LOGIT, math.log(STAY[0]), -LOGIT, -LOGIT]
    hi = [math.log(STAY[1]), LOGIT, LOGIT, math.log(STAY[1]), LOGIT, math.log(STAY[1]), LOGIT, LOGIT]
    return np.array(lo * n_per), np.array(hi * n_per)


def logit(x):
    return math.log(x / (1 - x))


def start_point(n_per, s, model):
    """Start 0 (fixed): ward stay 6 d, 12 % ward to ICU, 6 % of the other ward exits die, ICU stay 9 d,
    35 % ICU deaths, post ICU stay 5 d, 5 % direct ICU, h 0.7. Starts 1..: uniform in bounds, fixed seeds."""
    lo, hi = bounds(n_per)
    if s == 0:
        base = [math.log(6.0), logit(0.12), logit(0.06), math.log(9.0), logit(0.35), math.log(5.0),
                logit(0.05 / Q_MAX), logit((0.7 - H_BOUNDS[0]) / (H_BOUNDS[1] - H_BOUNDS[0]))]
        x = np.array(base * n_per)
    else:
        rng = np.random.default_rng(20261011 + 1000 * s + (4 if model == "H4" else 5))
        x = lo + (hi - lo) * rng.uniform(0.05, 0.95, size=len(lo))
    return np.clip(x, lo + 1e-9, hi - 1e-9)


def fit(df, model):
    present = sorted(set(df["period"].values)); remap = {p: i for i, p in enumerate(present)}
    pid_local = np.array([remap[p] for p in df["period"].values]); n_per = len(present)
    lo, hi = bounds(n_per); use_prior = (model == "H4")
    best, costs = None, []
    for s in range(A.starts):
        r = least_squares(residuals, start_point(n_per, s, model), bounds=(lo, hi), method="trf",
                          args=(n_per, df, pid_local, present, use_prior), max_nfev=A.max_nfev, x_scale="jac")
        costs.append(float(r.cost))
        if best is None or r.cost < best.cost:
            best = r
    return best, costs, n_per, present, pid_local


def relmae(m, o):
    ok = np.isfinite(o) & np.isfinite(m)
    return float(np.mean(np.abs(m[ok] - o[ok])) / max(np.mean(o[ok]), 1e-9)) if ok.any() else float("nan")


# ------------------------------------------------------------------ in sample
counties = [c.strip() for c in A.counties.split(",") if c.strip()]
models = [m.strip() for m in A.models.split(",") if m.strip()]
log(f"Phase 14 | models {models} | counties {counties} | adm {A.adm_series} | starts {A.starts} | max_nfev {A.max_nfev}")
data = {c: load_county(c) for c in counties}
met_rows, par_rows, proj_rows, crit = [], [], [], {}
for county in counties:
    df = data[county]
    log(f"\n{county}: {len(df)} days, admissions filled on {int(df['adm_filled'].sum())} days")
    for model in models:
        res, costs, n_per, present, pid_local = fit(df, model)
        ward_m, icu_m, Dm, sets, _ = simulate(res.x, n_per, df["adm"].values, pid_local,
                                              df["ward"].iloc[0], df["icu"].iloc[0])
        traj = df.copy(); traj["ward_model"] = ward_m; traj["icu_model"] = icu_m; traj["deaths_model"] = Dm
        traj.to_csv(out / f"fits_{CITY[county]}_{model}.csv", index_label="date")
        n_fit, n_pl, n_both = 0, 0, 0
        for j, p in enumerate(present):
            sub = traj[traj["period"] == p]
            wk = sub[["deaths", "deaths_model"]].resample("W-SAT").sum(min_count=7).dropna()
            m = {"county": county, "model": model, "period": PERIODS[p][0], "days": len(sub),
                 "relMAE_icu": relmae(sub["icu_model"].values, sub["icu"].values),
                 "relMAE_ward": relmae(sub["ward_model"].values, sub["ward"].values),
                 "relMAE_deaths_weekly": relmae(wk["deaths_model"].values, wk["deaths"].values)}
            m["fit_met"] = bool(m["relMAE_icu"] <= CRIT["icu"] and m["relMAE_ward"] <= CRIT["ward"]
                                and m["relMAE_deaths_weekly"] <= CRIT["deaths"])
            dq = derived(sets[j]); col = PRIOR_SET[p]; zmax = 0.0
            for k, v in PRIORS.items():
                z = (math.log(dq[k]) - math.log(v[col])) / v[3]
                m[f"{k}"] = dq[k]; m[f"{k}_prior"] = v[col]; m[f"{k}_z"] = z; zmax = max(zmax, abs(z))
            m["max_abs_z"] = zmax; m["plausible"] = bool(zmax <= CRIT["plaus_z"])
            m["met_both"] = bool(m["fit_met"] and m["plausible"])
            n_fit += m["fit_met"]; n_pl += m["plausible"]; n_both += m["met_both"]
            met_rows.append(m)
            par_rows.append({"county": county, "model": model, "period": PERIODS[p][0],
                             **{k: round(v, 5) for k, v in sets[j].items()}, "cost": float(res.cost),
                             "cost_starts": json.dumps([round(c, 4) for c in costs]), "nfev": int(res.nfev)})
        need = min(CRIT["periods_needed"], len(present))
        crit[f"{county}|{model}"] = {"periods": len(present), "fit_met": n_fit, "plausible": n_pl,
                                     "both": n_both, "pass_fit": bool(n_fit >= need),
                                     "pass_plausible": bool(n_pl >= need), "pass": bool(n_fit >= need and n_pl >= need)}
        log(f"  {model}: cost {res.cost:.3f} (starts {[round(c, 2) for c in costs]}, nfev {res.nfev}) | fit "
            f"{n_fit}/{len(present)} plausible {n_pl}/{len(present)}")

decision = {"H4_pass_all_counties": bool("H4" in models and all(crit[f"{c}|H4"]["pass"] for c in counties)),
            "rule": "H4 meets the in sample and plausibility criteria in all three counties (PHASE14 section 4)"}
log("\nDecision:", decision)

# ------------------------------------------------------------------ conditional projections
if not A.skip_projections:
    ctx = pd.concat([pd.read_csv(R / p) for p in A.phase10.split(",")])
    for p in A.phase10.split(","):
        INPUTS.add(str(R / p))
    p13 = pd.read_csv(R / A.phase13) if (R / A.phase13).exists() else pd.DataFrame()
    if not p13.empty:
        INPUTS.add(str(R / A.phase13))
    for county in counties:
        df = data[county]
        for oname in [o.strip() for o in A.origins.split(",") if o.strip()]:
            o = pd.Timestamp(ORIGIN_DATES[oname])
            tr = df[df.index <= o]; fu = df[(df.index > o) & (df.index <= o + pd.Timedelta(days=28))]
            for model in models:
                res, costs, n_per, present, pid_local = fit(tr, model)
                _, _, _, sets, zend = simulate(res.x, n_per, tr["adm"].values, pid_local,
                                               tr["ward"].iloc[0], tr["icu"].iloc[0])
                p_last = sets[-1]; Phi, Gam = transition(p_last)
                # Origin state: model W:P split, scaled to the observed census on the origin day
                wp = zend[0] + zend[2]; fw = zend[0] / wp if wp > 0 else 1.0
                z = np.array([fw * tr["ward"].iloc[-1], tr["icu"].iloc[-1], (1 - fw) * tr["ward"].iloc[-1], 0.0])
                Wp, Cp = [], []
                for a in fu["adm"].values:
                    z = Phi @ z + Gam * a; Wp.append(z[0] + z[2]); Cp.append(z[1])
                Wp, Cp = np.array(Wp), np.array(Cp)
                row = {"county": county, "origin": oname, "cut_date": ORIGIN_DATES[oname], "model": model,
                       "relMAE_icu_28d": relmae(Cp, fu["icu"].values), "relMAE_ward_28d": relmae(Wp, fu["ward"].values),
                       "NAIVE_icu_28d": relmae(np.full(len(fu), tr["icu"].iloc[-1]), fu["icu"].values),
                       "NAIVE_ward_28d": relmae(np.full(len(fu), tr["ward"].iloc[-1]), fu["ward"].values),
                       "adm_filled_days_in_projection": int(fu["adm_filled"].sum())}
                c10 = ctx[(ctx["city"] == CITY[county]) & (ctx["regime"] == oname)]
                if len(c10):
                    row["Phase10_ARIMA_icu_28d"] = float(c10["S2_ARIMA_28d_relMAE"].iloc[0])
                if not p13.empty:
                    c13 = p13[(p13["county"] == county) & (p13["origin"] == oname) & (p13["model"] == "H3")]
                    if len(c13):
                        row["Phase13_H3_icu_28d"] = float(c13["relMAE_icu_28d"].iloc[0])
                row["met"] = bool(row["relMAE_icu_28d"] <= CRIT["proj_icu"] and row["relMAE_ward_28d"] <= CRIT["proj_ward"])
                proj_rows.append(row)
                log(f"  projection {county} {oname} {model}: ICU {row['relMAE_icu_28d']:.3f} "
                    f"ward {row['relMAE_ward_28d']:.3f} (NAIVE ICU {row['NAIVE_icu_28d']:.3f})")

pd.DataFrame(met_rows).to_csv(out / "insample_metrics.csv", index=False)
pd.DataFrame(par_rows).to_csv(out / "params.csv", index=False)
if proj_rows:
    pd.DataFrame(proj_rows).to_csv(out / "projections.csv", index=False)
(out / "criteria_insample.json").write_text(json.dumps(crit, indent=2))
(out / "decision.json").write_text(json.dumps({**decision, "criteria": CRIT, "priors": PRIORS}, indent=2))
import scipy
man = {"script": Path(__file__).name, "script_sha256": sha(__file__), "command": " ".join([sys.executable] + sys.argv),
       "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0],
       "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
       "inputs_sha256": {p: sha(p) for p in sorted(INPUTS)},
       "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".json")}}
(out / "run_manifest.json").write_text(json.dumps(man, indent=2))
log(f"\nSaved to {out}/")
