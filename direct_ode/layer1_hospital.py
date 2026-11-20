#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 13, Layer 1: the hospital pathway (PHASE13_PROTOCOL.md with Addendum A). No neural network,
no transmission model, no torch.

Input  : observed daily confirmed COVID admissions A(t) (HHS, weekly sums spread over the week, from the
         official Phase 12 audit tables).
Fitted : CHHS ward census, CHHS ICU census (daily), CDPH deaths by date of death (weekly sums).

Model (Addendum A2 parameterization):
  ward  W: total outflow rate k_w = 1 / mean ward stay (stay 1 to 60 days), split into
           to ICU p_wc, death p_wd, discharge p_wr = 1 - p_wc - p_wd
  ICU   C: total outflow rate k_c = 1 / mean ICU stay (stay 1 to 60 days), split into
           death p_cd, survivors 1 - p_cd (H1: discharged, as in SUEIHCDR; H2 and H3: step down to ward)
    dW/dt = A(t) - k_w W + [H2,H3] (1 - p_cd) k_c C
    dC/dt = p_wc k_w W - k_c C
    hospital deaths/day = p_wd k_w W + p_cd k_c C ; observed deaths = hospital deaths / h, h in [0.5, 1]
  H1: p_wd = 0, constant rates.  H2: constant rates.  H3: H2 with rates specific to each Phase 12 period.
The linear system is solved exactly day by day (matrix exponential, admissions constant within a day).

Outputs in --out: insample_metrics.csv, criteria_insample.json, params.csv, fits_<county>_<model>.csv,
projections.csv, model_choice.json, run_log.txt, run_manifest.json.

Usage (project root, with the .venv Python that has scipy):
  python layer1_hospital.py --out phase13_layer1
"""
import argparse, hashlib, json, sys, platform, datetime as dt, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".", help="project root")
ap.add_argument("--audit", default="phase12_epi_audit")
ap.add_argument("--hosp-file", default="covid19hospitalbycounty.csv")
ap.add_argument("--cdph-ts", default="data_epi/statewide-covid-19-cases-deaths-tests.csv")
ap.add_argument("--phase10", default="phase10_sandiego_N_stage2/stage2_regime_table.csv,"
                                    "phase10_LA_SF_N_stage2/stage2_regime_table.csv")
ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
ap.add_argument("--models", default="H1,H2,H3")
ap.add_argument("--adm-series", default="adm_per_day", choices=["adm_per_day", "adm_per_day_supp2"])
ap.add_argument("--starts", type=int, default=5)
ap.add_argument("--max-nfev", type=int, default=400)
ap.add_argument("--skip-projections", action="store_true")
ap.add_argument("--out", default="phase13_layer1")
ap.add_argument("--overwrite", action="store_true")
A = ap.parse_args()

WINDOW = ("2020-08-02", "2022-12-31")
PERIODS = [("P1_FirstWave", "2020-03-29", "2020-09-30"), ("P2_Winter20", "2020-10-01", "2021-03-31"),
           ("P3_Delta", "2021-04-01", "2021-11-30"), ("P4_Omicron", "2021-12-01", "2022-04-30"),
           ("P5_BA2_BA5", "2022-05-01", "2022-12-31")]
ORIGINS = [("Winter20", "2021-01-15"), ("Delta", "2021-08-15"), ("Omicron", "2022-01-15"),
           ("BA5_Waning", "2022-06-15")]
CRIT = {"icu": 0.15, "ward": 0.15, "deaths": 0.25, "periods_needed": 4, "proj_icu": 0.20, "proj_ward": 0.20}
STAY = (1.0, 60.0); H_BOUNDS = (0.5, 1.0); LOGIT = 8.0
CITY = {"San Diego": "SanDiego", "Los Angeles": "LosAngeles", "San Francisco": "SanFrancisco"}

R = Path(A.root); out = Path(A.out)
if out.exists() and any(out.iterdir()) and not A.overwrite:
    raise SystemExit(f"{out} is not empty; use a new --out or --overwrite")
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "run_log.txt", "a", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


INPUTS = set()


# ------------------------------------------------------------------ data
def load_county(county):
    days = pd.date_range(WINDOW[0], WINDOW[1], freq="D")
    wk = pd.read_csv(R / A.audit / f"weekly_{CITY[county]}.csv", parse_dates=["week_start"])
    INPUTS.add(str(R / A.audit / f"weekly_{CITY[county]}.csv"))
    wk = wk.set_index("week_start")[A.adm_series]
    # Addendum A3: weeks without complete reporting are filled by log linear interpolation between
    # neighbouring complete weeks; the filled days are flagged.
    weekly_idx = pd.date_range(pd.Timestamp(WINDOW[0]) - pd.Timedelta(days=7), WINDOW[1], freq="7D")
    w = wk.reindex(weekly_idx)
    filled = w.isna()
    w = np.expm1(np.log1p(w).interpolate(limit_direction="both"))
    adm = w.reindex(days, method="ffill")
    adm_filled = filled.reindex(days, method="ffill").astype(bool)
    h = pd.read_csv(R / A.hosp_file); INPUTS.add(str(R / A.hosp_file))
    h = h[h["county"] == county].copy()
    h["date"] = pd.to_datetime(h["todays_date"], format="mixed")
    h = h.drop_duplicates("date", keep="last").set_index("date").sort_index()
    icu = h["icu_covid_confirmed_patients"].astype(float).reindex(days)
    ward = (h["hospitalized_covid_confirmed_patients"] - h["icu_covid_confirmed_patients"]).astype(float).reindex(days)
    t = pd.read_csv(R / A.cdph_ts, low_memory=False); INPUTS.add(str(R / A.cdph_ts))
    t = t[t["area"] == county].copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    deaths = pd.to_numeric(t.dropna(subset=["date"]).set_index("date")["deaths"], errors="coerce").reindex(days)
    df = pd.DataFrame({"adm": adm.values, "adm_filled": adm_filled.values, "ward": ward.values,
                       "icu": icu.values, "deaths": deaths.values}, index=days)
    pid = np.zeros(len(days), int)
    for i, (_, a, b) in enumerate(PERIODS):
        pid[(days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))] = i
    df["period"] = pid
    return df


# ------------------------------------------------------------------ model
def n_par(model, n_per):
    per = 4 if model == "H1" else 5
    return per * (n_per if model == "H3" else 1) + 1


def unpack(theta, model, n_per):
    """theta -> list of per period rate dicts, h. Raw: log stay_w, logit p_wc, [logit q], log stay_c,
    logit p_cd; p_wd = (1 - p_wc) * sigmoid(q)."""
    per = 4 if model == "H1" else 5
    sets = []
    for j in range(n_per if model == "H3" else 1):
        r = theta[j * per:(j + 1) * per]
        sig = lambda x: 1.0 / (1.0 + math.exp(-x))
        stay_w = math.exp(r[0]); p_wc = sig(r[1])
        if model == "H1":
            p_wd = 0.0; stay_c = math.exp(r[2]); p_cd = sig(r[3])
        else:
            p_wd = (1.0 - p_wc) * sig(r[2]); stay_c = math.exp(r[3]); p_cd = sig(r[4])
        sets.append({"k_w": 1.0 / stay_w, "p_wc": p_wc, "p_wd": p_wd, "k_c": 1.0 / stay_c, "p_cd": p_cd,
                     "stay_w": stay_w, "stay_c": stay_c})
    return sets, float(theta[-1])


def transition(p, stepdown):
    """Exact one day map for z = [W, C, D_cum]: z1 = Phi z0 + Gam * a (a = admissions per day)."""
    M = np.zeros((4, 4))
    M[0, 0] = -p["k_w"]; M[1, 0] = p["p_wc"] * p["k_w"]; M[1, 1] = -p["k_c"]
    if stepdown:
        M[0, 1] = (1.0 - p["p_cd"]) * p["k_c"]
    M[2, 0] = p["p_wd"] * p["k_w"]; M[2, 1] = p["p_cd"] * p["k_c"]
    M[0, 3] = 1.0  # input enters the ward
    E = expm(M)
    return E[:3, :3], E[:3, 3]


def simulate(theta, model, n_per, adm, pid, z0):
    sets, h = unpack(theta, model, n_per)
    maps = [transition(s, model != "H1") for s in sets]
    n = len(adm); Z = np.zeros((n + 1, 3)); Z[0] = z0
    for d in range(n):
        Phi, Gam = maps[pid[d] if model == "H3" else 0]
        Z[d + 1] = Phi @ Z[d] + Gam * adm[d]
    W, C = Z[:n, 0], Z[:n, 1]
    deaths_obs_model = np.diff(Z[:, 2]) / h
    return W, C, deaths_obs_model


def residuals(theta, model, n_per, df, pid_local):
    W, C, Dm = simulate(theta, model, n_per, df["adm"].values, pid_local,
                        np.array([df["ward"].iloc[0], df["icu"].iloc[0], 0.0]))
    rw = np.log1p(np.clip(W, 0, None)) - np.log1p(df["ward"].values)
    rc = np.log1p(np.clip(C, 0, None)) - np.log1p(df["icu"].values)
    nwk = len(df) // 7
    dm = Dm[:nwk * 7].reshape(nwk, 7).sum(1); do = df["deaths"].values[:nwk * 7].reshape(nwk, 7).sum(1)
    rd = np.log1p(np.clip(dm, 0, None)) - np.log1p(do)
    r = np.concatenate([rw, rc, rd])
    return np.where(np.isfinite(r), r, 0.0)


def bounds(model, n_per):
    per_lo = [math.log(STAY[0]), -LOGIT] + ([-LOGIT] if model != "H1" else []) + [math.log(STAY[0]), -LOGIT]
    per_hi = [math.log(STAY[1]), LOGIT] + ([LOGIT] if model != "H1" else []) + [math.log(STAY[1]), LOGIT]
    k = n_per if model == "H3" else 1
    return np.array(per_lo * k + [H_BOUNDS[0]]), np.array(per_hi * k + [H_BOUNDS[1]])


def start_point(model, n_per, s):
    """Start 0: ward stay 8 d, 20 % to ICU, 3 % ward death share; ICU stay 12 d, 30 % ICU death; h 0.85.
    Starts 1..: uniform in the bounds with a fixed seed."""
    lo, hi = bounds(model, n_per)
    if s == 0:
        base = [math.log(8.0), math.log(0.2 / 0.8)] + ([math.log(0.03 / 0.97)] if model != "H1" else []) + \
               [math.log(12.0), math.log(0.3 / 0.7)]
        x = np.array(base * (n_per if model == "H3" else 1) + [0.85])
    else:
        rng = np.random.default_rng(20261011 + 1000 * s + {"H1": 1, "H2": 2, "H3": 3}[model])
        x = lo + (hi - lo) * rng.uniform(0.05, 0.95, size=len(lo))
    return np.clip(x, lo + 1e-9, hi - 1e-9)


def fit(df, model):
    present = sorted(set(df["period"].values))
    remap = {p: i for i, p in enumerate(present)}
    pid_local = np.array([remap[p] for p in df["period"].values])
    n_per = len(present)
    lo, hi = bounds(model, n_per)
    best, costs = None, []
    for s in range(A.starts):
        r = least_squares(residuals, start_point(model, n_per, s), bounds=(lo, hi), method="trf",
                          args=(model, n_per, df, pid_local), max_nfev=A.max_nfev, x_scale="jac")
        costs.append(float(r.cost))
        if best is None or r.cost < best.cost:
            best = r
    return best, costs, n_per, present, pid_local


def relmae(m, o):
    ok = np.isfinite(o) & np.isfinite(m)
    return float(np.mean(np.abs(m[ok] - o[ok])) / max(np.mean(o[ok]), 1e-9)) if ok.any() else float("nan")


# ------------------------------------------------------------------ run
counties = [c.strip() for c in A.counties.split(",") if c.strip()]
models = [m.strip() for m in A.models.split(",") if m.strip()]
log(f"Phase 13 Layer 1 | models {models} | counties {counties} | adm {A.adm_series} | starts {A.starts}")
met_rows, par_rows, proj_rows, crit = [], [], [], {}
data = {c: load_county(c) for c in counties}
for county in counties:
    df = data[county]
    log(f"\n{county}: {len(df)} days, admissions filled on {int(df['adm_filled'].sum())} days")
    for model in models:
        res, costs, n_per, present, pid_local = fit(df, model)
        W, C, Dm = simulate(res.x, model, n_per, df["adm"].values, pid_local,
                            np.array([df["ward"].iloc[0], df["icu"].iloc[0], 0.0]))
        sets, h = unpack(res.x, model, n_per)
        traj = df.copy(); traj["ward_model"] = W; traj["icu_model"] = C; traj["deaths_model"] = Dm
        traj.to_csv(out / f"fits_{CITY[county]}_{model}.csv", index_label="date")
        n_ok = 0
        for p in present:
            sub = traj[traj["period"] == p]
            wk = sub[["deaths", "deaths_model"]].resample("W-SAT").sum(min_count=7).dropna()
            m = {"county": county, "model": model, "period": PERIODS[p][0], "days": len(sub),
                 "relMAE_icu": relmae(sub["icu_model"].values, sub["icu"].values),
                 "relMAE_ward": relmae(sub["ward_model"].values, sub["ward"].values),
                 "relMAE_deaths_weekly": relmae(wk["deaths_model"].values, wk["deaths"].values)}
            m["met"] = bool(m["relMAE_icu"] <= CRIT["icu"] and m["relMAE_ward"] <= CRIT["ward"]
                            and m["relMAE_deaths_weekly"] <= CRIT["deaths"])
            n_ok += m["met"]; met_rows.append(m)
        crit[f"{county}|{model}"] = {"periods_met": n_ok, "periods": len(present),
                                     "pass": bool(n_ok >= min(CRIT["periods_needed"], len(present)))}
        for j, s in enumerate(sets):
            par_rows.append({"county": county, "model": model,
                             "period": PERIODS[present[j]][0] if model == "H3" else "all",
                             **{k: round(v, 5) for k, v in s.items()}, "h": round(h, 4),
                             "cost": float(res.cost), "cost_starts": json.dumps([round(c, 4) for c in costs])})
        log(f"  {model}: cost {res.cost:.3f} (starts {[round(c, 2) for c in costs]}) | periods met "
            f"{n_ok}/{len(present)} | h {h:.3f}")

choice = {}
for model in models:
    if all(crit[f"{c}|{model}"]["pass"] for c in counties):
        choice = {"chosen_model": model, "rule": "simplest of H1, H2, H3 meeting the in sample criteria in all counties"}
        break
if not choice:
    choice = {"chosen_model": None, "rule": "none met the in sample criteria in all counties: report failure"}
log("\nModel choice:", choice)

# ------------------------------------------------------------------ projections (Addendum A5)
if not A.skip_projections:
    ctx = pd.concat([pd.read_csv(R / p) for p in A.phase10.split(",")]) if A.phase10 else pd.DataFrame()
    for p in A.phase10.split(","):
        INPUTS.add(str(R / p))
    for county in counties:
        df = data[county]
        for oname, odate in ORIGINS:
            o = pd.Timestamp(odate)
            tr = df[df.index <= o]; fu = df[(df.index > o) & (df.index <= o + pd.Timedelta(days=28))]
            for model in models:
                res, costs, n_per, present, pid_local = fit(tr, model)
                sets, h = unpack(res.x, model, n_per)
                p_last = sets[-1]
                Phi, Gam = transition(p_last, model != "H1")
                z = np.array([tr["ward"].iloc[-1], tr["icu"].iloc[-1], 0.0])   # observed census at the origin
                Wp, Cp = [], []
                for a in fu["adm"].values:
                    z = Phi @ z + Gam * a; Wp.append(z[0]); Cp.append(z[1])
                Wp, Cp = np.array(Wp), np.array(Cp)
                row = {"county": county, "origin": oname, "cut_date": odate, "model": model,
                       "relMAE_icu_28d": relmae(Cp, fu["icu"].values),
                       "relMAE_ward_28d": relmae(Wp, fu["ward"].values),
                       "NAIVE_icu_28d": relmae(np.full(len(fu), tr["icu"].iloc[-1]), fu["icu"].values),
                       "NAIVE_ward_28d": relmae(np.full(len(fu), tr["ward"].iloc[-1]), fu["ward"].values),
                       "adm_filled_days_in_projection": int(fu["adm_filled"].sum())}
                if not ctx.empty:
                    c10 = ctx[(ctx["city"] == CITY[county]) & (ctx["regime"] == oname)]
                    if len(c10):
                        row["Phase10_ARIMA_icu_28d"] = float(c10["S2_ARIMA_28d_relMAE"].iloc[0])
                        row["Phase10_PINN_icu_28d_allseeds"] = float(c10["S2_relMAE_all_median"].iloc[0])
                row["met"] = bool(row["relMAE_icu_28d"] <= CRIT["proj_icu"] and row["relMAE_ward_28d"] <= CRIT["proj_ward"])
                proj_rows.append(row)
                log(f"  projection {county} {oname} {model}: ICU {row['relMAE_icu_28d']:.3f} "
                    f"ward {row['relMAE_ward_28d']:.3f} (NAIVE ICU {row['NAIVE_icu_28d']:.3f})")

pd.DataFrame(met_rows).to_csv(out / "insample_metrics.csv", index=False)
pd.DataFrame(par_rows).to_csv(out / "params.csv", index=False)
if proj_rows:
    pd.DataFrame(proj_rows).to_csv(out / "projections.csv", index=False)
(out / "criteria_insample.json").write_text(json.dumps(crit, indent=2))
(out / "model_choice.json").write_text(json.dumps({**choice, "criteria": CRIT}, indent=2))
import scipy
man = {"script": Path(__file__).name, "script_sha256": sha(__file__), "command": " ".join([sys.executable] + sys.argv),
       "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0],
       "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
       "inputs_sha256": {p: sha(p) for p in sorted(INPUTS)},
       "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".json")}}
(out / "run_manifest.json").write_text(json.dumps(man, indent=2))
log(f"\nSaved to {out}/")
