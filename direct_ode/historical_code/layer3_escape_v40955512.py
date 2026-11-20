#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 17: immune escape in real time and an external check of immunity (PHASE17_PROTOCOL.md with
Addendum A; Addendum B records this implementation). No neural network, no torch.

L3 = Phase 16 L2 plus leaky protection by variant group:
  groups pre (before Omicron), O1 (BA.1, BA.2 and their sublineages), O2 (BA.4, BA.5 and later);
  shares p_pre, p_O1, p_O2: Omicron share from CDPH (known for specimens up to t0 - 14), split into O1 and
  O2 with the CDC Variant Proportions for HHS Region 9 published on or before t0; later days extrapolated
  by logistic growth fitted to the last 3 known weeks.
  States: S, U_v, U_b, E, I, R_pre, R_O1, R_O2 (+ CUM, cumulative infections per person, bookkeeping).
    lam = beta I
    dS    = -lam S - v S - a_b s S + omega_v U_v + omega_b (1 - s) U_b
    dU_v  =  v S - omega_v U_v - lam U_v (p_O1 + p_O2) f_V
    dU_b  =  a_b s S - omega_b (1 - s) U_b
    dE    =  lam [S + U_v (p_O1 + p_O2) f_V + R_pre (p_O1 f_pre,O1 + p_O2 f_pre,O2) + R_O1 p_O2 f_O1,O2] - sigma E
    dI    =  sigma E - delta I
    dR_g  =  delta I p_g - (leak of R_g)
  f values are used only from the date their source became public (protocol section 3); eta = 0.
Usage (project root, .venv Python):  python layer3_escape.py --out phase17_layer3
"""
import argparse, hashlib, json, sys, platform, datetime as dt, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import layer2_admissions as L2

DAY = pd.Timedelta(days=1)
F_R_PRE_O1 = [("2022-01-06", 0.38)]
F_R_PRE_O2 = [("2022-07-12", 0.69)]
F_R_O1_O2 = [("2022-07-12", 0.11)]
F_V_DATE = "2021-12-14"; F_V_BOOST = 0.18; F_V_PRIMARY = 0.45
CDPH_LAG = 14
O1_NAMES = {"B.1.1.529", "BA.1", "BA.1.1", "BA.2", "BA.2.12.1", "BA.3"}
PRE_PREFIX = ("A.", "AY.", "B.1", "P.", "R.")
BETA_MAX = L2.BETA_BOUNDS[1]
SERO_LO, SERO_HI = -0.10, 0.20
SERO_MEASURE = "compartments"     # Addendum B1 (exact ever infected, since eta = 0); "poisson" = protocol formula
CHAINS = [("L3", 1, True), ("L3", 0, True), ("L3_NOBEHAV", 1, False)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def fval(table, t0):
    v = 0.0
    for d, x in table:
        if t0 >= pd.Timestamp(d):
            v = x
    return v


def group_of(name):
    if name == "Other":
        return None
    if name in O1_NAMES:
        return "O1"
    if name.startswith("B.1.1.529"):
        return "O1"
    if name.startswith(PRE_PREFIX):
        return "pre"
    return "O2"


# ------------------------------------------------------------------ variant data
def load_variants(root, inputs):
    p1 = Path(root) / "data_epi/covid19_variants.csv"; inputs.add(str(p1))
    c = pd.read_csv(p1); c = c[c["variant_name"] == "Omicron"].copy(); c["date"] = pd.to_datetime(c["date"])
    val = c["percentage_7d_avg"].where(c["percentage_7d_avg"].notna(), c["percentage"])
    om = pd.Series(val.values / 100.0, index=c["date"]).sort_index().clip(0, 1)
    p2 = Path(root) / "data_epi/jr58-6ysp.csv"; inputs.add(str(p2))
    v = pd.read_csv(p2)
    v = v[(v["modeltype"] == "weighted") & (v["usa_or_hhsregion"].astype(str) == "9")].copy()
    v["week_ending"] = pd.to_datetime(v["week_ending"]); v["creation_date"] = pd.to_datetime(v["creation_date"])
    v["group"] = v["variant"].map(group_of)
    v = v[v["group"].isin(["O1", "O2"])]
    return om, v[["week_ending", "creation_date", "group", "share"]]


def logistic_extrap(series_known, days_out):
    """Weekly known values (index = dates); returns values on days_out: known values held between points
    (last known on or before the day), logistic extrapolation of the last 3 weekly points after the end."""
    s = series_known.dropna()
    if s.empty:
        return np.zeros(len(days_out))
    wk = s.reindex(pd.date_range(end=s.index.max(), periods=400, freq="7D")).dropna()
    last_day = s.index.max(); out = np.empty(len(days_out))
    k = 0.0; z_last = None
    pts = wk.iloc[-3:]
    if len(pts) == 3 and pts.between(0.001, 0.999).all():
        z = np.log(pts.values / (1 - pts.values)); tt = np.array([(d - pts.index[-1]).days for d in pts.index], float)
        k = float(np.polyfit(tt, z, 1)[0]); z_last = float(z[-1])
    for i, d in enumerate(days_out):
        if d <= last_day:
            out[i] = float(s[s.index <= d].iloc[-1]) if (s.index <= d).any() else 0.0
        elif z_last is not None:
            out[i] = 1.0 / (1.0 + math.exp(-(z_last + k * (d - last_day).days)))
        else:
            out[i] = float(s.iloc[-1])
    return np.clip(out, 0.0, 1.0)


def shares_at(om, cdc, t0, days):
    """p_pre, p_O1, p_O2 on the given days, as known on t0."""
    om_known = om[om.index <= t0 - pd.Timedelta(days=CDPH_LAG)]
    p_om = logistic_extrap(om_known, days)
    vint = cdc[cdc["creation_date"] <= t0]
    if vint.empty:
        q = np.zeros(len(days))
    else:
        last = vint.groupby("week_ending")["creation_date"].transform("max")
        vv = vint[vint["creation_date"].values == last.values]
        g = vv.groupby(["week_ending", "group"])["share"].sum().unstack(fill_value=0.0)
        if "O2" not in g:
            q = np.zeros(len(days))
        else:
            tot = g.sum(axis=1); qs = (g["O2"] / tot.where(tot > 0)).dropna()
            qs = qs[qs.index <= t0]
            q = logistic_extrap(qs, days) if len(qs) else np.zeros(len(days))
            if (qs > 0).sum() == 0:
                q = np.zeros(len(days))
    return 1.0 - p_om, p_om * (1.0 - q), p_om * q


# ------------------------------------------------------------------ model
def rhs3(x, beta, v, ab_s, omb_eff, omv, f, pO1, pO2, ppre):
    S, Uv, Ub, E, I, Rp, R1, R2 = x
    lam = beta * I
    lv = lam * Uv * (pO1 + pO2) * f["V"]
    lp = lam * Rp * (pO1 * f["pre_O1"] + pO2 * f["pre_O2"])
    l1 = lam * R1 * pO2 * f["O1_O2"]
    inf = lam * S + lv + lp + l1
    return (-lam * S - v * S - ab_s * S + omv * Uv + omb_eff * Ub,
            v * S - omv * Uv - lv,
            ab_s * S - omb_eff * Ub,
            inf - L2.SIGMA * E,
            L2.SIGMA * E - L2.DELTA * I,
            L2.DELTA * I * ppre - lp,
            L2.DELTA * I * pO1 - l1,
            L2.DELTA * I * pO2)


def simulate3(x0, beta_d, dose_d, s_d, sh, ab, omb, omv, ve, f):
    n = len(beta_d); X = np.empty((n + 1, 9)); fE = np.empty(n); fI = np.empty(n)
    x = list(x0[:8]); cum = float(x0[8]); X[0] = list(x) + [cum]; h = 1.0 / L2.SUBSTEPS
    ppre_a, pO1_a, pO2_a = sh
    for d in range(n):
        b = beta_d[d]; s = s_d[d]; ab_s = ab * s; omb_eff = omb * (1.0 - s)
        ppre, pO1, pO2 = ppre_a[d], pO1_a[d], pO2_a[d]
        accE = 0.0; accI = 0.0
        for _ in range(L2.SUBSTEPS):
            S, Uv, Ub, E, I, Rp, R1, R2 = x
            nv = max(S + E + I + Rp + R1 + R2, 1e-12)
            v = min(ve * dose_d[d] / nv, 1.0)
            a = (b, v, ab_s, omb_eff, omv, f, pO1, pO2, ppre)
            k1 = rhs3(x, *a)
            x2 = [x[i] + 0.5 * h * k1[i] for i in range(8)]; k2 = rhs3(x2, *a)
            x3 = [x[i] + 0.5 * h * k2[i] for i in range(8)]; k3 = rhs3(x3, *a)
            x4 = [x[i] + h * k3[i] for i in range(8)]; k4 = rhs3(x4, *a)
            Eavg = (x[3] + 2 * x2[3] + 2 * x3[3] + x4[3]) / 6.0; Iavg = (x[4] + 2 * x2[4] + 2 * x3[4] + x4[4]) / 6.0
            accE += L2.SIGMA * Eavg * h; accI += L2.DELTA * Iavg * h
            x = [max(x[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0, 0.0) for i in range(8)]
        cum += accE
        X[d + 1] = list(x) + [cum]; fE[d] = accE; fI[d] = accI
    return X, fE, fI


class Problem3(L2.Problem):
    def __init__(self, K, D, t0, x_carry, behav, n_extra, om, cdc, w_b):
        super().__init__(K, D, t0, x_carry, behav, 0.0, n_extra)
        self.sh = shares_at(om, cdc, t0, self.days)
        fv = (w_b * F_V_BOOST + (1 - w_b) * F_V_PRIMARY) if t0 >= pd.Timestamp(F_V_DATE) else 0.0
        self.f = {"V": fv, "pre_O1": fval(F_R_PRE_O1, t0), "pre_O2": fval(F_R_PRE_O2, t0), "O1_O2": fval(F_R_O1_O2, t0)}
        self.w_b = w_b

    def x0(self, th):
        x = self.x_carry.copy(); x[3] *= math.exp(th[13]); x[4] *= math.exp(th[14])
        x[0] = max(1.0 - (x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]), 1e-6)
        return x

    def run(self, th, n=None):
        p = self.unpack(th); n = n or self.n
        X, fE, fI = simulate3(self.x0(th), L2.beta_series(p["logk"], n), self.dose_d[:n], self.s_d[:n],
                              tuple(a[:n] for a in self.sh), p["a_b"], p["omega_b"], self.omv, self.ve, self.f)
        return p, X, fE * self.N * p["rho"], fI * self.N * p["ihr"]


def booster_weight(vax_cum, t0):
    k = vax_cum[vax_cum.index <= t0 - pd.Timedelta(days=7)]
    if k.empty:
        return 0.0
    r = k.iloc[-1]
    return float(r["boost"] / r["full"]) if r["full"] > 0 else 0.0


def load_vax_cum(vax_file, county, inputs):
    v = pd.read_csv(vax_file); inputs.add(str(vax_file)); v = v[v["county"] == county].copy()
    v["date"] = pd.to_datetime(v["administered_date"]); v = v.set_index("date").sort_index()
    return pd.DataFrame({"full": pd.to_numeric(v["cumulative_fully_vaccinated"], errors="coerce").fillna(0.0),
                         "boost": pd.to_numeric(v["cumulative_booster_recip_count"], errors="coerce").fillna(0.0)})


def l3_origin(D, t0, delay, behav, x_carry, prev_theta, starts, max_nfev, seed_key, om, cdc, vax_cum):
    K = L2.known_inputs(D, t0, delay)
    if K["wk"].dropna().empty:
        return None
    last_ws = K["wk"].dropna().index.max()
    target_weeks = [last_ws + pd.Timedelta(days=7 * h) for h in range(1, L2.HWEEKS + 1)]
    end_day = target_weeks[-1] + pd.Timedelta(days=6)
    P = Problem3(K, D, t0, x_carry, behav, (end_day - (t0 - DAY)).days, om, cdc, booster_weight(vax_cum, t0))
    lo, hi = P.bounds(); best, costs = None, []
    for s in range(starts):
        r = least_squares(P.residuals, P.start(s, prev_theta, seed_key), bounds=(lo, hi), method="trf",
                          max_nfev=max_nfev, x_scale="jac")
        costs.append(float(r.cost))
        if best is None or r.cost < best.cost:
            best = r
    p, X, cases, adm = P.run(best.x)
    pred = {h: (ws, float(adm[(ws - P.w0).days:(ws - P.w0).days + 7].mean())) for h, ws in enumerate(target_weeks, 1)}
    i0 = L2.WINDOW; x = X[i0]
    S_eff = x[0] + x[1] * (P.sh[1][i0] + P.sh[2][i0]) * P.f["V"] + x[5] * (P.sh[1][i0] * P.f["pre_O1"] + P.sh[2][i0] * P.f["pre_O2"]) \
        + x[6] * P.sh[2][i0] * P.f["O1_O2"]
    names = ["S", "U_v", "U_b", "E", "I", "R_pre", "R_O1", "R_O2", "CUM"]
    info = {"origin": str(t0.date()), "rho": p["rho"], "ihr": p["ihr"], "a_b": p["a_b"], "omega_b": p["omega_b"],
            "mult_E": p["mult_E"], "mult_I": p["mult_I"], "beta_last": float(math.exp(p["logk"][-1])),
            "Rt_t0": float(math.exp(p["logk"][-1]) * S_eff / L2.DELTA), "S_eff_t0": float(S_eff),
            **{f"{k}_t0": float(x[j]) for j, k in enumerate(names)},
            "p_pre_t0": float(P.sh[0][i0]), "p_O1_t0": float(P.sh[1][i0]), "p_O2_t0": float(P.sh[2][i0]),
            "f_V": P.f["V"], "f_pre_O1": P.f["pre_O1"], "f_pre_O2": P.f["pre_O2"], "f_O1_O2": P.f["O1_O2"], "w_b": P.w_b,
            "s_t0": float(P.s_d[L2.WINDOW - 1]), "VE": P.ve, "omega_v": P.omv, "rho_prior": P.rho_prior,
            "cost": float(best.cost), "cost_starts": json.dumps([round(c, 4) for c in costs]), "nfev": int(best.nfev),
            "last_known_week_start": str(last_ws.date()), "n_adm_weeks_in_window": len(P.weeks),
            "beta_knots": json.dumps([round(float(math.exp(v)), 5) for v in p["logk"]])}
    traj = pd.DataFrame({"date": P.days, "adm_model": adm, "cases_model": cases,
                         **{k: X[:-1, j] for j, k in enumerate(names)},
                         "p_pre": P.sh[0], "p_O1": P.sh[1], "p_O2": P.sh[2]})
    return {"pred": pred, "x_next": X[7].copy(), "theta": best.x, "info": info, "traj": traj, "K": K,
            "target_weeks": target_weeks, "adm_daily": pd.Series(adm, index=P.days)}


def initial_state3(D, w0, rho_prior):
    x = L2.initial_state(D, w0, rho_prior)          # S, U_v, U_b, E, I, R
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5], 0.0, 0.0, x[3] + x[4] + x[5]])


# ------------------------------------------------------------------ serology
def parse_range(txt):
    """'Jul 30 -  Aug 5, 2020' or 'Dec 28, 2020 - Jan 10, 2021' -> (start, end)."""
    try:
        a, b = [x.strip() for x in " ".join(txt.split()).split(" - ")]
        b_d = pd.to_datetime(b)
        a_d = pd.to_datetime(a) if "," in a else pd.to_datetime(f"{a}, {b_d.year}")
        if a_d > b_d:
            a_d = a_d - pd.DateOffset(years=1)
        return a_d, b_d
    except Exception:
        return None, None


def load_serology(root, inputs):
    rows = []
    p = Path(root) / "data_epi/Nationwide_Commercial_Laboratory_Seroprevalence_Survey.csv"; inputs.add(str(p))
    c = pd.read_csv(p); c = c[c["Site"] == "CA"]
    col = "Rate (%) [Anti-N, All Ages Cumulative Prevalence, Rounds 1-30 only]"
    for _, r in c.iterrows():
        if pd.isna(r.get(col)):
            continue
        a_d, b_d = parse_range(str(r["Date Range of Specimen Collection"]))
        if a_d is None:
            continue
        mid = a_d + (b_d - a_d) / 2
        for cty in ("San Diego", "Los Angeles", "San Francisco"):
            rows.append({"county": cty, "source": "commercial_lab_CA", "date": mid.normalize(), "sero": float(r[col]) / 100.0})
    p = Path(root) / "data_epi/2020-2021_Nationwide_Blood_Donor_Seroprevalence_Survey_Infection-Induced_Seroprevalence_Estimates.csv"
    inputs.add(str(p)); b = pd.read_csv(p)
    datecol = [c_ for c_ in b.columns if c_.startswith("Median")][0]
    for _, r in b.iterrows():
        ab = r["Region Abbreviation"]
        ctys = {"CA-1": ["San Francisco"], "CA-3": ["Los Angeles", "San Diego"]}.get(ab, [])
        for cty in ctys:
            rows.append({"county": cty, "source": f"blood_donor_{ab}", "date": pd.to_datetime(r[datecol]),
                         "sero": float(r["Rate %[Total Prevalence]"]) / 100.0})
    p = Path(root) / "data_epi/2022_2023_Nationwide_Blood_Donor_Seroprevalence_Survey_Combined_Infection-_and_Vaccination-Induced_Seroprevalence_Estimates.csv"
    inputs.add(str(p)); b = pd.read_csv(p)
    b = b[(b["Indicator"] == "Past infection with or without vaccination") & (b["Geographic Area"] == "California")
          & (b["Race"] == "Overall") & (b["Sex"] == "Overall") & (b["Age"] == "Overall")]
    for _, r in b.iterrows():
        yr, q = r["Time Period"].split(" Quarter ")
        mid = pd.Timestamp(int(yr), 3 * int(q) - 1, 15)
        for cty in ("San Diego", "Los Angeles", "San Francisco"):
            rows.append({"county": cty, "source": "blood_donor_2022_CA", "date": mid, "sero": float(r["Estimate % (weighted)"]) / 100.0})
    return pd.DataFrame(rows)


def model_at(trajs, county, model, delay, d, cols):
    """State at day d from the fit at the first origin on or after d whose trajectory covers d."""
    t = trajs[(trajs["county"] == county) & (trajs["model"] == model) & (trajs["delay"] == delay)]
    t = t[(t["date"] == d) & (pd.to_datetime(t["origin"]) >= d)]
    if t.empty:
        return None
    r = t.sort_values("origin").iloc[0]
    return {c: float(r[c]) for c in cols}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--adm-dir", default="phase15_admissions")
    ap.add_argument("--vax", default="data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv")
    ap.add_argument("--hosp-file", default="covid19hospitalbycounty.csv")
    ap.add_argument("--phase16", default="phase16_layer2/trajectories.csv")
    ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
    ap.add_argument("--first-origin", default=L2.FIRST_ORIGIN)
    ap.add_argument("--max-origins", type=int, default=0, help="smoke tests only")
    ap.add_argument("--chains", default="all", help="smoke tests only")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--max-nfev", type=int, default=300)
    ap.add_argument("--skip-arima", action="store_true", help="smoke tests only")
    ap.add_argument("--skip-icu", action="store_true", help="smoke tests only")
    ap.add_argument("--out", default="phase17_layer3")
    ap.add_argument("--overwrite", action="store_true")
    A = ap.parse_args(argv)
    if A.first_origin != L2.FIRST_ORIGIN and not A.max_origins:
        raise SystemExit("--first-origin is for smoke tests only")
    out = Path(A.out)
    if out.exists() and any(out.iterdir()) and not A.overwrite:
        raise SystemExit(f"{out} is not empty; use a new --out or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    LOG = open(out / "run_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    inputs = set()
    counties = [c.strip() for c in A.counties.split(",") if c.strip()]
    origins = list(pd.date_range(A.first_origin, L2.LAST_ORIGIN, freq="7D"))
    if A.max_origins:
        origins = origins[:A.max_origins]
    chains = CHAINS if A.chains == "all" else [CHAINS[int(i)] for i in A.chains.split(",")]
    om, cdc = load_variants(A.root, inputs)
    log(f"Phase 17 | counties {counties} | origins {len(origins)} ({origins[0].date()} to {origins[-1].date()}) | "
        f"chains {[(c[0], c[1]) for c in chains]} | starts {A.starts} | max_nfev {A.max_nfev}")
    import layer1c_recalibration as L1C
    pred_rows, info_rows, trajs, icu_rows = [], [], [], []
    for ci, county in enumerate(counties):
        D = L2.load_county(A.root, county, A.adm_dir, A.vax, inputs)
        vax_cum = load_vax_cum(A.vax, county, inputs)
        census = None if A.skip_icu else L1C.load_county(A.root, A.adm_dir, A.hosp_file, county, inputs)[0]
        for chain_i, (name, delay, behav) in enumerate(chains):
            t_start = time.time()
            x = initial_state3(D, origins[0] - pd.Timedelta(days=L2.WINDOW), L2.sched(L2.RHO_PRIOR_SCHED, origins[0]))
            theta = None
            for oi, t0 in enumerate(origins):
                res = l3_origin(D, t0, delay, behav, x, theta, A.starts, A.max_nfev, (ci, chain_i, int(t0.toordinal())),
                                om, cdc, vax_cum)
                if res is None:
                    continue
                x, theta = res["x_next"], res["theta"]
                info_rows.append({"county": county, "model": name, "delay": delay, "period": L2.period_of(t0), **res["info"]})
                tr = res["traj"]; tr.insert(0, "origin", str(t0.date())); tr.insert(0, "delay", delay)
                tr.insert(0, "model", name); tr.insert(0, "county", county); trajs.append(tr)
                for h, (ws, v) in res["pred"].items():
                    pred_rows.append({"county": county, "origin": str(t0.date()), "period": L2.period_of(t0), "model": name,
                                      "delay": delay, "h": h, "week_start": str(ws.date()), "pred": v,
                                      "obs": float(D["wk"].get(ws, np.nan))})
                if name == "L3":
                    for mname, (preds, cinfo) in L2.comparators(res["K"], res["target_weeks"], A.skip_arima).items():
                        for h, ws in enumerate(res["target_weeks"], 1):
                            pred_rows.append({"county": county, "origin": str(t0.date()), "period": L2.period_of(t0),
                                              "model": mname, "delay": delay, "h": h, "week_start": str(ws.date()),
                                              "pred": preds[h], "obs": float(D["wk"].get(ws, np.nan)), **cinfo})
                    if census is not None and delay == 1:
                        icu_rows.append(L2.icu_from_l2(L1C, census, res["K"], res["adm_daily"], t0, A, county))
                if (oi + 1) % 10 == 0 or oi + 1 == len(origins):
                    log(f"  {county} {name} delay {delay}: {oi + 1}/{len(origins)} origins ({time.time() - t_start:.0f} s)")
    Pd = pd.DataFrame(pred_rows); Pd["abs_err"] = (Pd["pred"] - Pd["obs"]).abs(); Pd.to_csv(out / "forecasts_weekly.csv", index=False)
    Inf = pd.DataFrame(info_rows); Inf.to_csv(out / "params_states.csv", index=False)
    T = pd.concat(trajs); T.to_csv(out / "trajectories.csv", index=False)
    Icu = pd.DataFrame(icu_rows)
    if len(Icu):
        Icu.to_csv(out / "icu_secondary.csv", index=False)

    prim = (Pd[Pd["h"].isin(L2.PRIMARY_H)].groupby(["county", "origin", "period", "model", "delay"])
            .agg(err=("abs_err", "mean"), obs=("obs", "mean"), n=("abs_err", "count")).reset_index())
    prim = prim[prim["n"] == len(L2.PRIMARY_H)]; prim.to_csv(out / "primary_error_origin.csv", index=False)
    summ = prim.groupby(["county", "model", "delay"]).agg(n_origins=("err", "size"), mean_err=("err", "mean"),
                                                         mean_obs=("obs", "mean")).reset_index()
    summ["rel_err"] = summ["mean_err"] / summ["mean_obs"]; summ.to_csv(out / "summary.csv", index=False)
    Pd.groupby(["county", "model", "delay", "h"]).agg(mean_abs_err=("abs_err", "mean")).reset_index().to_csv(out / "summary_by_h.csv", index=False)
    prim.groupby(["county", "period", "model", "delay"])["err"].mean().reset_index().to_csv(out / "summary_by_period.csv", index=False)

    def paired(county, a, b, delay=1):
        x_ = prim[(prim["county"] == county) & (prim["model"] == a) & (prim["delay"] == delay)].set_index("origin")["err"]
        y_ = prim[(prim["county"] == county) & (prim["model"] == b) & (prim["delay"] == delay)].set_index("origin")["err"]
        j = pd.concat([x_.rename("a"), y_.rename("b")], axis=1).dropna().sort_index()
        d = (j["a"] - j["b"]).values; lo, hi = L2.block_boot_ci(d)
        return {"n_origins": int(len(j)), f"mean_{a}": float(j["a"].mean()) if len(j) else None,
                f"mean_{b}": float(j["b"].mean()) if len(j) else None, "mean_diff": float(d.mean()) if len(d) else None,
                "ci95": [lo, hi], "significantly_better": bool(len(d) and d.mean() < 0 and np.isfinite(hi) and hi < 0),
                "not_significantly_worse": bool(len(d) and np.isfinite(lo) and lo <= 0)}

    crit = {"plausibility": {}, "serology": {}, "adequacy": {}, "D1": {}, "D2": {}}
    for c in counties:
        q = Inf[(Inf["county"] == c) & (Inf["model"] == "L3") & (Inf["delay"] == 1)]
        nb = int((q["beta_last"] > BETA_MAX - 0.01).sum()); nr = int(((q["rho"] < 0.0051) | (q["rho"] > 0.99)).sum())
        crit["plausibility"][c] = {"n_origins": int(len(q)), "beta_at_upper": nb, "rho_at_bound": nr,
                                   "met": bool(len(q) and nb <= 0.10 * len(q) and nr <= 0.10 * len(q))}
        if "ARIMA_ADM" in set(prim["model"]):
            crit["D1"][c] = paired(c, "L3", "ARIMA_ADM")
            crit["adequacy"][c] = {"not_significantly_worse_than_ARIMA_ADM": crit["D1"][c]["not_significantly_worse"]}
        crit["D2"][c] = paired(c, "L3", "CASELAG")

    # serology check (never used in fitting)
    sero = load_serology(A.root, inputs)
    T["date"] = pd.to_datetime(T["date"])
    lastd = T["date"].max(); firstd = T["date"].min()
    sero = sero[(sero["date"] >= firstd) & (sero["date"] <= lastd)].copy()
    p16 = None
    if Path(A.phase16).exists():
        inputs.add(str(A.phase16)); p16 = pd.read_csv(A.phase16, usecols=["county", "model", "delay", "origin", "date", "S", "U_v", "U_b", "E", "I", "R"])
        p16["date"] = pd.to_datetime(p16["date"])
    srows = []
    for _, r in sero.iterrows():
        m = model_at(T, r["county"], "L3", 1, r["date"], ["CUM", "S", "U_v", "U_b", "E", "I", "R_pre", "R_O1", "R_O2"])
        row = {**r.to_dict()}
        if m is not None:
            row["L3_ever_poisson"] = 1.0 - math.exp(-m["CUM"])
            row["L3_ever_compartments"] = m["R_pre"] + m["R_O1"] + m["R_O2"] + m["E"] + m["I"]
            for mname in ("compartments", "poisson"):
                dfv = row[f"L3_ever_{mname}"] - r["sero"]
                row[f"L3_diff_{mname}"] = dfv; row[f"L3_within_{mname}"] = bool(SERO_LO <= dfv <= SERO_HI)
            row["L3_within"] = row[f"L3_within_{SERO_MEASURE}"]
        if p16 is not None:
            m2 = model_at(p16, r["county"], "L2", 1, r["date"], ["R", "E", "I"])
            if m2 is not None:
                row["L2_ever_compartments"] = m2["R"] + m2["E"] + m2["I"]
        srows.append(row)
    Sg = pd.DataFrame(srows); Sg.to_csv(out / "serology_check.csv", index=False)
    for c in counties:
        s_ = Sg[(Sg["county"] == c) & Sg["L3_within"].notna()] if "L3_within" in Sg else Sg.iloc[0:0]
        n = int(len(s_)); k = int(s_["L3_within"].sum()) if n else 0
        kp = int(s_["L3_within_poisson"].sum()) if n else 0
        crit["serology"][c] = {"measure": SERO_MEASURE, "n_dates": n, "within": k, "met": bool(n and k >= (2.0 / 3.0) * n),
                               "within_poisson_formula": kp, "met_poisson_formula": bool(n and kp >= (2.0 / 3.0) * n),
                               "by_source": {src: {"n": int(len(g)), "within": int(g["L3_within"].sum())} for src, g in s_.groupby("source")}}
    crit["plausibility_met"] = bool(all(v["met"] for v in crit["plausibility"].values()))
    crit["serology_met"] = bool(sum(v["met"] for v in crit["serology"].values()) >= 2)
    crit["adequacy_met"] = bool(sum(v["not_significantly_worse_than_ARIMA_ADM"] for v in crit["adequacy"].values()) >= 2) if crit["adequacy"] else None
    crit["D1_met"] = bool(sum(v.get("significantly_better", False) for v in crit["D1"].values()) >= 2)
    crit["D2_met"] = bool(sum(v.get("significantly_better", False) for v in crit["D2"].values()) >= 2)
    if len(Icu):
        crit["ICU_secondary"] = {}
        for c in counties:
            j = Icu[Icu["county"] == c][["MAE14_H4_L2", "MAE14_ARIMA"]].dropna()
            d = (j["MAE14_H4_L2"] - j["MAE14_ARIMA"]).values; lo, hi = L2.block_boot_ci(d)
            crit["ICU_secondary"][c] = {"n": int(len(j)), "mean_H4_L3": float(j["MAE14_H4_L2"].mean()) if len(j) else None,
                                        "mean_ARIMA": float(j["MAE14_ARIMA"].mean()) if len(j) else None, "ci95_diff": [lo, hi]}
    crit["rules"] = {"plausibility": "L3 delay 1: beta at upper bound and rho at a bound in <= 10 % of origins, every county",
                     "serology": f"model ever infected ({SERO_MEASURE}) minus survey within [{SERO_LO}, {SERO_HI}] at >= 2/3 of dates, >= 2 of 3 counties; the protocol formula 1 - exp(-cumulative infections) is reported too",
                     "adequacy": "L3 not significantly worse than ARIMA_ADM in >= 2 of 3 counties"}
    (out / "criteria.json").write_text(json.dumps(crit, indent=2, default=str))
    log("\nplausibility:", json.dumps(crit["plausibility"])); log("serology:", json.dumps({k: {kk: v[kk] for kk in ("n_dates", "within", "met")} for k, v in crit["serology"].items()}))
    log("met: plausibility", crit["plausibility_met"], "| serology", crit["serology_met"], "| adequacy", crit["adequacy_met"],
        "| D1", crit["D1_met"], "| D2", crit["D2_met"])
    import scipy, statsmodels
    man = {"script": Path(__file__).name, "script_sha256": sha(__file__), "layer2_sha256": sha(Path(L2.__file__)),
           "layer1c_sha256": sha(Path(L1C.__file__)), "command": " ".join([sys.executable] + sys.argv), "args": vars(A),
           "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
           "inputs_sha256": {p: sha(p) for p in sorted(inputs)},
           "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".json")}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    log(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
