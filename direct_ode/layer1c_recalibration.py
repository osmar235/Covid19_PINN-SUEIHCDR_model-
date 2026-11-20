#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 15: real time recalibration of the hospital pathway (PHASE15_PROTOCOL.md). No neural network, no
transmission model, no torch.

At every Saturday origin t0 the H4 hospital model (Phase 14 structure, one set of 7 rates) is fitted to the
ward and ICU census of a trailing window, using only information a county had on t0:
  census known through t0 - 1; admissions (HHS weeks Sunday to Saturday) known through the week ending on
  t0 - 7 * delay (delay 1 primary, delay 0 = timely reporting sensitivity); admissions after the last known
  week filled by the projection rule (PERSIST or TREND), also inside the window; deaths not used;
  one fixed set of priors published before 2020-10-03; Phase 12 periods used only for reporting.
H4_ORACLE uses the observed admissions throughout (pathway test, never a forecast).
Comparators: ARIMA (Phase 10 definition) on ICU and ward census, 150 day lookback; NAIVE.

Structure H4 (one set of rates per origin):
    dW/dt = (1-q) A - W/L_w
    dC/dt = q A + p_wc W/L_w - C/L_c
    dP/dt = (1-p_cd) C/L_c - P/L_p
  observed ward = W + P, observed ICU = C. Convention: census(d + 1) = Phi census(d) + Gam A(d).

Usage (project root, .venv Python):  python layer1c_recalibration.py --out phase15_recalibration
"""
import argparse, hashlib, json, sys, platform, datetime as dt, math, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares

# ------------------------------------------------------------------ fixed design (protocol)
PERIODS = [("P1_FirstWave", "2020-03-29", "2020-09-30"), ("P2_Winter20", "2020-10-01", "2021-03-31"),
           ("P3_Delta", "2021-04-01", "2021-11-30"), ("P4_Omicron", "2021-12-01", "2022-04-30"),
           ("P5_BA2_BA5", "2022-05-01", "2022-12-31")]           # reporting only
FIRST_ORIGIN, LAST_ORIGIN, FIRST_ORIGIN_84 = "2020-10-03", "2022-12-03", "2020-10-31"
CENSUS_START, DATA_END = "2020-03-29", "2022-12-31"
HMAX = 28; HORIZONS = (7, 14, 28); PRIMARY_WINDOW = 56; PRIMARY_H = 14
# Real time priors (protocol section 4; values as in the protocol table), sd 0.5 on the log scale
PRIORS = {"mean_hosp_stay": 7.0, "p_icu": 0.32, "p_death_hosp": 0.17, "mean_icu_stay": 9.3, "p_death_icu": 0.416}
PRIOR_SD = 0.5
STAY = (1.0, 60.0); Q_MAX = 0.9; LOGIT = 8.0; NPAR = 7
TREND_CAP = math.log(2.0) / 7.0
ARIMA_LOOKBACK = 150
C1_THR = 0.20; BOOT_BLOCK = 4; BOOT_REPS = 2000; BOOT_SEED = 20261011
CITY = {"San Diego": "SanDiego", "Los Angeles": "LosAngeles", "San Francisco": "SanFrancisco"}
DAY = pd.Timedelta(days=1)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(x):
    return math.log(x / (1 - x))


def period_of(d):
    for name, a, b in PERIODS:
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return name
    return "outside"


# ------------------------------------------------------------------ data
def load_county(root, audit, hosp_file, county, inputs):
    """Returns census (daily ward, icu from CENSUS_START), weekly admissions per day (NaN = not usable,
    PHASE12 Addendum A3) and the oracle daily admissions (Phase 14 construction, whole series)."""
    p = Path(root) / audit / f"weekly_{CITY[county]}.csv"; inputs.add(str(p))
    wk = pd.read_csv(p, parse_dates=["week_start"]).set_index("week_start")["adm_per_day"].astype(float)
    days = pd.date_range(CENSUS_START, DATA_END, freq="D")
    hp = Path(root) / hosp_file; inputs.add(str(hp))
    h = pd.read_csv(hp)
    h = h[h["county"] == county].copy()
    h["date"] = pd.to_datetime(h["todays_date"], format="mixed")
    h = h.drop_duplicates("date", keep="last").set_index("date").sort_index()
    icu = h["icu_covid_confirmed_patients"].astype(float).reindex(days)
    ward = (h["hospitalized_covid_confirmed_patients"] - h["icu_covid_confirmed_patients"]).astype(float).reindex(days)
    census = pd.DataFrame({"ward": ward.values, "icu": icu.values}, index=days)
    # Oracle admissions: as Phase 14 (log interpolation over the whole series, daily from the week start)
    widx = pd.date_range(pd.Timestamp("2020-08-02") - pd.Timedelta(days=7), DATA_END, freq="7D")
    w = wk.reindex(widx)
    w = np.expm1(np.log1p(w).interpolate(limit_direction="both"))
    oracle = w.reindex(pd.date_range(widx[0], DATA_END, freq="D"), method="ffill")
    return census, wk, oracle


def causal_admissions(wk, t0, delay, rule, d_from, d_to):
    """Daily admissions on days d_from..d_to as known on t0. Weeks published: those ending on or before
    t0 - 7 * delay. Gaps between known weeks: log(1 + x) interpolation using known weeks only; days after
    the last known week: rule PERSIST (last week held) or TREND (log linear from the last two weeks)."""
    last_pub_end = t0 - pd.Timedelta(days=7 * delay)
    pub = wk[wk.index + pd.Timedelta(days=6) <= last_pub_end]
    known = pub.dropna()
    if known.empty:
        return None, {}
    Lw = known.index.max()
    pub = pub[pub.index <= Lw]
    filled = np.expm1(np.log1p(pub).interpolate(limit_direction="both"))
    E = Lw + pd.Timedelta(days=6)
    A_last = float(filled.loc[Lw])
    prev = Lw - pd.Timedelta(days=7)
    A_prev = float(filled.loc[prev]) if prev in filled.index and np.isfinite(filled.loc[prev]) else float("nan")
    r = 0.0
    if rule == "TREND" and np.isfinite(A_prev) and A_prev > 0 and A_last > 0:
        r = float(np.clip(math.log(A_last / A_prev) / 7.0, -TREND_CAP, TREND_CAP))
    days = pd.date_range(d_from, d_to, freq="D")
    out = np.empty(len(days))
    for i, d in enumerate(days):
        if d <= E:
            ws = d - pd.Timedelta(days=(d.dayofweek + 1) % 7)       # Sunday on or before d
            out[i] = float(filled.loc[ws]) if ws in filled.index else float("nan")
        else:
            out[i] = A_last * math.exp(r * (d - E).days) if rule == "TREND" else A_last
    info = {"last_known_week_start": str(Lw.date()), "last_known_week_end": str(E.date()),
            "A_last": A_last, "A_prev": A_prev, "trend_r": r}
    return out, info


# ------------------------------------------------------------------ model
def unpack(r):
    """raw: log L_w, logit p_wc, logit s_wd (share of non ICU ward exits that die), log L_c, logit p_cd,
    log L_p, logit q / Q_MAX."""
    L_w = math.exp(r[0]); p_wc = sig(r[1]); p_wd = (1.0 - p_wc) * sig(r[2]); L_c = math.exp(r[3])
    p_cd = sig(r[4]); L_p = math.exp(r[5]); q = Q_MAX * sig(r[6])
    return {"L_w": L_w, "p_wc": p_wc, "p_wd": p_wd, "L_c": L_c, "p_cd": p_cd, "L_p": L_p, "q": q}


def derived(p):
    p_icu = p["q"] + (1 - p["q"]) * p["p_wc"]
    return {"mean_hosp_stay": (1 - p["q"]) * p["L_w"] + p_icu * p["L_c"] + p_icu * (1 - p["p_cd"]) * p["L_p"],
            "p_icu": p_icu, "p_death_hosp": (1 - p["q"]) * p["p_wd"] + p_icu * p["p_cd"],
            "mean_icu_stay": p["L_c"], "p_death_icu": p["p_cd"]}


def transition(p):
    """Exact one day map for z = [W, C, P, D_cum] with constant admissions a: z1 = Phi z0 + Gam a."""
    M = np.zeros((5, 5)); kw, kc, kp = 1 / p["L_w"], 1 / p["L_c"], 1 / p["L_p"]
    M[0, 0] = -kw
    M[1, 0] = p["p_wc"] * kw; M[1, 1] = -kc
    M[2, 1] = (1 - p["p_cd"]) * kc; M[2, 2] = -kp
    M[3, 0] = p["p_wd"] * kw; M[3, 1] = p["p_cd"] * kc
    M[0, 4] = 1 - p["q"]; M[1, 4] = p["q"]
    E = expm(M)
    return E[:4, :4], E[:4, 4]


def steady_fw(p):
    """Steady state ward share W / (W + P) per unit admission (Phase 14 steady_split)."""
    W = (1 - p["q"]) * p["L_w"]
    C = (p["q"] + p["p_wc"] * (1 - p["q"])) * p["L_c"]
    P = (1 - p["p_cd"]) * C / p["L_c"] * p["L_p"]
    return W / max(W + P, 1e-12)


def simulate(p, adm, ward0, icu0):
    """Census trajectory on the n window days (index 0 = first window day, set to the observed census)."""
    Phi, Gam = transition(p); fw = steady_fw(p)
    n = len(adm); Z = np.zeros((n, 4)); Z[0] = [fw * ward0, icu0, (1 - fw) * ward0, 0.0]
    for d in range(n - 1):
        Z[d + 1] = Phi @ Z[d] + Gam * adm[d]
    return Z


def residuals(theta, adm, ward, icu, use_prior=True):
    p = unpack(theta)
    Z = simulate(p, adm, ward[0], icu[0])
    rw = np.log1p(np.clip(Z[:, 0] + Z[:, 2], 0, None)) - np.log1p(ward)
    rc = np.log1p(np.clip(Z[:, 1], 0, None)) - np.log1p(icu)
    parts = [rw, rc]
    if use_prior:
        dq = derived(p)
        parts.append(np.array([(math.log(dq[k]) - math.log(v)) / PRIOR_SD for k, v in PRIORS.items()]))
    r = np.concatenate(parts)
    return np.where(np.isfinite(r), r, 0.0)


def bounds():
    lo = [math.log(STAY[0]), -LOGIT, -LOGIT, math.log(STAY[0]), -LOGIT, math.log(STAY[0]), -LOGIT]
    hi = [math.log(STAY[1]), LOGIT, LOGIT, math.log(STAY[1]), LOGIT, math.log(STAY[1]), LOGIT]
    return np.array(lo), np.array(hi)


def start_point(s, seed_key):
    """Start 0: Phase 14 fixed start (ward stay 6 d, 12 % to ICU, 6 % of other ward exits die, ICU stay
    9 d, 35 % ICU deaths, post ICU stay 5 d, 5 % direct ICU). Starts 1, 2: uniform in bounds, seeded."""
    lo, hi = bounds()
    if s == 0:
        x = np.array([math.log(6.0), logit(0.12), logit(0.06), math.log(9.0), logit(0.35), math.log(5.0),
                      logit(0.05 / Q_MAX)])
    else:
        rng = np.random.default_rng([20261011, s] + list(seed_key))
        x = lo + (hi - lo) * rng.uniform(0.05, 0.95, size=len(lo))
    return np.clip(x, lo + 1e-9, hi - 1e-9)


def fit_window(adm, ward, icu, starts, max_nfev, seed_key):
    lo, hi = bounds(); best, costs = None, []
    for s in range(starts):
        r = least_squares(residuals, start_point(s, seed_key), bounds=(lo, hi), method="trf",
                          args=(adm, ward, icu), max_nfev=max_nfev, x_scale="jac")
        costs.append(float(r.cost))
        if best is None or r.cost < best.cost:
            best = r
    return best, costs


def project(p, z_last, adm_future):
    """z_last = state on the last known census day L; adm_future[k-1] = admissions on day L + k - 1."""
    Phi, Gam = transition(p); z = z_last.copy(); ward, icu = [], []
    for a in adm_future:
        z = Phi @ z + Gam * a; ward.append(z[0] + z[2]); icu.append(z[1])
    return np.array(ward), np.array(icu)


# ------------------------------------------------------------------ ARIMA (Phase 10 definition)
def small_auto_arima(y):
    import statsmodels.api as sm
    best = None
    for p_ in (0, 1):
        for d_ in (0, 1):
            for q_ in (0, 1):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        m = sm.tsa.SARIMAX(y, order=(p_, d_, q_), enforce_stationarity=False,
                                           enforce_invertibility=False)
                        r = m.fit(disp=False, maxiter=50, method="lbfgs")
                        aic = getattr(r, "aic", np.inf)
                        if best is None or aic < best[0]:
                            best = (aic, r)
                except Exception:
                    pass
    return best[1] if best else None


def census_arima(y_hist, steps):
    y = np.asarray(y_hist, float); y = y[np.isfinite(y)]
    if y.size < 14:
        return np.full(steps, float(y[-1]) if y.size else np.nan)
    if np.nanvar(y) < 1e-9:
        return np.full(steps, float(y[-1]))
    model = small_auto_arima(y)
    if model is None:
        return np.full(steps, float(y[-1]))
    try:
        pm = np.asarray(model.get_forecast(steps=steps).predicted_mean, float).reshape(-1)
    except Exception:
        return np.full(steps, float(y[-1]))
    pm[~np.isfinite(pm)] = float(y[-1])
    return np.clip(pm[:steps], 0.0, None)


# ------------------------------------------------------------------ metrics
def mae(pred, obs, h):
    p, o = np.asarray(pred[:h], float), np.asarray(obs[:h], float)
    ok = np.isfinite(p) & np.isfinite(o)
    if not ok.any():
        return float("nan"), float("nan"), 0
    m = float(np.mean(np.abs(p[ok] - o[ok])))
    return m, m / max(float(np.mean(o[ok])), 1e-9), int(ok.sum())


def block_boot_ci(diff, block=BOOT_BLOCK, reps=BOOT_REPS, seed=BOOT_SEED):
    """Moving block bootstrap of the mean of diff (ordered by origin); percentile 95 % interval."""
    x = np.asarray(diff, float); n = len(x)
    if n < block + 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed); nb = int(math.ceil(n / block)); means = np.empty(reps)
    for b in range(reps):
        st = rng.integers(0, n - block + 1, size=nb)
        idx = (st[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        means[b] = x[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ one origin
ARMS = [("H4_ORACLE", None, "ORACLE"), ("H4_PERSIST", 1, "PERSIST"), ("H4_TREND", 1, "TREND"),
        ("H4_D0", 0, "PERSIST")]      # with delay 0 the window needs no fill: one fit serves PERSIST and TREND


def run_origin(census, wk, oracle, t0, windows, starts, max_nfev, seed_base, with_arima=True):
    """All forecasts issued on t0. Returns (daily rows, param rows). Only data known on t0 are passed to
    the non oracle arms (census through t0 - 1; admissions by causal_admissions)."""
    L = t0 - DAY
    target_days = pd.date_range(L + DAY, L + pd.Timedelta(days=HMAX), freq="D")
    obs_ward = census["ward"].reindex(target_days).values; obs_icu = census["icu"].reindex(target_days).values
    known_census = census[census.index <= L]                      # the only census the forecaster sees
    daily, prows = [], []

    def add(model, delay, window, wp, ip):
        for k in range(HMAX):
            daily.append({"origin": str(t0.date()), "model": model, "delay": delay, "window": window, "k": k + 1,
                          "date": str(target_days[k].date()), "ward_pred": wp[k], "icu_pred": ip[k],
                          "ward_obs": obs_ward[k], "icu_obs": obs_icu[k]})

    # last known census day with both values (normally L)
    kc = known_census.dropna()
    if kc.empty:
        return daily, prows
    Lc = kc.index.max()
    for W in windows:
        if W == 84 and t0 < pd.Timestamp(FIRST_ORIGIN_84):
            continue
        win_days = pd.date_range(L - pd.Timedelta(days=W - 1), L, freq="D")
        cw = known_census.reindex(win_days)
        first_ok = cw.dropna().index.min()
        if pd.isna(first_ok):
            continue
        cw = cw[cw.index >= first_ok]; wdays = cw.index
        for arm, delay, rule in ARMS:
            if arm == "H4_ORACLE":
                adm_w = oracle.reindex(wdays).values
                adm_f = oracle.reindex(pd.date_range(Lc, Lc + pd.Timedelta(days=(L - Lc).days + HMAX - 1))).values
                info = {}
            else:
                full, info = causal_admissions(wk, t0, delay, rule, wdays[0], Lc + pd.Timedelta(days=(L - Lc).days + HMAX - 1))
                if full is None:
                    continue
                adm_w = full[:len(wdays)]
                adm_f = full[(Lc - wdays[0]).days:]
            if not np.all(np.isfinite(adm_w)) or not np.all(np.isfinite(adm_f)):
                continue
            seed_key = (seed_base, int(t0.toordinal()), W, ["H4_ORACLE", "H4_PERSIST", "H4_TREND", "H4_D0"].index(arm))
            res, costs = fit_window(adm_w, cw["ward"].values, cw["icu"].values, starts, max_nfev, seed_key)
            p = unpack(res.x)
            Z = simulate(p, adm_w, cw["ward"].values[0], cw["icu"].values[0])
            j = (Lc - wdays[0]).days
            wp_ = Z[j, 0] + Z[j, 2]; fw = Z[j, 0] / wp_ if wp_ > 0 else 1.0
            z_last = np.array([fw * kc.loc[Lc, "ward"], kc.loc[Lc, "icu"], (1 - fw) * kc.loc[Lc, "ward"], 0.0])
            dq = derived(p)
            prow = {"origin": str(t0.date()), "arm": arm, "window": W, **{k: v for k, v in p.items()},
                    **{f"{k}": v for k, v in dq.items()},
                    **{f"{k}_z": (math.log(dq[k]) - math.log(v)) / PRIOR_SD for k, v in PRIORS.items()},
                    "cost": float(res.cost), "cost_starts": json.dumps([round(c, 5) for c in costs]),
                    "nfev": int(res.nfev), "window_days_used": len(wdays), "last_census_day": str(Lc.date()), **info}
            prows.append(prow)
            # project from Lc through L + HMAX, keep the HMAX target days
            n_steps = (L - Lc).days + HMAX
            wp, ip = project(p, z_last, adm_f[:n_steps])
            wp, ip = wp[-HMAX:], ip[-HMAX:]
            if arm == "H4_D0":
                add("H4_PERSIST", 0, W, wp, ip)
                # TREND under delay 0: same fit, admissions after the last known week from the trend rule
                full_t, _ = causal_admissions(wk, t0, 0, "TREND", wdays[0], Lc + pd.Timedelta(days=n_steps - 1))
                adm_ft = full_t[(Lc - wdays[0]).days:]
                wpt, ipt = project(p, z_last, adm_ft[:n_steps])
                add("H4_TREND", 0, W, wpt[-HMAX:], ipt[-HMAX:])
            else:
                add(arm, delay if delay is not None else -1, W, wp, ip)
    # comparators, same information (census through t0 - 1)
    hist = known_census[known_census.index > L - pd.Timedelta(days=ARIMA_LOOKBACK)]
    nv_w = np.full(HMAX, float(kc.loc[Lc, "ward"])); nv_i = np.full(HMAX, float(kc.loc[Lc, "icu"]))
    add("NAIVE", -1, -1, nv_w, nv_i)
    if with_arima:
        gap = (L - Lc).days
        aw = census_arima(hist["ward"].values, HMAX + gap)[-HMAX:]
        ai = census_arima(hist["icu"].values, HMAX + gap)[-HMAX:]
        add("ARIMA", -1, -1, aw, ai)
    return daily, prows


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--audit", default="phase15_admissions", help="folder with weekly_<City>.csv (corrected series, Addendum B)")
    ap.add_argument("--hosp-file", default="covid19hospitalbycounty.csv")
    ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
    ap.add_argument("--windows", default="56,28,84")
    ap.add_argument("--first-origin", default=FIRST_ORIGIN)
    ap.add_argument("--last-origin", default=LAST_ORIGIN)
    ap.add_argument("--max-origins", type=int, default=0, help="smoke tests only (0 = all)")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--max-nfev", type=int, default=300)
    ap.add_argument("--skip-arima", action="store_true", help="smoke tests only")
    ap.add_argument("--out", default="phase15_recalibration")
    ap.add_argument("--overwrite", action="store_true")
    A = ap.parse_args(argv)
    out = Path(A.out)
    if out.exists() and any(out.iterdir()) and not A.overwrite:
        raise SystemExit(f"{out} is not empty; use a new --out or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    LOG = open(out / "run_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    inputs = set()
    counties = [c.strip() for c in A.counties.split(",") if c.strip()]
    windows = [int(x) for x in A.windows.split(",") if x.strip()]
    origins = list(pd.date_range(A.first_origin, A.last_origin, freq="7D"))
    assert all(o.dayofweek == 5 for o in origins), "origins must be Saturdays"
    if A.max_origins:
        origins = origins[:A.max_origins]
    log(f"Phase 15 | counties {counties} | windows {windows} | origins {len(origins)} "
        f"({origins[0].date()} to {origins[-1].date()}) | starts {A.starts} | max_nfev {A.max_nfev} | "
        f"arima {not A.skip_arima}")
    all_daily, all_par = [], []
    for ci, county in enumerate(counties):
        census, wk, oracle = load_county(A.root, A.audit, A.hosp_file, county, inputs)
        t_start = time.time()
        for oi, t0 in enumerate(origins):
            d, pr = run_origin(census, wk, oracle, t0, windows, A.starts, A.max_nfev, ci, not A.skip_arima)
            for r in d:
                r["county"] = county; r["period"] = period_of(t0)
            for r in pr:
                r["county"] = county; r["period"] = period_of(t0)
            all_daily += d; all_par += pr
            if (oi + 1) % 10 == 0 or oi + 1 == len(origins):
                log(f"  {county}: {oi + 1}/{len(origins)} origins ({time.time() - t_start:.0f} s)")
    D = pd.DataFrame(all_daily); P = pd.DataFrame(all_par)
    D.to_csv(out / "forecasts_daily.csv", index=False); P.to_csv(out / "params.csv", index=False)

    # per origin metrics
    rows = []
    for key, g in D.groupby(["county", "origin", "period", "model", "delay", "window"], sort=False):
        g = g.sort_values("k")
        for tgt in ("icu", "ward"):
            for h in HORIZONS:
                m, rm, n = mae(g[f"{tgt}_pred"].values, g[f"{tgt}_obs"].values, h)
                rows.append(dict(zip(["county", "origin", "period", "model", "delay", "window"], key),
                                 target=tgt, h=h, MAE=m, relMAE=rm, n_days=n))
    M = pd.DataFrame(rows); M.to_csv(out / "metrics_origin.csv", index=False)

    # summaries (with win fractions against ARIMA and NAIVE on the same origin, target and horizon)
    ref = M[M["model"].isin(["ARIMA", "NAIVE"])].pivot_table(index=["county", "origin", "target", "h"],
                                                             columns="model", values="MAE")
    Mj = M.join(ref, on=["county", "origin", "target", "h"])

    def summarize(df, by):
        out_rows = []
        for key, g in df.groupby(by):
            r = dict(zip(by, key)); r["n_origins"] = int(g["MAE"].notna().sum())
            r["mean_MAE"] = g["MAE"].mean(); r["mean_relMAE"] = g["relMAE"].mean(); r["median_relMAE"] = g["relMAE"].median()
            for c in ("ARIMA", "NAIVE"):
                if c in g:
                    ok = g["MAE"].notna() & g[c].notna()
                    r[f"win_vs_{c}"] = float((g.loc[ok, "MAE"] < g.loc[ok, c]).mean()) if ok.any() else float("nan")
            out_rows.append(r)
        return pd.DataFrame(out_rows)

    by = ["county", "model", "delay", "window", "target", "h"]
    summarize(Mj, by).to_csv(out / "summary.csv", index=False)
    summarize(Mj, ["period"] + by).to_csv(out / "summary_by_period.csv", index=False)
    # window sensitivity on common origins (84 day window exists from FIRST_ORIGIN_84)
    common = Mj[pd.to_datetime(Mj["origin"]) >= pd.Timestamp(FIRST_ORIGIN_84)]
    summarize(common, by).to_csv(out / "summary_common_origins_from_2020-10-31.csv", index=False)

    # criteria
    crit = {"C1": {}, "C2": {}}
    for county in counties:
        o = M[(M["county"] == county) & (M["model"] == "H4_ORACLE") & (M["window"] == PRIMARY_WINDOW) & (M["h"] == 28)]
        med_i = float(o[o["target"] == "icu"]["relMAE"].median()); med_w = float(o[o["target"] == "ward"]["relMAE"].median())
        crit["C1"][county] = {"median_relMAE_icu_28d": med_i, "median_relMAE_ward_28d": med_w,
                              "n_origins": int(o[o["target"] == "icu"]["relMAE"].notna().sum()),
                              "met": bool(med_i <= C1_THR and med_w <= C1_THR)}
        sel = M[(M["county"] == county) & (M["target"] == "icu") & (M["h"] == PRIMARY_H)]
        a = sel[(sel["model"] == "H4_PERSIST") & (sel["delay"] == 1) & (sel["window"] == PRIMARY_WINDOW)].set_index("origin")["MAE"]
        b = sel[sel["model"] == "ARIMA"].set_index("origin")["MAE"]
        j = pd.concat([a.rename("H4"), b.rename("ARIMA")], axis=1).dropna().sort_index()
        diff = (j["H4"] - j["ARIMA"]).values
        lo, hi = block_boot_ci(diff)
        crit["C2"][county] = {"n_origins": int(len(j)), "mean_MAE_H4_PERSIST": float(j["H4"].mean()) if len(j) else None,
                              "mean_MAE_ARIMA": float(j["ARIMA"].mean()) if len(j) else None,
                              "mean_diff": float(diff.mean()) if len(diff) else None, "ci95": [lo, hi],
                              "met": bool(len(diff) and diff.mean() < 0 and np.isfinite(hi) and hi < 0)}
    crit["C1_met_all_counties"] = bool(all(v["met"] for v in crit["C1"].values()))
    crit["C2_met"] = bool(sum(v["met"] for v in crit["C2"].values()) >= 2)
    crit["rules"] = {"C1": f"H4_ORACLE window {PRIMARY_WINDOW}: median 28 day relMAE <= {C1_THR} for ICU and ward in every county",
                     "C2": f"H4_PERSIST delay 1 window {PRIMARY_WINDOW} vs ARIMA, ICU MAE at {PRIMARY_H} days: mean lower and "
                           f"block bootstrap 95 % interval of the mean paired difference below 0, in >= 2 of 3 counties "
                           f"(block {BOOT_BLOCK}, {BOOT_REPS} replicates, seed {BOOT_SEED})"}
    crit["priors"] = {"values": PRIORS, "sd_log": PRIOR_SD}
    (out / "criteria.json").write_text(json.dumps(crit, indent=2))
    log("\nC1:", json.dumps(crit["C1"]))
    log("C2:", json.dumps(crit["C2"]))
    log("C1 met in all counties:", crit["C1_met_all_counties"], "| C2 met:", crit["C2_met"])

    import scipy, statsmodels
    man = {"script": Path(__file__).name, "script_sha256": sha(__file__), "command": " ".join([sys.executable] + sys.argv),
           "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0],
           "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
           "inputs_sha256": {p: sha(p) for p in sorted(inputs)},
           "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".json")}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    log(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
