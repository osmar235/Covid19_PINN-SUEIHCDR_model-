#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 16: Layer 2, forecasting weekly hospital admissions in real time (PHASE16_PROTOCOL.md with Addenda A
and B). No neural network, no torch.

Model L2 (transmission and immunity part of SUEIHCDR; population fractions; per day):
    dS/dt   = -beta S I - v S - a_b s S + omega_v U_v + omega_b (1 - s) U_b + eta R
    dU_v/dt =  v S - omega_v U_v                       (vaccination)
    dU_b/dt =  a_b s S - omega_b (1 - s) U_b           (protective behaviour; s = paper's deaths signal)
    dE/dt   =  beta S I - sigma E
    dI/dt   =  sigma E - delta I
    dR/dt   =  delta I - eta R
    reported cases = rho sigma E N;  admissions = ihr delta I N
Moving horizon estimation at every Saturday origin t0 on the 56 days ending t0 - 1; S, U_v, U_b, R carried
from the previous origin's fit, E and I carried with an estimated multiplicative correction.
Real time rule: cases and report date deaths through t0 - 1; admissions through the week ending
t0 - 7 * delay; vaccination doses through t0 - 7; fixed values used only from the date they became public.

Usage (project root, .venv Python):  python layer2_admissions.py --out phase16_layer2
"""
import argparse, hashlib, json, sys, platform, datetime as dt, math, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ------------------------------------------------------------------ fixed design (protocol + Addendum B)
FIRST_ORIGIN, LAST_ORIGIN = "2020-10-03", "2022-12-03"
WINDOW = 56; HWEEKS = 4; PRIMARY_H = (1, 2, 3)
PERIODS = [("P1_FirstWave", "2020-03-29", "2020-09-30"), ("P2_Winter20", "2020-10-01", "2021-03-31"),
           ("P3_Delta", "2021-04-01", "2021-11-30"), ("P4_Omicron", "2021-12-01", "2022-04-30"),
           ("P5_BA2_BA5", "2022-05-01", "2022-12-31")]           # reporting only
FIPS = {"San Diego": 6073, "Los Angeles": 6037, "San Francisco": 6075}
CITY = {"San Diego": "SanDiego", "Los Angeles": "LosAngeles", "San Francisco": "SanFrancisco"}
SIGMA = 1 / 5.1; DELTA = 1 / 7.0
# (public from, value); before the first date the first value applies
VE_SCHED = [("2020-12-10", 0.95), ("2021-03-29", 0.90)]
OMEGA_V_SCHED = [("2020-01-01", 0.0), ("2021-10-04", 1 / 191.0)]
# Case ascertainment prior centre for L2 and L3 (Phase 16 Addendum B).
# Correction C1 (2026-10-24, before submission): the value used from 2020-11-25 was recorded as 1/4.6 and attributed
# to Reese et al., Clin Infect Dis 2021;72(12):e1010-e1017 (online 2020-11-25), but that study reports that 1 in 7.7
# infections were identified and reported through September 2020. C1 uses 1/7.7. "protocol_v1" reproduces the
# outputs of 2026-10-12 (phase16_layer2, phase17_layer3).
RHO_PRIOR_SCHEDULES = {"c1": [("2020-07-21", 1 / 12.0), ("2020-11-25", 1 / 7.7)],
                       "protocol_v1": [("2020-07-21", 1 / 12.0), ("2020-11-25", 1 / 4.6)]}
RHO_PRIOR_SCHED = RHO_PRIOR_SCHEDULES["c1"]
IHR_PRIOR = 0.026; PRIOR_SD_RHO = 0.7; PRIOR_SD_IHR = 0.7
AB_BOUNDS = (0.0, 0.003)                       # paper ALPH_BOUNDS
OMB_BOUNDS = (1 / 180.0, 1 / 30.0); OMB_PRIOR = 1 / 73.5; OMB_SD = 0.5   # paper OMEGA_BOUNDS
EI_SD = 0.5; BETA_RW_SD = 0.1; BETA_BOUNDS = (0.01, 3.0)
KNOTS = [0, 7, 14, 21, 28, 35, 42, 49, 55]
SUBSTEPS = 2
ALPH_HI = 0.003; KAPPA = 0.08; LAM = 0.995      # paper call of bayesian_alpha_series
ARIMA_LOOKBACK = 150; CASELAG_MAX = 14
BOOT_BLOCK = 4; BOOT_REPS = 2000; BOOT_SEED = 20261012
DAY = pd.Timedelta(days=1)
# chains: (name, delay, behaviour on, eta)
CHAINS = [("L2", 1, True, 0.0), ("L2", 0, True, 0.0), ("L2_NOBEHAV", 1, False, 0.0), ("L2_ETA365", 1, True, 1 / 365.0)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def sched(table, t):
    v = table[0][1]
    for d, val in table:
        if t >= pd.Timestamp(d):
            v = val
    return v


def period_of(d):
    for name, a, b in PERIODS:
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return name
    return "outside"


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


# ------------------------------------------------------------------ paper behaviour signal (core_phase10.py)
def weekly_avg_np(x, k=7):
    x = np.asarray(x, float)
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy() if x.size else x


def daily_from_cum_np(cum):
    cum = np.asarray(cum, float)
    return np.clip(np.diff(np.r_[cum[0], cum]), 0, None) if cum.size else cum


def behavior_signal_from_deaths(deaths_cum):
    dw = weekly_avg_np(daily_from_cum_np(deaths_cum))
    d_dw = np.diff(np.r_[dw[:1], dw])
    s = (d_dw - np.median(d_dw)) / (np.median(np.abs(d_dw - np.median(d_dw))) + 1e-9)
    return np.clip(s, -4.0, 4.0)


def behaviour_s(deaths_cum):
    """s(t) in [0, 1] = (alpha - lo) / (hi - lo) of the paper's bayesian_alpha_series (kappa 0.08, lambda 0.995)."""
    z = behavior_signal_from_deaths(deaths_cum)
    a, b = 2.0, 2.0; out = np.zeros_like(z, float)
    for t, v in enumerate(z):
        a, b = LAM * a, LAM * b
        if v >= 0:
            a += KAPPA * float(v)
        else:
            b += KAPPA * float(-v)
        a = max(a, 0.1); b = max(b, 0.1); out[t] = a / (a + b)
    return out


# ------------------------------------------------------------------ data
def load_county(root, county, adm_dir, vax_file, inputs):
    R = Path(root)
    pop = pd.read_csv(R / "covid_county_population_usafacts.csv", encoding="utf-8-sig"); inputs.add(str(R / "covid_county_population_usafacts.csv"))
    N = float(pop.loc[pop["countyFIPS"] == FIPS[county], "population"].iloc[0])
    out = {}
    for name, f in (("cases", "covid_confirmed_usafacts.csv"), ("deaths", "covid_deaths_usafacts.csv")):
        w = pd.read_csv(R / f, encoding="utf-8-sig"); inputs.add(str(R / f))
        row = w[w["countyFIPS"] == FIPS[county]].iloc[0]
        cols = [c for c in w.columns if c[:2] == "20"]
        out[name] = pd.Series(pd.to_numeric(row[cols].values, errors="coerce"), index=pd.to_datetime(cols)).astype(float)
    cum_cases, cum_deaths = out["cases"], out["deaths"]
    days = pd.date_range(cum_cases.index[0], "2023-01-31", freq="D")
    cum_cases = cum_cases.reindex(days).ffill(); cum_deaths = cum_deaths.reindex(days).ffill()
    vp = Path(vax_file); inputs.add(str(vp))
    v = pd.read_csv(vp); v = v[v["county"] == county].copy(); v["date"] = pd.to_datetime(v["administered_date"])
    v = v.set_index("date").sort_index()
    cols = [c for c in ("fully_vaccinated", "booster_recip_count", "bivalent_booster_recip_count") if c in v.columns]
    doses = v[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).reindex(days).fillna(0.0)
    ap = Path(root) / adm_dir / f"weekly_{CITY[county]}.csv"; inputs.add(str(ap))
    wk = pd.read_csv(ap, parse_dates=["week_start"]).set_index("week_start")["adm_per_day"].astype(float)
    return {"N": N, "cum_cases": cum_cases, "cum_deaths": cum_deaths, "doses": doses / N, "wk": wk}


def known_inputs(D, t0, delay):
    """Everything the forecaster knows on t0 (protocol section 1)."""
    L = t0 - DAY
    cc = D["cum_cases"][D["cum_cases"].index <= L]
    cases7 = pd.Series(weekly_avg_np(daily_from_cum_np(cc.values)), index=cc.index)
    cd = D["cum_deaths"][D["cum_deaths"].index <= L]
    s = pd.Series(behaviour_s(cd.values), index=cd.index)
    doses = D["doses"][D["doses"].index <= t0 - pd.Timedelta(days=7)]
    last_pub_end = t0 - pd.Timedelta(days=7 * delay)
    wk = D["wk"][D["wk"].index + pd.Timedelta(days=6) <= last_pub_end]
    return {"cases7": cases7, "cum_cases": cc, "s": s, "doses": doses, "wk": wk, "L": L,
            "last_pub_end": last_pub_end}


# ------------------------------------------------------------------ model
def rhs(x, beta, v, ab_s, omb_eff, omv, eta):
    S, Uv, Ub, E, I, R = x
    inf = beta * S * I
    return (-inf - v * S - ab_s * S + omv * Uv + omb_eff * Ub + eta * R,
            v * S - omv * Uv,
            ab_s * S - omb_eff * Ub,
            inf - SIGMA * E,
            SIGMA * E - DELTA * I,
            DELTA * I - eta * R)


def simulate(x0, beta_d, dose_d, s_d, ab, omb, omv, ve, eta):
    """Daily RK4 (SUBSTEPS per day). Returns states at the start of each day (n+1 x 6) and the daily
    integrated fluxes sigma E (new infectious) and delta I (leaving I)."""
    n = len(beta_d); X = np.empty((n + 1, 6)); fE = np.empty(n); fI = np.empty(n)
    x = list(x0); X[0] = x; h = 1.0 / SUBSTEPS
    for d in range(n):
        b = beta_d[d]; s = s_d[d]; ab_s = ab * s; omb_eff = omb * (1.0 - s)
        accE = 0.0; accI = 0.0
        for _ in range(SUBSTEPS):
            S, Uv, Ub, E, I, R = x
            nv = max(S + E + I + R, 1e-12)
            v = min(ve * dose_d[d] / nv, 1.0)
            k1 = rhs(x, b, v, ab_s, omb_eff, omv, eta)
            x2 = [x[i] + 0.5 * h * k1[i] for i in range(6)]; k2 = rhs(x2, b, v, ab_s, omb_eff, omv, eta)
            x3 = [x[i] + 0.5 * h * k2[i] for i in range(6)]; k3 = rhs(x3, b, v, ab_s, omb_eff, omv, eta)
            x4 = [x[i] + h * k3[i] for i in range(6)]; k4 = rhs(x4, b, v, ab_s, omb_eff, omv, eta)
            Eavg = (x[3] + 2 * x2[3] + 2 * x3[3] + x4[3]) / 6.0; Iavg = (x[4] + 2 * x2[4] + 2 * x3[4] + x4[4]) / 6.0
            accE += SIGMA * Eavg * h; accI += DELTA * Iavg * h
            x = [max(x[i] + h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0, 0.0) for i in range(6)]
        X[d + 1] = x; fE[d] = accE; fI[d] = accI
    return X, fE, fI


def beta_series(logk, n):
    return np.exp(np.interp(np.arange(n), KNOTS, logk))      # held at the last knot beyond day 55


class Problem:
    """One origin of one chain: data on the window, the carried state, and the parameter layout
    [log beta knots (9), log rho, log ihr, logit a_b, logit omega_b, log mult E, log mult I]."""

    def __init__(self, K, D, t0, x_carry, behav, eta, n_extra):
        self.t0 = t0; self.N = D["N"]; self.behav = behav; self.eta = eta
        self.w0 = t0 - pd.Timedelta(days=WINDOW)
        self.n = WINDOW + n_extra                                 # days simulated from w0
        days = pd.date_range(self.w0, periods=self.n, freq="D"); self.days = days
        s_known = K["s"]; s_last = float(s_known.iloc[-1])
        self.s_d = np.array([float(s_known.get(d, s_last)) if d <= K["L"] else s_last for d in days])
        dk = K["doses"]; d_last = float(dk.iloc[-7:].mean()) if len(dk) else 0.0
        self.dose_d = np.array([float(dk.get(d, 0.0)) if d <= dk.index.max() else d_last for d in days]) \
            if len(dk) else np.zeros(self.n)
        self.ve = sched(VE_SCHED, t0); self.omv = sched(OMEGA_V_SCHED, t0)
        self.rho_prior = sched(RHO_PRIOR_SCHED, t0)
        c7 = K["cases7"].reindex(days[:WINDOW]).values             # window days w0 .. t0 - 1
        self.case_obs = c7; self.case_ok = np.zeros(WINDOW, bool); self.case_ok[6:] = np.isfinite(c7[6:])
        wk = K["wk"].dropna(); self.weeks = [(ws, float(v)) for ws, v in wk.items()
                                            if ws >= self.w0 and ws + pd.Timedelta(days=6) < self.w0 + pd.Timedelta(days=self.n)]
        self.x_carry = np.asarray(x_carry, float)
        last_obs = max([WINDOW] + [(ws - self.w0).days + 7 for ws, _ in self.weeks])
        self.n_fit = min(last_obs, self.n)                        # days needed for the residuals

    def x0(self, th):
        x = self.x_carry.copy(); x[3] *= math.exp(th[13]); x[4] *= math.exp(th[14])
        x[0] = max(1.0 - (x[1] + x[2] + x[3] + x[4] + x[5]), 1e-6)
        return x

    def unpack(self, th):
        ab = AB_BOUNDS[0] + (AB_BOUNDS[1] - AB_BOUNDS[0]) * sig(th[11]) if self.behav else 0.0
        omb = OMB_BOUNDS[0] + (OMB_BOUNDS[1] - OMB_BOUNDS[0]) * sig(th[12])
        return {"logk": th[:9], "rho": math.exp(th[9]), "ihr": math.exp(th[10]), "a_b": ab, "omega_b": omb,
                "mult_E": math.exp(th[13]), "mult_I": math.exp(th[14])}

    def run(self, th, n=None):
        p = self.unpack(th); n = n or self.n
        X, fE, fI = simulate(self.x0(th), beta_series(p["logk"], n), self.dose_d[:n], self.s_d[:n],
                             p["a_b"], p["omega_b"], self.omv, self.ve, self.eta)
        return p, X, fE * self.N * p["rho"], fI * self.N * p["ihr"]

    def residuals(self, th):
        p, X, cases, adm = self.run(th, self.n_fit)
        c7 = weekly_avg_np(cases[:WINDOW])
        rc = (np.log1p(np.clip(c7, 0, None)) - np.log1p(np.where(self.case_ok, self.case_obs, 0.0)))[self.case_ok]
        ra = []
        for ws, v in self.weeks:
            i = (ws - self.w0).days
            ra.append(math.sqrt(7.0) * (math.log1p(max(adm[i:i + 7].mean(), 0.0)) - math.log1p(v)))
        lk = np.asarray(p["logk"]); rw = np.diff(lk, 2) / BETA_RW_SD
        pri = [(th[9] - math.log(self.rho_prior)) / PRIOR_SD_RHO, (th[10] - math.log(IHR_PRIOR)) / PRIOR_SD_IHR,
               (math.log(p["omega_b"]) - math.log(OMB_PRIOR)) / OMB_SD, th[13] / EI_SD, th[14] / EI_SD]
        r = np.concatenate([rc, np.array(ra), rw, np.array(pri)])
        return np.where(np.isfinite(r), r, 0.0)

    def bounds(self):
        lo = [math.log(BETA_BOUNDS[0])] * 9 + [math.log(0.005), math.log(1e-4), -8.0, -8.0, -3.0, -3.0]
        hi = [math.log(BETA_BOUNDS[1])] * 9 + [math.log(1.0), math.log(0.5), 8.0, 8.0, 3.0, 3.0]
        return np.array(lo), np.array(hi)

    def start(self, s, prev_theta, seed_key):
        lo, hi = self.bounds()
        if prev_theta is not None and s == 0:
            x = np.array(prev_theta, float); x[:9] = x[8]; x[13] = 0.0; x[14] = 0.0
        else:
            S0 = max(self.x_carry[0], 0.05)
            x = np.array([math.log(DELTA / S0)] * 9 + [math.log(self.rho_prior), math.log(IHR_PRIOR), 0.0,
                          math.log((OMB_PRIOR - OMB_BOUNDS[0]) / (OMB_BOUNDS[1] - OMB_PRIOR)), 0.0, 0.0])
            if s > 0:
                rng = np.random.default_rng([BOOT_SEED, s] + list(seed_key))
                x = x + rng.normal(0, 0.5, size=len(x))
        return np.clip(x, lo + 1e-9, hi - 1e-9)


def initial_state(D, w0, rho_prior):
    """First window (Addendum B): R from cumulative reported cases / rho prior; E and I from the 7 day mean
    of cases at the window start / rho prior; U_v = U_b = 0."""
    N = D["N"]; cc = D["cum_cases"]
    c7 = weekly_avg_np(daily_from_cum_np(cc[cc.index <= w0].values))[-1]
    inf = c7 / rho_prior / N
    R = float(cc.loc[w0]) / rho_prior / N
    E = inf / SIGMA; I = inf / DELTA
    return np.array([1.0 - R - E - I, 0.0, 0.0, E, I, R])


# ------------------------------------------------------------------ comparators
def small_auto_arima(y):
    import statsmodels.api as sm
    best = None
    for p_ in (0, 1):
        for d_ in (0, 1):
            for q_ in (0, 1):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        r = sm.tsa.SARIMAX(y, order=(p_, d_, q_), enforce_stationarity=False,
                                           enforce_invertibility=False).fit(disp=False, maxiter=50, method="lbfgs")
                        aic = getattr(r, "aic", np.inf)
                        if best is None or aic < best[0]:
                            best = (aic, r)
                except Exception:
                    pass
    return best[1] if best else None


def arima_forecast(y_hist, steps):
    """Phase 10 census_arima logic."""
    y = np.asarray(y_hist, float); y = y[np.isfinite(y)]
    if y.size < 14:
        return np.full(steps, float(y[-1]) if y.size else np.nan)
    if np.nanvar(y) < 1e-9:
        return np.full(steps, float(y[-1]))
    m = small_auto_arima(y)
    if m is None:
        return np.full(steps, float(y[-1]))
    try:
        pm = np.asarray(m.get_forecast(steps=steps).predicted_mean, float).reshape(-1)
    except Exception:
        return np.full(steps, float(y[-1]))
    pm[~np.isfinite(pm)] = float(y[-1])
    return np.clip(pm[:steps], 0.0, None)


def daily_from_weeks(wk):
    """Known weekly admissions per day spread over the 7 days of each week (missing weeks stay missing)."""
    rows = {}
    for ws, v in wk.items():
        for k in range(7):
            rows[ws + pd.Timedelta(days=k)] = v
    return pd.Series(rows).sort_index()


def caselag_forecast(K, target_days):
    """admissions(t) = k cases7(t - L), L in 0..14 chosen by least squares on log weekly values over the known
    weeks of the window (last 8 known weeks), cases7 held at its last known value."""
    wk = K["wk"].dropna(); wk = wk[wk > 0].iloc[-8:]
    c7 = K["cases7"]; last = float(c7.iloc[-1])
    best = None
    for Lg in range(CASELAG_MAX + 1):
        lx = []
        for ws, v in wk.items():
            dd = [ws + pd.Timedelta(days=k) - pd.Timedelta(days=Lg) for k in range(7)]
            m = np.mean([float(c7.get(d, last)) for d in dd])
            if m > 0:
                lx.append(math.log(v) - math.log(m))
        if len(lx) < 3:
            continue
        lk = float(np.mean(lx)); sse = float(np.sum((np.array(lx) - lk) ** 2))
        if best is None or sse < best[0]:
            best = (sse, Lg, lk)
    if best is None:
        return np.full(len(target_days), np.nan), {}
    _, Lg, lk = best
    pred = np.array([math.exp(lk) * float(c7.get(d - pd.Timedelta(days=Lg), last)) if d - pd.Timedelta(days=Lg) <= K["L"]
                     else math.exp(lk) * last for d in target_days])
    return pred, {"caselag_L": Lg, "caselag_k": math.exp(lk)}


def comparators(K, tw, skip_arima=False):
    """NAIVE_ADM, CASELAG and ARIMA_ADM weekly predictions (per day) for the target weeks, from K only."""
    tdays = pd.date_range(tw[0], tw[-1] + pd.Timedelta(days=6), freq="D")
    wkk = K["wk"].dropna(); lastv = float(wkk.iloc[-1])
    comp = {"NAIVE_ADM": (np.full(len(tdays), lastv), {})}
    cl, cinfo = caselag_forecast(K, tdays); comp["CASELAG"] = (cl, cinfo)
    if not skip_arima:
        dd = daily_from_weeks(wkk)
        dd = dd[dd.index > K["last_pub_end"] - pd.Timedelta(days=ARIMA_LOOKBACK)]
        gap = max((tdays[0] - (dd.index.max() + DAY)).days, 0)
        comp["ARIMA_ADM"] = (arima_forecast(dd.values, gap + len(tdays))[gap:], {})
    out = {}
    for m, (arr, info) in comp.items():
        out[m] = ({h: float(np.mean(arr[(ws - tdays[0]).days:(ws - tdays[0]).days + 7])) for h, ws in enumerate(tw, 1)}, info)
    return out


def block_boot_ci(diff, block=BOOT_BLOCK, reps=BOOT_REPS, seed=BOOT_SEED):
    x = np.asarray(diff, float); n = len(x)
    if n < block + 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed); nb = int(math.ceil(n / block)); means = np.empty(reps)
    for b in range(reps):
        st = rng.integers(0, n - block + 1, size=nb)
        idx = (st[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
        means[b] = x[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ one origin of one chain
def l2_origin(D, t0, delay, behav, eta, x_carry, prev_theta, starts, max_nfev, seed_key):
    K = known_inputs(D, t0, delay)
    if K["wk"].dropna().empty:
        return None
    last_ws = K["wk"].dropna().index.max()
    target_weeks = [last_ws + pd.Timedelta(days=7 * h) for h in range(1, HWEEKS + 1)]
    end_day = target_weeks[-1] + pd.Timedelta(days=6)
    n_extra = (end_day - (t0 - DAY)).days
    P = Problem(K, D, t0, x_carry, behav, eta, n_extra)
    lo, hi = P.bounds(); best, costs = None, []
    for s in range(starts):
        r = least_squares(P.residuals, P.start(s, prev_theta, seed_key), bounds=(lo, hi), method="trf",
                          max_nfev=max_nfev, x_scale="jac")
        costs.append(float(r.cost))
        if best is None or r.cost < best.cost:
            best = r
    p, X, cases, adm = P.run(best.x)
    pred = {}
    for h, ws in enumerate(target_weeks, 1):
        i = (ws - P.w0).days
        pred[h] = (ws, float(adm[i:i + 7].mean()))
    x_next = X[7].copy()                                          # state at the next window start
    i0 = WINDOW                                                   # start of day t0
    Rt = float(math.exp(p["logk"][-1]) * X[i0][0] / DELTA)
    info = {"origin": str(t0.date()), "rho": p["rho"], "ihr": p["ihr"], "a_b": p["a_b"], "omega_b": p["omega_b"],
            "mult_E": p["mult_E"], "mult_I": p["mult_I"], "beta_last": float(math.exp(p["logk"][-1])),
            "Rt_t0": Rt, **{f"{k}_t0": float(X[i0][j]) for j, k in enumerate(["S", "U_v", "U_b", "E", "I", "R"])},
            "s_t0": float(P.s_d[WINDOW - 1]), "VE": P.ve, "omega_v": P.omv, "rho_prior": P.rho_prior,
            "cost": float(best.cost), "cost_starts": json.dumps([round(c, 4) for c in costs]), "nfev": int(best.nfev),
            "last_known_week_start": str(last_ws.date()), "n_adm_weeks_in_window": len(P.weeks),
            "beta_knots": json.dumps([round(float(math.exp(v)), 5) for v in p["logk"]])}
    traj = pd.DataFrame({"date": P.days, "adm_model": adm, "cases_model": cases,
                         **{k: X[:-1, j] for j, k in enumerate(["S", "U_v", "U_b", "E", "I", "R"])}})
    return {"pred": pred, "x_next": x_next, "theta": best.x, "info": info, "traj": traj, "K": K,
            "target_weeks": target_weeks, "adm_daily": pd.Series(adm, index=P.days)}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--adm-dir", default="phase15_admissions")
    ap.add_argument("--vax", default="data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv")
    ap.add_argument("--hosp-file", default="covid19hospitalbycounty.csv")
    ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
    ap.add_argument("--first-origin", default=FIRST_ORIGIN)
    ap.add_argument("--max-origins", type=int, default=0, help="smoke tests only (0 = all)")
    ap.add_argument("--chains", default="all", help="smoke tests only: comma separated chain indices")
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--max-nfev", type=int, default=300)
    ap.add_argument("--skip-arima", action="store_true", help="smoke tests only")
    ap.add_argument("--skip-icu", action="store_true", help="smoke tests only")
    ap.add_argument("--rho-schedule", default="c1", choices=sorted(RHO_PRIOR_SCHEDULES),
                    help="ascertainment prior schedule; protocol_v1 reproduces the 2026-10-12 outputs")
    ap.add_argument("--out", default="phase16_layer2_c1")
    ap.add_argument("--overwrite", action="store_true")
    A = ap.parse_args(argv)
    global RHO_PRIOR_SCHED
    RHO_PRIOR_SCHED = RHO_PRIOR_SCHEDULES[A.rho_schedule]
    if A.first_origin != FIRST_ORIGIN and not A.max_origins:
        raise SystemExit("--first-origin is for smoke tests only (moving horizon starts at 2020-10-03)")
    out = Path(A.out)
    if out.exists() and any(out.iterdir()) and not A.overwrite:
        raise SystemExit(f"{out} is not empty; use a new --out or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    LOG = open(out / "run_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    inputs = set()
    counties = [c.strip() for c in A.counties.split(",") if c.strip()]
    origins = list(pd.date_range(A.first_origin, LAST_ORIGIN, freq="7D"))
    assert all(o.dayofweek == 5 for o in origins)
    if A.max_origins:
        origins = origins[:A.max_origins]
    chains = CHAINS if A.chains == "all" else [CHAINS[int(i)] for i in A.chains.split(",")]
    log(f"Phase 16 | counties {counties} | origins {len(origins)} ({origins[0].date()} to {origins[-1].date()}) | "
        f"chains {[(c[0], c[1]) for c in chains]} | starts {A.starts} | max_nfev {A.max_nfev}")
    log(f"rho prior schedule {A.rho_schedule}: {RHO_PRIOR_SCHED}")
    import layer1c_recalibration as L1C                             # Phase 15 hospital model (ICU, secondary)
    pred_rows, info_rows, trajs, icu_rows = [], [], [], []
    for ci, county in enumerate(counties):
        D = load_county(A.root, county, A.adm_dir, A.vax, inputs)
        wk_true = D["wk"]
        census = None
        if not A.skip_icu:
            census, _, _ = L1C.load_county(A.root, A.adm_dir, A.hosp_file, county, inputs)
        for chain_i, (name, delay, behav, eta) in enumerate(chains):
            t_start = time.time()
            x = initial_state(D, origins[0] - pd.Timedelta(days=WINDOW), sched(RHO_PRIOR_SCHED, origins[0]))
            theta = None
            for oi, t0 in enumerate(origins):
                res = l2_origin(D, t0, delay, behav, eta, x, theta, A.starts, A.max_nfev,
                                (ci, chain_i, int(t0.toordinal())))
                if res is None:
                    continue
                x, theta = res["x_next"], res["theta"]
                info_rows.append({"county": county, "model": name, "delay": delay, "period": period_of(t0), **res["info"]})
                tr = res["traj"]; tr.insert(0, "origin", str(t0.date())); tr.insert(0, "delay", delay)
                tr.insert(0, "model", name); tr.insert(0, "county", county); trajs.append(tr)
                for h, (ws, v) in res["pred"].items():
                    pred_rows.append({"county": county, "origin": str(t0.date()), "period": period_of(t0), "model": name,
                                      "delay": delay, "h": h, "week_start": str(ws.date()), "pred": v,
                                      "obs": float(wk_true.get(ws, np.nan))})
                # comparators and ICU, once per county and delay (with the primary L2 chain)
                if name == "L2":
                    K = res["K"]; tw = res["target_weeks"]
                    for mname, (preds, cinfo) in comparators(K, tw, A.skip_arima).items():
                        for h, ws in enumerate(tw, 1):
                            pred_rows.append({"county": county, "origin": str(t0.date()), "period": period_of(t0),
                                              "model": mname, "delay": delay, "h": h, "week_start": str(ws.date()),
                                              "pred": preds[h], "obs": float(wk_true.get(ws, np.nan)), **cinfo})
                    if census is not None and delay == 1:
                        icu_rows.append(icu_from_l2(L1C, census, K, res["adm_daily"], t0, A, county))
                if (oi + 1) % 10 == 0 or oi + 1 == len(origins):
                    log(f"  {county} {name} delay {delay}: {oi + 1}/{len(origins)} origins ({time.time() - t_start:.0f} s)")
    Pd = pd.DataFrame(pred_rows); Pd["abs_err"] = (Pd["pred"] - Pd["obs"]).abs()
    Pd.to_csv(out / "forecasts_weekly.csv", index=False)
    pd.DataFrame(info_rows).to_csv(out / "params_states.csv", index=False)
    pd.concat(trajs).to_csv(out / "trajectories.csv", index=False)
    Icu = pd.DataFrame(icu_rows)
    if len(Icu):
        Icu.to_csv(out / "icu_secondary.csv", index=False)

    # primary error per origin: mean over h = 1..3
    prim = (Pd[Pd["h"].isin(PRIMARY_H)].groupby(["county", "origin", "period", "model", "delay"])
            .agg(err=("abs_err", "mean"), obs=("obs", "mean"), n=("abs_err", "count")).reset_index())
    prim = prim[prim["n"] == len(PRIMARY_H)]
    prim.to_csv(out / "primary_error_origin.csv", index=False)
    summ = prim.groupby(["county", "model", "delay"]).agg(n_origins=("err", "size"), mean_err=("err", "mean"),
                                                         mean_obs=("obs", "mean")).reset_index()
    summ["rel_err"] = summ["mean_err"] / summ["mean_obs"]
    summ.to_csv(out / "summary.csv", index=False)
    byh = Pd.groupby(["county", "model", "delay", "h"]).agg(mean_abs_err=("abs_err", "mean"), mean_obs=("obs", "mean")).reset_index()
    byh.to_csv(out / "summary_by_h.csv", index=False)
    prim.groupby(["county", "period", "model", "delay"])["err"].mean().reset_index().to_csv(out / "summary_by_period.csv", index=False)

    def paired(county, a, b, delay=1):
        x = prim[(prim["county"] == county) & (prim["model"] == a) & (prim["delay"] == delay)].set_index("origin")["err"]
        y = prim[(prim["county"] == county) & (prim["model"] == b) & (prim["delay"] == delay)].set_index("origin")["err"]
        j = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna().sort_index()
        d = (j["a"] - j["b"]).values; lo, hi = block_boot_ci(d)
        return {"n_origins": int(len(j)), f"mean_{a}": float(j["a"].mean()) if len(j) else None,
                f"mean_{b}": float(j["b"].mean()) if len(j) else None, "mean_diff": float(d.mean()) if len(d) else None,
                "ci95": [lo, hi], "significantly_better": bool(len(d) and d.mean() < 0 and np.isfinite(hi) and hi < 0),
                "not_significantly_worse": bool(len(d) and np.isfinite(lo) and lo <= 0)}

    crit = {"D1": {}, "D2": {}, "adequacy_A2": {}}
    for c in counties:
        if "ARIMA_ADM" in set(prim["model"]):
            r = paired(c, "L2", "ARIMA_ADM"); crit["D1"][c] = r
            crit["adequacy_A2"][c] = {"not_significantly_worse_than_ARIMA_ADM": r["not_significantly_worse"]}
        crit["D2"][c] = paired(c, "L2", "CASELAG")
    crit["D1_met"] = bool(sum(v.get("significantly_better", False) for v in crit["D1"].values()) >= 2)
    crit["D2_met"] = bool(sum(v.get("significantly_better", False) for v in crit["D2"].values()) >= 2)
    crit["adequate_A2"] = bool(sum(v["not_significantly_worse_than_ARIMA_ADM"] for v in crit["adequacy_A2"].values()) >= 2)
    if len(Icu):
        crit["ICU_secondary"] = {}
        for c in counties:
            j = Icu[Icu["county"] == c][["MAE14_H4_L2", "MAE14_ARIMA"]].dropna()
            d = (j["MAE14_H4_L2"] - j["MAE14_ARIMA"]).values; lo, hi = block_boot_ci(d)
            crit["ICU_secondary"][c] = {"n": int(len(j)), "mean_H4_L2": float(j["MAE14_H4_L2"].mean()) if len(j) else None,
                                        "mean_ARIMA": float(j["MAE14_ARIMA"].mean()) if len(j) else None, "ci95_diff": [lo, hi]}
    crit["rules"] = {"primary_error": "mean absolute error of weekly admissions per day over h = 1, 2, 3",
                     "D1": "L2 (delay 1) significantly better than ARIMA_ADM in >= 2 of 3 counties",
                     "D2": "L2 (delay 1) significantly better than CASELAG in >= 2 of 3 counties",
                     "A2": "L2 not significantly worse than ARIMA_ADM (interval lower end <= 0) in >= 2 of 3 counties",
                     "bootstrap": f"moving block, block {BOOT_BLOCK}, {BOOT_REPS} replicates, seed {BOOT_SEED}"}
    (out / "criteria.json").write_text(json.dumps(crit, indent=2))
    log("\nD1:", json.dumps(crit["D1"])); log("D2:", json.dumps(crit["D2"]))
    log("D1 met:", crit["D1_met"], "| D2 met:", crit["D2_met"], "| adequate (A2):", crit["adequate_A2"])
    import scipy, statsmodels
    man = {"script": Path(__file__).name, "script_sha256": sha(__file__),
           "layer1c_sha256": sha(Path(L1C.__file__)), "command": " ".join([sys.executable] + sys.argv),
           "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0],
           "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
           "inputs_sha256": {p: sha(p) for p in sorted(inputs)},
           "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".json")}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    log(f"\nSaved to {out}/")


def icu_from_l2(L1C, census, K, adm_l2, t0, A, county):
    """Secondary (protocol section 5): Phase 15 H4, window 56, admissions = known weeks then L2 daily
    admissions; ICU 14 day MAE against Phase 15 ICU ARIMA on the same origin."""
    Ld = t0 - DAY
    known = daily_from_weeks(K["wk"].dropna())
    win_days = pd.date_range(Ld - pd.Timedelta(days=55), Ld, freq="D")
    kc = census[census.index <= Ld]
    cw = kc.reindex(win_days); first_ok = cw.dropna().index.min()
    row = {"county": county, "origin": str(t0.date()), "period": period_of(t0)}
    if pd.isna(first_ok):
        return row
    cw = cw[cw.index >= first_ok]; wd = cw.index
    Lc = kc.dropna().index.max()
    alld = pd.date_range(wd[0], Ld + pd.Timedelta(days=14), freq="D")
    adm = np.array([float(known.get(d, np.nan)) if d in known.index else float(adm_l2.get(d, np.nan)) for d in alld])
    if not np.all(np.isfinite(adm)):
        return row
    res, _ = L1C.fit_window(adm[:len(wd)], cw["ward"].values, cw["icu"].values, A.starts, A.max_nfev, (0, int(t0.toordinal()), 56, 9))
    p = L1C.unpack(res.x); Z = L1C.simulate(p, adm[:len(wd)], cw["ward"].values[0], cw["icu"].values[0])
    j = (Lc - wd[0]).days; wp = Z[j, 0] + Z[j, 2]; fw = Z[j, 0] / wp if wp > 0 else 1.0
    z = np.array([fw * kc.loc[Lc, "ward"], kc.loc[Lc, "icu"], (1 - fw) * kc.loc[Lc, "ward"], 0.0])
    n_steps = (Ld - Lc).days + 14
    _, ip = L1C.project(p, z, adm[(Lc - wd[0]).days:(Lc - wd[0]).days + n_steps])
    ip = ip[-14:]
    obs = census["icu"].reindex(pd.date_range(Ld + DAY, Ld + pd.Timedelta(days=14))).values
    hist = kc[kc.index > Ld - pd.Timedelta(days=L1C.ARIMA_LOOKBACK)]
    ar = L1C.census_arima(hist["icu"].values, 14 + (Ld - Lc).days)[-14:]
    row["MAE14_H4_L2"] = L1C.mae(ip, obs, 14)[0]; row["MAE14_ARIMA"] = L1C.mae(ar, obs, 14)[0]
    return row


if __name__ == "__main__":
    main()
