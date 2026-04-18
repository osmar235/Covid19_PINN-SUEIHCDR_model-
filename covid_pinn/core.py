#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNIFIED MULTI-CITY PINN RUNNER
==============================
Modes:
  fit_check  → Only train_full_and_export (fast, ~5-15 min/city)
  full       → train_full_and_export + run_multi_window_eval (5-10 hrs/city)

Usage:
  python run_cities.py --mode fit_check
  python run_cities.py --mode fit_check --cities "Seattle,London,Rome"
  python run_cities.py --mode full --cities "Seattle,London"
  python run_cities.py --skip-us   # world cities only
  python run_cities.py --skip-world # US cities only
"""

import os, sys, time, math, warnings, random, argparse, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from pathlib import Path
from scipy.signal import butter, filtfilt

# ======================== SEED & THREADS ========================
SEED = 1337; random.seed(SEED); np.random.seed(SEED)
N_THREADS = int(os.environ.get("PINN_NUM_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))
plt.rcParams.update({"figure.dpi": 300, "font.size": 12})

try:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    HAS_SM = True
except: HAS_SM = False

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(N_THREADS)

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ======================== PATHS ========================
BASE_PATH_STR = os.environ.get("PINN_DATA_PATH", ".")
PATH = Path(BASE_PATH_STR)
OUT_PATH_STR = os.environ.get("PINN_OUT_PATH", BASE_PATH_STR)

# ----- Data-file resolution ---------------------------------------------
# Data files (covid_confirmed_usafacts.csv, epidemiology.csv, etc.) may live
# under PINN_DATA_PATH, at the repo root, in a sibling data folder, or in a
# sibling CODES_04032021 folder. This resolver looks in all of those places,
# so the code works out of the box for most layouts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_SEARCH_ROOTS = [
    PATH,
    _REPO_ROOT,
    _REPO_ROOT / "data",
    _REPO_ROOT.parent / "CODES_04032021",
    _REPO_ROOT / "CODES_04032021",
]


def _resolve_data_path(filename: str) -> Path:
    """Return the first existing location for ``filename`` among the data roots.

    Checks PINN_DATA_PATH first, then common fallbacks (repo root, a
    ``data/`` subfolder, a sibling ``CODES_04032021/`` folder). If none of
    the candidates exist, returns ``PATH / filename`` so the resulting
    FileNotFoundError carries a recognizable path to debug.
    """
    for root in _DATA_SEARCH_ROOTS:
        cand = Path(root) / filename
        if cand.exists():
            return cand
    return PATH / filename

# ======================== CITY CONFIGS ========================
US_CITIES = [
    # (label, county_name, state_abbr, subregion_1)
    ("SanDiego",  "San Diego County",  "CA", "California"),
    ("Seattle",   "King County",       "WA", "Washington"),
    ("NewYork",   "New York County",   "NY", "New York"),
    ("Chicago",   "Cook County",       "IL", "Illinois"),
    ("Houston",   "Harris County",     "TX", "Texas"),
    ("Phoenix",   "Maricopa County",   "AZ", "Arizona"),
    ("Miami",     "Miami-Dade County", "FL", "Florida"),
    ("Denver",    "Denver County",     "CO", "Colorado"),
    ("LosAngeles",  "Los Angeles County",  "CA", "California"),
    ("SanFrancisco",  "San Francisco County",  "CA", "California"),
]

CITY_SPECS = {
    "London":   {"key": "GB_ENG"},
    "SaoPaulo": {"country": "Brazil",    "match": {"subregion1_name": "São Paulo"}, "prefer_agg": True,
                 "fallback_keys": ["BR_SP", "BR_SP_SAO"]},
    "Rome":     {"key": "IT_62"},
    "Paris":    {"country": "France",    "match": {"subregion2_name": "Paris"},           "prefer_agg": True},
    "Tokyo":    {"country": "Japan",     "match": {"subregion1_name": "Tokyo"},           "prefer_agg": True},
    "Sydney":   {"country": "Australia", "match": {"subregion1_name": "New South Wales"}, "prefer_agg": True},
    "Berlin":   {"country": "Germany",   "match": {"subregion1_name": "Berlin"},          "prefer_agg": True,
                 "fallback_keys": ["DE_BE"]},
    "Moscow":   {"country": "Russia",    "match": {"subregion1_name": "Moscow"},          "prefer_agg": True,
                 "fallback_keys": ["RU_MOW", "RU_MOS"]},
}
CITY_POP = {
    "London": 9e6, "Paris": 2.16e6, "SaoPaulo": 12.3e6, "Tokyo": 14e6,
    "Sydney": 5.3e6, "Berlin": 3.7e6, "Moscow": 12.6e6, "Rome": 2.87e6,
}
WORLD_CITIES = list(CITY_SPECS.keys())

# ======================== HYPERPARAMETERS ========================
CASES_ONLY = False
EPOCHS_FULL = 3500; LR_FULL = 3e-3; SD_LAG_DAYS = 7
EPOCHS_MAX = 7000; VALIDATION_DAYS = 14; PATIENCE_EPOCHS = 500; VAL_CHECK_FREQ = 25

W_CUM_CASES=2.0; W_WEEK_CASES=8.0; W_PHYS=8.0; W_IC_ANCHOR=1.0
W_POP=5.0; W_PRIOR=1.0; W_FSmooth=0.04; W_FOURIER_L2=1e-3; W_F_VAR=1e-3; W_S0=5.0
PR_DUR_INCUB=5.0; PR_DUR_INF=7.0; PR_DUR_WARD=10.0; PR_DUR_ICU=14.0
PR_W_INCUB=2.0; PR_W_INF=3.0; PR_W_WARD=3.0; PR_W_ICU=4.0

BETA0_BOUNDS=(0.05,0.50); SIGMA_BOUNDS=(1/7,1/2); DELTA_BOUNDS=(1/9,1/6)
ZETA_BOUNDS=(1/14,1/3); EPSI_BOUNDS=(1/21,1/5)
M_BOUNDS=(0.60,0.99); C_BOUNDS=(0.20,0.60); F_BOUNDS=(0.35,0.65)
OMEGA_BOUNDS=(1/180,1/30); ETA_BOUNDS=(1/365,1/60)
ALPH_BOUNDS=(0.00,0.003); KSD=1.5; RHO_BOUNDS=(0.10,1.0)

F_TIME_MODE="fourier"; FOURIER_K=4
SD_FUTURE_MODE="arima"; SD_FUTURE_PARAM={"arima_min_len": 30}

# Publication validation: 5 distinct regimes, 1 cut each, 2 horizons = 10 evals per city
# Each tuple: (regime_name, cut_date, lookback_days)
#   FirstWave     — early exponential growth, sparse data
#   Winter20_Peak — largest pre-vaccine wave
#   Delta_Peak    — VOC emergence + partial immunity
#   Omicron_Shock — massive regime shift, biggest stress test
#   BA5_ImmuneEsc — late pandemic, waning immunity + escape
VALIDATION_CONFIG = [
    ("FirstWave",  "2020-05-01",  60),   # early exponential, sparse data
    ("Winter20",   "2021-01-15", 120),   # largest pre-vaccine wave (optimized from 180)
    ("Delta",      "2021-08-15", 300),   # variant displacement — long history helps (optimized from 180)
    ("Omicron",    "2022-01-15", 150),   # immune-escape shock (optimized from 180)
    ("BA5_Waning", "2022-06-15", 150),   # waning immunity + subvariant (optimized from 180)
]
HORIZON_LIST=[7,14]; EPOCHS_TINY=2500; SD_LAG_GRID=[3,7]
ARIMA_W_LO=0.25; ARIMA_W_HI=0.65

VARIANT_EVENTS = [
    ("Delta","2021-06-15",1.6,10), ("Omicron","2021-12-15",3.50,5),
    ("BA.5","2022-06-15",1.30,10),  ("XBB","2022-12-15",1.20,10),
]
VAR_BUMP_BOUNDS=(1.00,8.00); VAR_PRIOR_STRENGTH=1.5

# ======================== UTILITY FUNCTIONS ========================
def butter_lowpass(x, cutoff=0.14, order=4):
    b, a = butter(order, cutoff, btype="low"); return filtfilt(b, a, x)

def weekly_avg_np(x, k=7):
    x = np.asarray(x, float)
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy() if x.size else x

def daily_from_cum_np(cum):
    cum = np.asarray(cum, float)
    return np.clip(np.diff(np.r_[cum[0], cum]), 0, None) if cum.size else cum

def make_quarter_ids(dates_like):
    q = pd.PeriodIndex(pd.to_datetime(dates_like), freq="Q")
    uq = pd.unique(q); mapd = {str(p):i for i,p in enumerate(uq)}
    return np.array([mapd[str(p)] for p in q], dtype=int), len(uq)

def safe_mape(y_true, y_pred, floor=1.0):
    denom = np.maximum(np.abs(np.asarray(y_true,float)), floor)
    return float(np.mean(np.abs(np.asarray(y_true,float) - np.asarray(y_pred,float)) / denom)) * 100.0

def _bounded(raw, lb, ub): return lb + (ub - lb) * torch.sigmoid(raw)
def _soft_prior(x, target, width): return ((x - target)/width)**2

def lag_tensor(x, lag):
    if lag <= 0: return x
    return torch.cat([x[:1].repeat(lag,1), x[:-lag]], dim=0)

def get_cut_idx_full(df, cut_date):
    idxs = np.where(pd.to_datetime(df["date"].values) <= pd.Timestamp(cut_date))[0]
    if idxs.size == 0: raise ValueError(f"cut_date precedes data")
    return int(idxs[-1])

# ======================== DATA LOADING: US ========================
def load_us_county_series(county_name, state_abbr, subregion_1):
    pop = pd.read_csv(_resolve_data_path("covid_county_population_usafacts.csv"))
    row = pop[(pop["County Name"]==county_name) & (pop["State"]==state_abbr)]
    if row.empty: raise ValueError(f"County not found: {county_name}, {state_abbr}")
    fips = int(row.iloc[0]["countyFIPS"]); Npop = float(row.iloc[0]["population"])

    cwide = pd.read_csv(_resolve_data_path("covid_confirmed_usafacts.csv"))
    crow = cwide[cwide["countyFIPS"]==fips].iloc[0].drop(["countyFIPS","County Name","State","StateFIPS"])
    dfc = pd.DataFrame({"date": pd.to_datetime(crow.index), "cases": crow.values.astype(float)})

    dwide = pd.read_csv(_resolve_data_path("covid_deaths_usafacts.csv"))
    drow = dwide[dwide["countyFIPS"]==fips].iloc[0].drop(["countyFIPS","County Name","State","StateFIPS"])
    dfd = pd.DataFrame({"date": pd.to_datetime(drow.index), "deaths": drow.values.astype(float)})

    mob_files = [p for p in (_resolve_data_path(f) for f in [
                 "2020_US_Region_Mobility_Report.csv",
                 "2021_US_Region_Mobility_Report.csv",
                 "2022_US_Region_Mobility_Report.csv"]) if p.exists()]
    if not mob_files:
        df_sd = pd.DataFrame({"date": dfc["date"].copy(), "sd": 0.0})
    else:
        mm = pd.concat([pd.read_csv(f) for f in mob_files], ignore_index=True)
        mm = mm[(mm["sub_region_1"]==subregion_1) & (mm["sub_region_2"]==county_name)].copy()
        mm["date"] = pd.to_datetime(mm["date"])
        cols = ["retail_and_recreation_percent_change_from_baseline",
                "grocery_and_pharmacy_percent_change_from_baseline",
                "parks_percent_change_from_baseline",
                "transit_stations_percent_change_from_baseline",
                "workplaces_percent_change_from_baseline",
                "residential_percent_change_from_baseline"]
        mm[cols] = mm[cols].fillna(0.0)
        sd_raw = -mm[cols].mean(axis=1).to_numpy()
        sd_smooth = butter_lowpass(sd_raw, cutoff=0.14) / 100.0
        df_sd = pd.DataFrame({"date": mm["date"].values, "sd": sd_smooth})

    df = dfc.merge(dfd, on="date").merge(df_sd, on="date", how="left").sort_values("date").reset_index(drop=True)
    df["sd"] = df["sd"].ffill().fillna(0.0).clip(0.0, 1.0)
    df["cases"] = np.maximum.accumulate(df["cases"].values)
    df["deaths"] = np.maximum.accumulate(df["deaths"].values)
    return df, Npop

# ======================== DATA LOADING: WORLD ========================
def _get_key_col(df):
    for c in ["location_key", "key"]:
        if c in df.columns: return c
    raise KeyError("No key column found")

def _ensure_google_data():
    import urllib.request
    BASE = "https://storage.googleapis.com/covid19-open-data/v3"
    for f in ["index.csv", "epidemiology.csv", "mobility.csv"]:
        p = PATH / f
        if not p.exists():
            print(f"  Downloading {f}...")
            urllib.request.urlretrieve(f"{BASE}/{f}", p)

def _resolve_key(spec):
    if "key" in spec: return spec["key"]
    idx = pd.read_csv(_resolve_data_path("index.csv")); KEY = _get_key_col(idx)
    df = idx.copy()
    for c in ["country_name","subregion1_name","subregion2_name"]:
        if c in df.columns: df[c] = df[c].astype(str)
    if "country" in spec and "country_name" in df.columns:
        df = df[df["country_name"].str.lower()==spec["country"].lower()]
    for col, val in spec.get("match", {}).items():
        if col in df.columns: df = df[df[col].str.lower()==str(val).lower()]
    if df.empty:
        # Try fallback keys before giving up
        for fk in spec.get("fallback_keys", []):
            check = idx[idx[KEY]==fk]
            if not check.empty:
                print(f"    Using fallback key: {fk}")
                return fk
        raise ValueError(f"No match for {spec} (tried fallbacks too)")
    if spec.get("prefer_agg") and "aggregation_level" in df.columns:
        df["aggregation_level"] = pd.to_numeric(df["aggregation_level"], errors="coerce")
        df = df.sort_values("aggregation_level", ascending=False)
    return df.iloc[0][KEY]

def load_world_city_series(city_name, mob_col="mobility_workplaces"):
    _ensure_google_data()
    spec = CITY_SPECS[city_name]
    loc_key = _resolve_key(spec)
    print(f"    Resolved key: {loc_key}")

    epi = pd.read_csv(_resolve_data_path("epidemiology.csv"), parse_dates=["date"])
    KEY = _get_key_col(epi); epi_filt = epi[epi[KEY]==loc_key].sort_values("date")

    # If primary key yields no data, try fallback keys
    if epi_filt.empty:
        for fk in spec.get("fallback_keys", []):
            epi_filt = epi[epi[KEY]==fk].sort_values("date")
            if not epi_filt.empty:
                print(f"    Primary key empty, using fallback: {fk}"); loc_key = fk; break
    if epi_filt.empty:
        raise ValueError(f"No epidemiology data for {city_name} (key={loc_key})")

    epi = epi_filt
    cases = epi["cumulative_confirmed"].astype(float).to_numpy() if "cumulative_confirmed" in epi.columns \
        else epi.get("new_confirmed",0).fillna(0).astype(float).cumsum().to_numpy()
    deaths = epi["cumulative_deceased"].astype(float).to_numpy() if "cumulative_deceased" in epi.columns \
        else epi.get("new_deceased",0).fillna(0).astype(float).cumsum().to_numpy()
    dfe = pd.DataFrame({"date": epi["date"].values, "cases": cases, "deaths": deaths})

    mob = pd.read_csv(_resolve_data_path("mobility.csv"), parse_dates=["date"])
    KEY_m = _get_key_col(mob); mob = mob[mob[KEY_m]==loc_key].sort_values("date")
    col = None
    for try_col in [mob_col, f"mobility_{mob_col}", mob_col.replace("mobility_","")]:
        if try_col in mob.columns: col = try_col; break
    if col is None:
        dfm = pd.DataFrame({"date": dfe["date"].copy(), "sd": 0.0})
    else:
        x = mob[col].astype(float).ffill().fillna(0.0).to_numpy()
        dfm = pd.DataFrame({"date": mob["date"].values, "sd": np.clip(-x/100, 0, 1)})

    df = dfe.merge(dfm, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if df.empty:
        # Fall back to left join if inner yields nothing (mobility may not overlap)
        df = dfe.copy(); df["sd"] = 0.0
    df["sd"] = df["sd"].ffill().fillna(0.0).clip(0.0, 1.0)

    # Enforce monotonicity + clip end-of-series outlier spikes
    # (fixes London-like data correction artifacts)
    cases_raw = df["cases"].values.astype(float)
    cases_mono = np.maximum.accumulate(np.nan_to_num(cases_raw, 0))
    daily = np.diff(np.r_[cases_mono[0], cases_mono])
    if daily.size > 14:
        med_daily = np.median(daily[-90:]) if daily.size >= 90 else np.median(daily)
        outlier_thresh = max(med_daily * 10, 500)
        for i in range(len(daily)-1, max(0, len(daily)-14), -1):
            if daily[i] > outlier_thresh:
                print(f"    Clipping outlier spike at end: day {i}, daily={daily[i]:.0f} > {outlier_thresh:.0f}")
                daily[i] = med_daily
        cases_mono = cases_mono[0] + np.cumsum(daily)
    df["cases"] = cases_mono
    df["deaths"] = np.maximum.accumulate(np.nan_to_num(df["deaths"].values.astype(float), 0))

    print(f"    Loaded {city_name}: {len(df)} days, {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    return df, float(CITY_POP[city_name])
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_cities_model.py — PINN model + training + multi-window eval + main()
Import this after run_cities.py (or paste below it in a single file).
"""

# ======================== VARIANT HELPERS ========================
def _logistic(x, k=4.0): return 1.0/(1.0+np.exp(-k*x))

def _event_step_series(dates_np, event_date_str, ramp_days):
    dates = pd.to_datetime(dates_np); t0 = pd.Timestamp(event_date_str)
    x = (dates - t0).days.to_numpy(dtype=float) / max(1.0, float(ramp_days))
    return _logistic(x, k=4.0).reshape(-1,1).astype(float)

def make_variant_structs(dates_np, events):
    T = len(dates_np); V = np.ones((T,1), float); steps = []
    for name, d0, m_prior, ramp in events:
        step = _event_step_series(dates_np, d0, ramp)
        V *= (1.0 + (m_prior - 1.0) * step); steps.append(step)
    return V, steps

# ======================== ARIMA HELPERS ========================
def _small_auto_arima_safe(y):
    if not HAS_SM: return None
    best = None
    for p in (0,1):
        for d in (0,1):
            for q in (0,1):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        m = sm.tsa.SARIMAX(y, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
                        r = m.fit(disp=False, maxiter=50, method="lbfgs")
                        aic = getattr(r, "aic", np.inf)
                        if best is None or aic < best[0]: best = (aic, r)
                except: pass
    return best[1] if best else None

def forecast_daily_levels_from_cum(obs_cum, steps, recalib_window_days=180):
    y = weekly_avg_np(daily_from_cum_np(obs_cum).astype(float))
    y_fit = y[-recalib_window_days:]; last = float(y[-1]) if y.size else 0.0
    if y_fit.size < 7 or np.nanvar(y_fit) < 1e-9 or not HAS_SM:
        return np.full(steps, last, float)
    model = _small_auto_arima_safe(y_fit)
    if model is None: return np.full(steps, last, float)
    try:
        pm = np.asarray(model.get_forecast(steps=steps).predicted_mean, float).reshape(-1)
        if pm.shape[0] < steps: pm = np.pad(pm, (0, steps-pm.shape[0]), mode="edge")
    except: pm = np.full(steps, last, float)
    pm = np.clip(pm, 0.0, None); pm[~np.isfinite(pm)] = last
    return pm[:steps]

def forecast_sd_future_arima(sd_hist, steps, min_len=30):
    if not HAS_SM or steps<=0 or sd_hist is None or len(sd_hist)<max(5,min_len):
        last = float(sd_hist[-1]) if sd_hist is not None and len(sd_hist) else 0.0
        return np.full(steps, last, float)
    y = np.asarray(sd_hist, float)
    if not np.all(np.isfinite(y)) or np.nanstd(y)<1e-6:
        return np.full(steps, float(y[-1]), float)
    model = _small_auto_arima_safe(y)
    if model is None: return np.full(steps, float(y[-1]), float)
    try:
        fut = np.asarray(model.get_forecast(steps=steps).predicted_mean, float).reshape(-1)
        if fut.shape[0]<steps: fut = np.pad(fut, (0, steps-fut.shape[0]), mode="edge")
    except: fut = np.full(steps, float(y[-1]), float)
    fut = np.clip(fut, 0.0, 1.0)
    if not np.all(np.isfinite(fut)): fut[~np.isfinite(fut)] = float(y[-1])
    return fut

def project_sd_future(sd_history, steps, mode=SD_FUTURE_MODE, param=SD_FUTURE_PARAM):
    if steps <= 0: return np.array([], dtype=float)
    sd_history = np.asarray(sd_history, float)
    if mode == "arima":
        return np.clip(forecast_sd_future_arima(sd_history, steps, int(param.get("arima_min_len",30))), 0, 1)
    return np.full(steps, float(sd_history[-1]) if len(sd_history) else 0.0, float)

# ======================== BAYESIAN ALPHA ========================
def behavior_signal_from_deaths(dates, deaths_cum, k=7):
    dw = weekly_avg_np(daily_from_cum_np(deaths_cum))
    d_dw = np.diff(np.r_[dw[:1], dw])
    s = (d_dw - np.median(d_dw)) / (np.median(np.abs(d_dw - np.median(d_dw))) + 1e-9)
    return np.clip(s, -4.0, 4.0)

def bayesian_alpha_series(dates_hist, deaths_cum, alpha_min=0.0, alpha_max=0.5, kappa=0.12, lam=0.995):
    sig = behavior_signal_from_deaths(dates_hist, deaths_cum)
    a=2.0; b=2.0; s_hist = np.zeros_like(sig, float)
    for t, z in enumerate(sig):
        a, b = lam*a, lam*b
        if z >= 0: a += kappa*float(z)
        else: b += kappa*float(-z)
        a = max(a, 0.1); b = max(b, 0.1); s_hist[t] = a/(a+b)
    return alpha_min + (alpha_max - alpha_min) * s_hist

def project_alpha_future(alpha_hist, steps, mode="exp_decay", tau=7, floor=0.05):
    if steps <= 0: return np.array([], float)
    last = float(alpha_hist[-1]) if len(alpha_hist) else floor
    base = max(floor, np.quantile(alpha_hist[-21:], 0.25) if len(alpha_hist)>=21 else floor)
    t = np.arange(1, steps+1, dtype=float)
    return np.clip(base + (last - base)*np.exp(-t/max(tau,1.0)), 0, 1)

# ======================== REGIME GATING ========================
def gating_from_regime(rt_hist, sd_hist):
    if rt_hist.size < 14: return (ARIMA_W_LO+ARIMA_W_HI)/2
    rt14 = rt_hist[-14:]; rt_cv = np.std(rt14)/(np.mean(rt14)+1e-9)
    stab = np.clip(1.0 - rt_cv/0.5, 0, 1)
    sd_sig = np.clip(sd_hist[-1]/0.5, 0, 1) if sd_hist.size >= 7 else 0.0
    w = np.clip(0.7*stab + 0.3*sd_sig, 0, 1)
    return ARIMA_W_LO + (ARIMA_W_HI - ARIMA_W_LO)*w

def carry_wavg_future_from_cum(obs_cum_hist, fut_cum, k=7):
    obs_cum_hist = np.asarray(obs_cum_hist, float); fut_cum = np.asarray(fut_cum, float)
    hist_daily = np.clip(np.diff(np.r_[obs_cum_hist[0], obs_cum_hist]), 0, None) if obs_cum_hist.size else np.array([])
    if fut_cum.size == 0: return np.array([])
    last = obs_cum_hist[-1] if obs_cum_hist.size else 0.0
    fut_daily = np.clip(np.diff(np.r_[last, fut_cum]), 0, None)
    carry = hist_daily[-(k-1):] if hist_daily.size >= (k-1) else hist_daily
    combo = np.r_[carry, fut_daily]; wk = weekly_avg_np(combo, k=k)
    return wk[len(carry):][:len(fut_cum)]

def forecast_loglinear_7d(series_7d, current_date, history_days=21, horizon=14):
    train = series_7d.loc[current_date-pd.Timedelta(days=history_days):current_date]
    if len(train) < 7: return np.full(horizon, float(train.iloc[-1]) if len(train) else 0.0, float)
    y = train.values.astype(float); X = np.arange(len(y)).reshape(-1,1)
    m = LinearRegression().fit(X, np.log(y+1.0))
    return np.clip(np.exp(m.predict(np.arange(len(y),len(y)+horizon).reshape(-1,1)))-1, 0, None)

def forecast_bayesian_poly_7d(series_7d, current_date, history_days=21, horizon=14, degree=2):
    train = series_7d.loc[current_date-pd.Timedelta(days=history_days):current_date]
    if len(train) < 7: return np.full(horizon, float(train.iloc[-1]) if len(train) else 0.0, float)
    y = train.values.astype(float); X = np.arange(len(y)).reshape(-1,1)
    m = make_pipeline(PolynomialFeatures(degree=degree,include_bias=False),
        BayesianRidge(alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6,fit_intercept=True)).fit(X, np.log(y+1.0))
    return np.clip(np.exp(m.predict(np.arange(len(y),len(y)+horizon).reshape(-1,1)))-1, 0, None)

# ======================== NN MODULES ========================
class SUE_MLP(nn.Module):
    def __init__(self, hidden=128, depth=5):
        super().__init__()
        dims = [1]+[hidden]*(depth-1)+[7]; layers=[]
        for k in range(len(dims)-2): layers += [nn.Linear(dims[k],dims[k+1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2],dims[-1])]
        self.net = nn.Sequential(*layers); self.soft = nn.Softplus(beta=2.0)
    def forward(self, x):
        y = self.soft(self.net(x)); S = torch.relu(1.0 - y.sum(1, keepdim=True))
        return torch.cat([S, y], dim=1)

class FTFourier(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, device=DEVICE))
        self.a = nn.Parameter(torch.zeros(k, device=DEVICE))
        self.b = nn.Parameter(torch.zeros(k, device=DEVICE)); self.k = k
    def forward(self, ts):
        t = ts.view(-1); g = self.bias.expand_as(t)
        for k in range(1, self.k+1):
            g = g + self.a[k-1]*torch.sin(2*math.pi*k*t) + self.b[k-1]*torch.cos(2*math.pi*k*t)
        return _bounded(g.view(-1,1), F_BOUNDS[0], F_BOUNDS[1])

# ======================== PREPARE TENSORS ========================
def _prepare_series(df, sd_lag, device=DEVICE):
    n = len(df); t = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    t_scale = max(1.0, float(t[-1])); ts = (t/t_scale).requires_grad_(True)
    sd = torch.tensor(df["sd"].values, dtype=torch.float32, device=device).unsqueeze(1)
    sd_l = lag_tensor(sd, sd_lag).clamp(0,1)
    Ccum = torch.tensor(df["cases"].values, dtype=torch.float32, device=device).unsqueeze(1)
    Dcum = torch.tensor(df["deaths"].values, dtype=torch.float32, device=device).unsqueeze(1)
    return ts, t_scale, sd_l, Ccum, Dcum

@dataclass
class TrainCfg:
    max_epochs: int; lr: float; sd_lag: int
    rollout_extra: int = 0; validation_days: int = 14; patience_epochs: int = 500

# ======================== CORE TRAINING ========================
def train_sueihcdr_once(dsub, Npop, cfg, return_all=False, sd_future=None):
    dates = pd.to_datetime(dsub["date"]).to_numpy()
    q_np, n_q = make_quarter_ids(dates)
    q_ids_full = torch.tensor(q_np, dtype=torch.long, device=DEVICE)
    ts_full, t_scale, sd_l_full, Ccum_full, Dcum_full = _prepare_series(dsub, cfg.sd_lag)
    Npop_t = torch.tensor([float(Npop)], dtype=torch.float32, device=DEVICE)
    net = SUE_MLP(128, 5).to(DEVICE)

    # Alpha
    Dcum_np = Dcum_full.squeeze().detach().cpu().numpy()
    alpha_np = bayesian_alpha_series(dates, Dcum_np, ALPH_BOUNDS[0], ALPH_BOUNDS[1], 0.08, 0.995)
    alpha_t_full = torch.tensor(alpha_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    s_hist_full = ((alpha_t_full - ALPH_BOUNDS[0])/(ALPH_BOUNDS[1]-ALPH_BOUNDS[0]+1e-9)).clamp(0,1)

    # Variants
    V_prior_np, steps_np = make_variant_structs(dates, VARIANT_EVENTS)
    V_prior_full = torch.tensor(V_prior_np, dtype=torch.float32, device=DEVICE)
    steps_full = [torch.tensor(s, dtype=torch.float32, device=DEVICE) for s in steps_np]
    var_params = nn.ParameterList([nn.Parameter(torch.zeros(1, device=DEVICE)) for _ in VARIANT_EVENTS])

    params = {k: nn.Parameter(torch.tensor([0.0], device=DEVICE))
              for k in ["beta0","sigma","delta","zeta","epsi","m","c","omega","eta","rho"]}
    fmod = FTFourier(FOURIER_K).to(DEVICE)
    adam_params = list(net.parameters()) + list(params.values()) + list(fmod.parameters()) + list(var_params)
    opt = optim.Adam(adam_params, lr=cfg.lr)

    def unpack(ts_s, sd_s, q_s, s_s, V_s, steps_s):
        b0 = _bounded(params["beta0"],BETA0_BOUNDS[0],BETA0_BOUNDS[1])
        sg = _bounded(params["sigma"],SIGMA_BOUNDS[0],SIGMA_BOUNDS[1])
        de = _bounded(params["delta"],DELTA_BOUNDS[0],DELTA_BOUNDS[1])
        ze = _bounded(params["zeta"],ZETA_BOUNDS[0],ZETA_BOUNDS[1])
        ep = _bounded(params["epsi"],EPSI_BOUNDS[0],EPSI_BOUNDS[1])
        m  = _bounded(params["m"],M_BOUNDS[0],M_BOUNDS[1])
        c  = _bounded(params["c"],C_BOUNDS[0],C_BOUNDS[1])
        om = _bounded(params["omega"],OMEGA_BOUNDS[0],OMEGA_BOUNDS[1])
        et = _bounded(params["eta"],ETA_BOUNDS[0],ETA_BOUNDS[1])
        rho= _bounded(params["rho"],RHO_BOUNDS[0],RHO_BOUNDS[1])
        om_t = om * (1.0 - s_s)
        f_vec = fmod(ts_s)
        l2 = (fmod.a**2).sum() + (fmod.b**2).sum() + 0.01*(fmod.bias**2)
        df_d = torch.diff(f_vec.view(-1), prepend=f_vec.view(-1)[:1])
        f_reg = W_FOURIER_L2*l2 + W_F_VAR*torch.mean(df_d**2)
        lo, hi = VAR_BUMP_BOUNDS; V_ex = torch.ones_like(V_s)
        for (nm,_,mp,_), pr, st in zip(VARIANT_EVENTS, var_params, steps_s):
            ml = lo + (hi-lo)*torch.sigmoid(pr); V_ex = V_ex*(1.0 + (ml/max(1e-6,mp) - 1.0)*st)
        return b0,sg,de,ze,ep,m,c,om,et,om_t,f_vec,f_reg,V_s*V_ex,rho

    n = len(dsub); run_val = n >= cfg.validation_days + 21
    vi = n - cfg.validation_days if run_val else n

    ts_tr, sd_tr = ts_full[:vi], sd_l_full[:vi]
    Cc_tr, Dc_tr = Ccum_full[:vi], Dcum_full[:vi]
    s_tr, q_tr = s_hist_full[:vi], q_ids_full[:vi]
    V_tr = V_prior_full[:vi]; st_tr = [s[:vi] for s in steps_full]
    al_tr = alpha_t_full[:vi]

    ts_va, sd_va = ts_full[vi:], sd_l_full[vi:]
    Cc_va = Ccum_full[vi:]
    s_va, q_va = s_hist_full[vi:], q_ids_full[vi:]
    V_va = V_prior_full[vi:]; st_va = [s[vi:] for s in steps_full]

    best_vl = np.inf; best_ep = 0; pat = 0; best_sd = None; history = []

    for ep_i in range(1, cfg.max_epochs+1):
        net.train(); opt.zero_grad()
        y = net(ts_tr); S,U,E,I,H,C,D,R = [y[:,k:k+1] for k in range(8)]
        ones = torch.ones_like(ts_tr); sc = 1.0/t_scale
        dS,dU,dE,dI,dH,dC,dD,dR = [torch.autograd.grad(c, ts_tr, ones, create_graph=True)[0]*sc
                                     for c in [S,U,E,I,H,C,D,R]]
        b0,sg,de,ze,ep_p,m,c,om,et,om_t,fv,fr,Vt,rho = unpack(ts_tr,sd_tr,q_tr,s_tr,V_tr,st_tr)
        be = b0*Vt*torch.exp(-KSD*sd_tr); inf = be*S*I
        if ep_i <= 500: de = de.detach() + 0*de

        rS = dS + inf + al_tr*S - om_t*U - et*R
        rU = dU - al_tr*S + om_t*U
        rE = dE - inf + sg*E
        rI = dI - sg*E + de*I
        rH = dH - (1-m)*de*I + ze*H
        rC = dC - c*ze*H + ep_p*C
        rD = dD - fv*ep_p*C
        rR = dR - m*de*I - (1-c)*ze*H - (1-fv)*ep_p*C + et*R
        L_phys = sum(torch.mean(r**2) for r in [rS,rU,rE,rI,rH,rC,rD,rR])

        S_,U_,E_,I_,H_,C_,D_,R_ = [v*Npop_t for v in (S,U,E,I,H,C,D,R)]
        Cpd = rho*sg*E_; Cc_m = Cc_tr[:1] + torch.cumsum(Cpd, dim=0)

        def dp(cum):
            d = torch.clamp(cum[1:]-cum[:-1], min=0.0); return torch.cat([cum[:1]*0, d], dim=0)
        def wa(x, k=7):
            xp = torch.cat([x[:1].repeat(k-1,1), x], dim=0)
            w = torch.ones(1,1,k,device=x.device)/k
            return F.conv1d(xp.T.unsqueeze(0), w).squeeze().unsqueeze(1)

        Cw_d = wa(dp(Cc_tr)); Cw_m = wa(Cpd)
        L_wk = W_WEEK_CASES*F.smooth_l1_loss(torch.log1p(Cw_m), torch.log1p(Cw_d), beta=0.25)
        L_cm = W_CUM_CASES*F.mse_loss(torch.log1p(Cc_m), torch.log1p(Cc_tr))
        L_ic = W_IC_ANCHOR*F.mse_loss(torch.log1p(Cc_m[:1]), torch.log1p(Cc_tr[:1]))
        S0t = torch.clamp(1-Cc_tr[:1]/(Npop_t+1e-9), 0, 1)
        L_s0 = W_S0*F.mse_loss(S[:1], S0t)
        L_pop = W_POP*torch.mean(((S+U+E+I+H+C+D+R)-1)**2)
        L_pr = W_PRIOR*(0.05*_soft_prior(1/(sg+1e-9),PR_DUR_INCUB,PR_W_INCUB) +
                        0.05*_soft_prior(1/(de+1e-9),PR_DUR_INF,PR_W_INF) +
                        0.05*_soft_prior(1/(ze+1e-9),PR_DUR_WARD,PR_W_WARD) +
                        0.05*_soft_prior(1/(ep_p+1e-9),PR_DUR_ICU,PR_W_ICU)).squeeze()
        Rt_e = (b0*Vt*torch.exp(-KSD*sd_tr))/(de+1e-9) * S  # effective Rt includes S
        L_rt = 0.5*torch.mean(F.relu(Rt_e-4)**2 + F.relu(0.6-Rt_e)**2)
        # Soft S-floor: gently penalize S dropping below 0.3 in first year (~365 days)
        n_floor = min(365, S.shape[0])
        L_sfloor = 5.0*torch.mean(F.relu(0.25 - S[:n_floor]))
        L_end = 10.0*F.mse_loss(torch.log1p(Cc_m[-1:]), torch.log1p(Cc_tr[-1:]))
        rho_pr = 0.01*((rho-0.4)**2).mean()
        bp_loss = 0.0
        if VAR_PRIOR_STRENGTH > 0:
            lo,hi = VAR_BUMP_BOUNDS
            for (_,_,mp,_),pr in zip(VARIANT_EVENTS, var_params):
                ml = lo+(hi-lo)*torch.sigmoid(pr)
                bp_loss += (torch.log(ml+1e-9)-math.log(mp+1e-9))**2
            bp_loss *= VAR_PRIOR_STRENGTH
        wp = 0.0 if ep_i <= 300 else W_PHYS
        loss = L_wk+L_cm+L_end+wp*L_phys+L_ic+L_pr+fr.squeeze()+L_s0+L_pop+L_rt+bp_loss+rho_pr+L_sfloor
        loss.backward(); opt.step()

        # Validation
        if run_val and ep_i % VAL_CHECK_FREQ == 0:
            net.eval()
            with torch.no_grad():
                yv = net(ts_va); Ev = yv[:,2:3]  # E is index 2 (S=0,U=1,E=2,I=3,...)
                sv = _bounded(params["sigma"],SIGMA_BOUNDS[0],SIGMA_BOUNDS[1])
                rv = _bounded(params["rho"],RHO_BOUNDS[0],RHO_BOUNDS[1])
                Cpd_v = rv*sv*(Ev*Npop_t)
                # Anchor validation cumulative to the END of the training period
                Cpd_t = (rho*sg*(E*Npop_t)).detach()
                Ccases_val_anchor = Cc_tr[-1:].detach()
                Cc_v = Ccases_val_anchor + torch.cumsum(Cpd_v, dim=0)
                vl = float(F.mse_loss(torch.log1p(Cc_v), torch.log1p(Cc_va)).item())
                if vl < best_vl:
                    best_vl = vl; best_ep = ep_i; pat = 0; best_sd = copy.deepcopy(net.state_dict())
                else:
                    pat += 1
                    if pat*VAL_CHECK_FREQ > cfg.patience_epochs:
                        print(f"  Early stop ep {ep_i}, best={best_ep}"); break
        if ep_i == 1 or ep_i % 500 == 0:
            print(f"  ep {ep_i}/{cfg.max_epochs} loss={float(loss):.4f}")

    if run_val and best_sd: net.load_state_dict(best_sd)

    # Rollout
    with torch.no_grad():
        b0f,sgf,def_,zef,epf,mf,cf,omf,etf,_,_,_,_,rhof = unpack(ts_full,sd_l_full,q_ids_full,s_hist_full,V_prior_full,steps_full)
        n_loc = len(dsub); n_ext = cfg.rollout_extra
        t_all = torch.arange(n_loc+n_ext, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        ts_all = t_all/t_scale
        dates_all = pd.date_range(start=pd.to_datetime(dsub["date"].iloc[0]), periods=len(t_all), freq="D").to_numpy()
        Vp_a, stn_a = make_variant_structs(dates_all, VARIANT_EVENTS)
        Vp_at = torch.tensor(Vp_a, dtype=torch.float32, device=DEVICE)
        st_at = [torch.tensor(s, dtype=torch.float32, device=DEVICE) for s in stn_a]

        sd_h = torch.tensor(dsub['sd'].values, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        if n_ext > 0:
            sdt = torch.tensor(sd_future, dtype=torch.float32, device=DEVICE).unsqueeze(1) if sd_future is not None else sd_h[-1:].repeat(n_ext,1)
            sd_a = torch.cat([sd_h, sdt[:n_ext]], dim=0)
        else: sd_a = sd_h
        sd_a = lag_tensor(sd_a, cfg.sd_lag).clamp(0,1)

        if n_ext > 0:
            af = project_alpha_future(alpha_np, n_ext)
            at = torch.tensor(af, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            al_a = torch.cat([alpha_t_full, at], dim=0).clamp(ALPH_BOUNDS[0], ALPH_BOUNDS[1])
        else: al_a = alpha_t_full

        y_a = net(ts_all)
        S_a,U_a,E_a,I_a,H_a,C_a,D_a,R_a = [y_a[:,k:k+1] for k in range(8)]
        S_n,U_n,E_n,I_n,H_n,C_n,D_n,R_n = [v*Npop_t for v in (S_a,U_a,E_a,I_a,H_a,C_a,D_a,R_a)]

        # Extend q_ids for extra rollout days (use last quarter id)
        if n_ext > 0:
            q_ext = q_ids_full[-1:].repeat(n_ext)
            q_ids_all = torch.cat([q_ids_full, q_ext], dim=0)
        else:
            q_ids_all = q_ids_full

        # Build s_hist for full + extra
        s_hist_a = ((al_a-ALPH_BOUNDS[0])/(ALPH_BOUNDS[1]-ALPH_BOUNDS[0]+1e-9)).clamp(0,1)

        _,_,_,_,_,_,_,_,_,_,_,_,Va,_ = unpack(ts_all, sd_a, q_ids_all,
            s_hist_a, Vp_at, st_at)

        comps = {k: v.squeeze().cpu().numpy() for k,v in
                 zip(["S","U","E","I","H","C","D","R"],[S_n,U_n,E_n,I_n,H_n,C_n,D_n,R_n])}
        inc = float(rhof.item())*sgf.item()*comps['E']
        C0 = float(dsub["cases"].values[0]) if len(dsub) else 0.0
        Ccum_all = C0 + np.cumsum(inc)
        be_t = (b0f.squeeze()*Va.squeeze())*torch.exp(-KSD*sd_a.squeeze())
        Rt_a = ((be_t/(def_+1e-9))*S_a.squeeze()).clamp(min=0).cpu().numpy()
        fs = fmod(ts_all).squeeze().cpu().numpy()

    lo,hi = VAR_BUMP_BOUNDS; lvm = {}
    with torch.no_grad():
        for i,(nm,d0,mp,rp) in enumerate(VARIANT_EVENTS):
            ml = lo+(hi-lo)*torch.sigmoid(var_params[i]).item()
            lvm[nm] = {'prior':mp,'learned':ml,'ratio':ml/mp}

    return {
        "Ccum": Ccum_all, "Dcum": comps['D'], "beta_eff": be_t.cpu().numpy(),
        "Rt": Rt_a, "Rt_paper": Rt_a, "f_series": fs,
        "alpha_series": al_a.squeeze().cpu().numpy(), "sd_used": sd_a.squeeze().cpu().numpy(),
        "history": history, "n_quarters": int(n_q), "comps": comps, "t_scale": float(t_scale),
        "S": S_a.squeeze().cpu().numpy(), "beta_base": float(b0f.item()), "delta": float(def_.item()),
        "beta0_final": float(b0f), "sigma_final": float(sgf), "delta_final": float(def_),
        "zeta_final": float(zef), "epsi_final": float(epf), "m_final": float(mf),
        "c_final": float(cf), "omega_final": float(omf), "eta_final": float(etf),
        "rho_final": float(rhof),
        "variant_multiplier": Va.squeeze().cpu().numpy(), "variant_events": VARIANT_EVENTS,
        "learned_variant_multipliers": lvm,
    }

# ======================== ENSEMBLE ========================
def train_ensemble(dsub, Npop, cfg, num_models=5, **kwargs):
    preds = []; last = None
    kw = {k:v for k,v in kwargs.items() if k != 'trainer_func'}
    for i in range(num_models):
        torch.manual_seed(i*100+42)
        r = train_sueihcdr_once(dsub, Npop, cfg, **kw); last = r; preds.append(r["Ccum"])
    stack = np.vstack(preds); res = copy.deepcopy(last)
    res["Ccum"] = np.median(stack, axis=0)
    return res

# ======================== DIAGNOSTICS ========================
def export_pinn_diagnostics(df_h, res, Npop, outdir, prefix="diag"):
    dates = pd.to_datetime(df_h["date"].values); T = len(df_h)
    cum_obs = df_h["cases"].values[:T].astype(float)
    cum_fit = np.asarray(res["Ccum"][:T], float)
    fig, ax = plt.subplots(1,1,figsize=(11.5,3.8))
    ax.plot(dates, cum_obs, label="Observed (cum)", color="k")
    ax.plot(dates, cum_fit, label="PINN fit (cum)", ls="--")
    ax.set_ylabel("Cumulative cases"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(outdir/f"{prefix}_cum_cases_fit.png", dpi=220); plt.close(fig)

    o7 = weekly_avg_np(daily_from_cum_np(cum_obs)); f7 = weekly_avg_np(daily_from_cum_np(cum_fit))
    fig, ax = plt.subplots(1,1,figsize=(11.5,3.8))
    ax.plot(dates[1:], o7[1:], "k-", lw=2, label="Observed (7d)")
    ax.plot(dates[1:], f7[1:], "-", lw=2, label="PINN (7d)")
    ax.set_ylabel("Daily cases (7d avg)"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(outdir/f"{prefix}_daily_cases.png", dpi=220); plt.close(fig)

    Rt = np.asarray(res["Rt"][:T], float).reshape(-1)
    fig, ax = plt.subplots(1,1,figsize=(11.5,3.8))
    ax.plot(dates, Rt, lw=2, label=r"$R_t$"); ax.axhline(1, color="k", ls=":", alpha=0.6)
    ax.set_ylabel(r"$R_t$"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(outdir/f"{prefix}_Rt.png", dpi=220); plt.close(fig)

    if "comps" in res:
        fig, ax = plt.subplots(1,1,figsize=(12,5))
        for k in ["S","U","E","I","H","C","D","R"]:
            if k in res["comps"]:
                ax.plot(dates, np.asarray(res["comps"][k][:T],float)/Npop, lw=1.6, label=k)
        ax.set_ylim(0,1); ax.set_ylabel("Fraction"); ax.legend(ncol=4); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(outdir/f"{prefix}_compartments_frac.png", dpi=220); plt.close(fig)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_publication_figures.py
===========================
Generates two publication figures from PINN model outputs:

  Figure 10: Inferred Compartment Dynamics (5-panel: A-E)
     A. Susceptible & Recovered (stacked area, fraction)
     B. Active Infection (E, I in counts)
     C. Protected/Behavioral (U in counts)
     D. Clinical Compartments (H, C, D in counts)
     E. Effective Reproduction Number Rt

  Figure 11: Time-varying Protection, Transmission & Variants (2-panel)
     Top: Behavioral protection α(t) vs mortality signal
     Bottom: Effective transmission β_eff(t) with variant ramps & SD overlay

Usage:
  1. Run your model:  python run_cities.py --mode fit_check --cities SanDiego
  2. Then:            python plot_publication_figures.py

  OR: call plot_compartment_dynamics() and plot_alpha_beta() directly
      from your runner after train_full_and_export().
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from pathlib import Path

# ======================== CONFIGURATION ========================
# Update this to your actual output path
BASE_PATH = Path(os.environ.get("PINN_DATA_PATH", "."))
DEFAULT_CITY = "SanDiego"

# Publication font settings
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 14,
    "font.family": "sans-serif",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

VARIANT_EVENTS = [
    ("Delta",   "2021-06-15", 1.60, 10),
    ("Omicron", "2021-12-15", 3.50,  5),
    ("BA.5",    "2022-06-15", 1.30, 10),
    ("XBB",     "2022-12-15", 1.20, 10),
]

VARIANT_COLORS = {
    "Delta": "#66BB6A", "Omicron": "#EF5350",
    "BA.5": "#AB47BC", "XBB": "#FF7043"
}


# ======================== FIGURE 10: COMPARTMENT DYNAMICS (5-PANEL) ========================
def plot_compartment_dynamics(dates, comps, Rt, Npop, outpath, city_label="San Diego"):
    """
    5-panel figure: A. S+R stacked area, B. E+I, C. U, D. H+C+D, E. Rt
    
    Parameters:
        dates:  array of datetime
        comps:  dict with keys S,U,E,I,H,C,D,R (in COUNTS, not fractions)
        Rt:     array of Rt values
        Npop:   population
        outpath: save path
    """
    dates = pd.to_datetime(dates)
    T = len(dates)
    
    # Ensure arrays are the right length
    S = np.asarray(comps["S"][:T], float)
    U = np.asarray(comps["U"][:T], float)
    E = np.asarray(comps["E"][:T], float)
    I = np.asarray(comps["I"][:T], float)
    H = np.asarray(comps["H"][:T], float)
    C = np.asarray(comps["C"][:T], float)
    D = np.asarray(comps["D"][:T], float)
    R = np.asarray(comps["R"][:T], float)
    Rt = np.asarray(Rt[:T], float)
    
    # Fractions for panel A
    S_frac = S / Npop
    R_frac = R / Npop
    
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.30, wspace=0.28)
    
    # --- Panel A: Susceptible & Recovered (stacked area, fraction) ---
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.fill_between(dates, 0, S_frac, color="#5B9BD5", alpha=0.85, label="S (Susceptible)")
    ax_a.fill_between(dates, S_frac, S_frac + R_frac, color="#A5A5A5", alpha=0.7, label="R (Recovered)")
    ax_a.set_ylabel("Fraction of population", fontsize=14)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title("A. Susceptible and Recovered Compartments", fontsize=15, fontweight="bold")
    ax_a.legend(loc="center right", fontsize=12, framealpha=0.9)
    ax_a.grid(alpha=0.2)
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # --- Panel B: Active Infection (E, I in counts) ---
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(dates, E, color="#C00000", lw=2, label="E (Exposed)")
    ax_b.plot(dates, I, color="#70AD47", lw=2, label="I (Infectious)")
    ax_b.set_ylabel("Individuals", fontsize=14)
    ax_b.set_title("B. Active Infection Compartments", fontsize=15, fontweight="bold")
    ax_b.legend(fontsize=12, framealpha=0.9)
    ax_b.grid(alpha=0.2)
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_b.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # --- Panel C: Protected/Behavioral (U) ---
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.fill_between(dates, 0, U, color="#ED7D31", alpha=0.7)
    ax_c.plot(dates, U, color="#ED7D31", lw=1.5, label="U (Protected)")
    ax_c.set_ylabel("Individuals", fontsize=14)
    ax_c.set_title("C. Protected (Behavioral) Compartment", fontsize=15, fontweight="bold")
    ax_c.legend(fontsize=12, framealpha=0.9)
    ax_c.grid(alpha=0.2)
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_c.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # --- Panel D: Clinical Compartments (H, C, D) ---
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(dates, H, color="#7030A0", lw=2, label="H (Hospitalized)")
    ax_d.plot(dates, C, color="#BF8F00", lw=2, label="C (Critical/ICU)")
    ax_d.plot(dates, D, color="#F4B6C2", lw=2, label="D (Deceased)")
    ax_d.set_ylabel("Individuals", fontsize=14)
    ax_d.set_title("D. Clinical Compartments", fontsize=15, fontweight="bold")
    ax_d.legend(fontsize=12, framealpha=0.9)
    ax_d.grid(alpha=0.2)
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax_d.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    # --- Panel E: Rt (spans full width) ---
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.plot(dates, Rt, color="#1565C0", lw=2.2, label=r"$R_t$")
    ax_e.axhline(1.0, color="black", ls=":", alpha=0.6, lw=1.2)
    ax_e.set_ylabel(r"$R_t$", fontsize=15)
    ax_e.set_xlabel("")
    ax_e.set_title("E. Effective Reproduction Number", fontsize=15, fontweight="bold")
    ax_e.legend(fontsize=13, framealpha=0.9)
    ax_e.grid(alpha=0.2)
    ax_e.set_ylim(0, max(4.0, np.percentile(Rt[np.isfinite(Rt)], 99) * 1.15))
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_e.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=0)
    
    fig.suptitle(f"Inferred Compartment Dynamics from SUEIHCDR-PINN ({city_label})",
                 fontsize=18, fontweight="bold", y=0.995)
    
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ======================== FIGURE 11: ALPHA & BETA_EFF ========================
def plot_alpha_beta(dates, alpha_series, beta_eff, sd_used, variant_mult,
                    deaths_cum, Npop, outpath, city_label="San Diego"):
    """
    2-panel figure:
      Top: α(t) behavioral protection signal + scaled death rate overlay
      Bottom: β_eff(t) with variant emergence annotations + SD overlay
    
    Parameters:
        dates:         array of datetime
        alpha_series:  array, behavioral protection α(t)
        beta_eff:      array, effective transmission rate
        sd_used:       array, social distancing covariate (lagged)
        variant_mult:  array, cumulative variant multiplier V(t)
        deaths_cum:    array, cumulative deaths (for mortality signal)
        Npop:          population
        outpath:       save path
    """
    dates = pd.to_datetime(dates)
    T = len(dates)
    
    alpha = np.asarray(alpha_series[:T], float)
    beff  = np.asarray(beta_eff[:T], float)
    sd    = np.asarray(sd_used[:T], float)
    vmult = np.asarray(variant_mult[:T], float)
    
    # Compute daily deaths (7d smoothed) for overlay
    dcum = np.asarray(deaths_cum[:T], float)
    ddaily = np.clip(np.diff(np.r_[dcum[0], dcum]), 0, None)
    ddaily_7d = pd.Series(ddaily).rolling(7, min_periods=1).mean().values
    # Normalize to [0, max_alpha] for visual overlay
    dmax = np.percentile(ddaily_7d, 99) if ddaily_7d.max() > 0 else 1.0
    ddaily_scaled = ddaily_7d / dmax * np.percentile(alpha, 95)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                    gridspec_kw={"hspace": 0.15})
    
    # --- TOP PANEL: α(t) ---
    ax1.plot(dates, alpha, color="#1565C0", lw=2.2, label=r"$\alpha(t)$ (protection adoption rate)")
    ax1.fill_between(dates, 0, alpha, color="#1565C0", alpha=0.12)
    
    # Death rate overlay (secondary y-axis)
    ax1r = ax1.twinx()
    ax1r.fill_between(dates, 0, ddaily_7d, color="#C62828", alpha=0.15, label="Daily deaths (7d avg)")
    ax1r.plot(dates, ddaily_7d, color="#C62828", lw=1.2, alpha=0.6)
    ax1r.set_ylabel("Daily deaths (7-day avg)", fontsize=13, color="#C62828")
    ax1r.tick_params(axis="y", labelcolor="#C62828", labelsize=11)
    ax1r.spines["right"].set_visible(True)
    ax1r.spines["right"].set_color("#C62828")
    
    ax1.set_ylabel(r"$\alpha(t)$ (day$^{-1}$)", fontsize=14, color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_title("Behavioral Protection Signal", fontsize=16, fontweight="bold")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc="upper right", framealpha=0.9)
    ax1.grid(alpha=0.15)
    
    # --- BOTTOM PANEL: β_eff(t) ---
    ax2.plot(dates, beff, color="#1565C0", lw=2.2, label=r"$\beta_{\rm eff}(t)$")
    
    # SD overlay (secondary y-axis)
    ax2r = ax2.twinx()
    ax2r.fill_between(dates, 0, sd, color="#FFA726", alpha=0.2, label="Social distancing (SD)")
    ax2r.plot(dates, sd, color="#FFA726", lw=1.2, alpha=0.6)
    ax2r.set_ylabel("Social distancing index", fontsize=13, color="#E65100")
    ax2r.tick_params(axis="y", labelcolor="#E65100", labelsize=11)
    ax2r.set_ylim(0, 1.0)
    ax2r.spines["right"].set_visible(True)
    ax2r.spines["right"].set_color("#E65100")
    
    # Variant annotations
    for vname, vdate, vprior, vramp in VARIANT_EVENTS:
        vt = pd.Timestamp(vdate)
        if vt >= dates.min() and vt <= dates.max():
            ax2.axvline(vt, color=VARIANT_COLORS.get(vname, "gray"),
                       ls="--", lw=1.8, alpha=0.7)
            # Place label at top of panel
            ypos = ax2.get_ylim()[1] * 0.92 if ax2.get_ylim()[1] > 0 else 0.5
            ax2.annotate(vname, xy=(vt, beff[np.searchsorted(dates, vt)] if np.searchsorted(dates, vt) < len(beff) else beff[-1]),
                        xytext=(10, 15), textcoords="offset points",
                        fontsize=12, fontweight="bold",
                        color=VARIANT_COLORS.get(vname, "gray"),
                        arrowprops=dict(arrowstyle="-", color=VARIANT_COLORS.get(vname, "gray"), lw=1.2))
    
    ax2.set_ylabel(r"$\beta_{\rm eff}(t)$ (day$^{-1}$)", fontsize=14, color="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#1565C0")
    ax2.set_title("Effective Transmission Rate with Variant Emergence", fontsize=16, fontweight="bold")
    
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2r.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, fontsize=12, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.15)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=0)
    
    fig.suptitle(f"Time-Varying Protection, Transmission, and Variant-Driven Changes ({city_label})",
                 fontsize=17, fontweight="bold", y=1.01)
    
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ======================== INTEGRATION FUNCTION ========================
def plot_all_publication_figures(df, Npop, res, outdir, city_label="San Diego"):
    """
    Call this after train_full_and_export() with the returned result dict.
    Generates both Figure 10 and Figure 11.
    
    Usage in your runner:
        res = train_full_and_export(df, Npop, outdir)
        plot_all_publication_figures(df, Npop, res, outdir, city_label="SanDiego")
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(df["date"].values)
    T = len(df)
    
    # Figure 10: Compartment dynamics
    plot_compartment_dynamics(
        dates=dates,
        comps=res["comps"],
        Rt=res["Rt"],
        Npop=Npop,
        outpath=outdir / "fig10_compartment_dynamics.png",
        city_label=city_label
    )
    
    # Figure 11: Alpha & beta_eff
    plot_alpha_beta(
        dates=dates,
        alpha_series=res["alpha_series"],
        beta_eff=res["beta_eff"],
        sd_used=res["sd_used"],
        variant_mult=res["variant_multiplier"],
        deaths_cum=df["deaths"].values,
        Npop=Npop,
        outpath=outdir / "fig11_alpha_beta_eff.png",
        city_label=city_label
    )


# ======================== STANDALONE USAGE ========================
def load_and_plot_from_saved(city_label=DEFAULT_CITY):
    """
    Load previously saved compartment data and plot.
    Requires compartments_counts.csv and parameters_final.json in the output dir.
    """
    outdir = BASE_PATH / f"outputs_SUEIHCDR_PUBLICATION_v2_{city_label}"
    comp_file = outdir / "compartments_counts.csv"
    param_file = outdir / "parameters_final.json"
    
    if not comp_file.exists():
        print(f"ERROR: {comp_file} not found. Run the model first.")
        return
    
    import json
    df_comp = pd.read_csv(comp_file, parse_dates=["date"])
    params = json.load(open(param_file))
    
    # For standalone mode, we need to also load the original data
    # to get deaths and sd. This is a simplified version.
    print(f"Loaded {len(df_comp)} rows from {comp_file}")
    print(f"NOTE: For Figure 11 (alpha/beta), you need to call plot_all_publication_figures()")
    print(f"      directly from your model runner, which has the full result dict.")
    
    # We can still plot Figure 10 from saved compartment data
    T = len(df_comp)
    dates = df_comp["date"].values
    
    # Need Npop for fractions
    from run_cities import US_CITIES, load_us_county_series, CITY_POP
    try:
        # Try US first
        match = [c for c in US_CITIES if c[0] == city_label]
        if match:
            _, Npop = load_us_county_series(match[0][1], match[0][2], match[0][3])
        else:
            Npop = CITY_POP.get(city_label, 3.3e6)
    except:
        Npop = 3.3e6
    
    comps = {k: df_comp[k].values for k in ["S","U","E","I","H","C","D","R"]}
    
    # Rt needs recomputation — use a simple proxy if not saved
    # For proper Rt, call from runner
    print("WARNING: Rt not available in saved data. Using placeholder.")
    Rt = np.ones(T)
    
    plot_compartment_dynamics(dates, comps, Rt, Npop, 
                            outdir / "fig10_compartment_dynamics.png", city_label)



# ============================================================
# ADD these two helper functions ABOVE train_full_and_export()
# (or anywhere in run_cities_model.py before it's called):
# ============================================================

def _plot_fig10_compartments(df, res, Npop, outdir):
    """Publication Figure 10: 5-panel compartment dynamics (A–E)."""
    import matplotlib.gridspec as gridspec

    dates = pd.to_datetime(df["date"].values)
    T = len(df)
    S = np.asarray(res["comps"]["S"][:T], float)
    U = np.asarray(res["comps"]["U"][:T], float)
    E = np.asarray(res["comps"]["E"][:T], float)
    I = np.asarray(res["comps"]["I"][:T], float)
    H = np.asarray(res["comps"]["H"][:T], float)
    C = np.asarray(res["comps"]["C"][:T], float)
    D = np.asarray(res["comps"]["D"][:T], float)
    R = np.asarray(res["comps"]["R"][:T], float)
    Rt = np.asarray(res["Rt"][:T], float)

    S_frac = S / Npop
    R_frac = R / Npop

    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.30, wspace=0.28)

    # A. Susceptible & Recovered (stacked area)
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(dates, 0, S_frac, color="#5B9BD5", alpha=0.85, label="S (Susceptible)")
    ax.fill_between(dates, S_frac, S_frac + R_frac, color="#A5A5A5", alpha=0.7, label="R (Recovered)")
    ax.set_ylabel("Fraction of population", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_title("A. Susceptible and Recovered Compartments", fontsize=15, fontweight="bold")
    ax.legend(loc="center right", fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # B. Active Infection (E, I)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(dates, E, color="#C00000", lw=2, label="E (Exposed)")
    ax.plot(dates, I, color="#70AD47", lw=2, label="I (Infectious)")
    ax.set_ylabel("Individuals", fontsize=14)
    ax.set_title("B. Active Infection Compartments", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # C. Protected (U)
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_between(dates, 0, U, color="#ED7D31", alpha=0.7)
    ax.plot(dates, U, color="#ED7D31", lw=1.5, label="U (Protected)")
    ax.set_ylabel("Individuals", fontsize=14)
    ax.set_title("C. Protected (Behavioral) Compartment", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # D. Clinical (H, C, D)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(dates, H, color="#7030A0", lw=2, label="H (Hospitalized)")
    ax.plot(dates, C, color="#BF8F00", lw=2, label="C (Critical/ICU)")
    ax.plot(dates, D, color="#F4B6C2", lw=2, label="D (Deceased)")
    ax.set_ylabel("Individuals", fontsize=14)
    ax.set_title("D. Clinical Compartments", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # E. Rt (full width)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(dates, Rt, color="#1565C0", lw=2.2, label=r"$R_t$")
    ax.axhline(1.0, color="black", ls=":", alpha=0.6, lw=1.2)
    ax.set_ylabel(r"$R_t$", fontsize=15)
    ax.set_title("E. Effective Reproduction Number", fontsize=15, fontweight="bold")
    ax.legend(fontsize=13, framealpha=0.9)
    ax.grid(alpha=0.2)
    rt_finite = Rt[np.isfinite(Rt)]
    ax.set_ylim(0, max(4.0, np.percentile(rt_finite, 99) * 1.15) if len(rt_finite) > 0 else 4.0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=0)

    fig.suptitle("Inferred Compartment Dynamics from SUEIHCDR-PINN",
                 fontsize=18, fontweight="bold", y=0.995)
    fig.savefig(Path(outdir)/"fig10_compartment_dynamics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig10_compartment_dynamics.png")


def _plot_fig11_alpha_beta(df, res, Npop, outdir):
    """Publication Figure 11: α(t) and β_eff(t) with variants + SD."""
    dates = pd.to_datetime(df["date"].values)
    T = len(df)

    alpha = np.asarray(res["alpha_series"][:T], float)
    beff  = np.asarray(res["beta_eff"][:T], float)
    sd    = np.asarray(res["sd_used"][:T], float)

    # Daily deaths (7d smoothed) for overlay
    dcum = df["deaths"].values[:T].astype(float)
    ddaily = np.clip(np.diff(np.r_[dcum[0], dcum]), 0, None)
    ddaily_7d = pd.Series(ddaily).rolling(7, min_periods=1).mean().values

    VCOLS = {"Delta":"#66BB6A","Omicron":"#EF5350","BA.5":"#AB47BC","XBB":"#FF7043"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                    gridspec_kw={"hspace": 0.18})

    # --- TOP: α(t) ---
    ax1.plot(dates, alpha, color="#1565C0", lw=2.2, label=r"$\alpha(t)$ (protection adoption)")
    ax1.fill_between(dates, 0, alpha, color="#1565C0", alpha=0.12)
    ax1r = ax1.twinx()
    ax1r.fill_between(dates, 0, ddaily_7d, color="#C62828", alpha=0.15, label="Daily deaths (7d avg)")
    ax1r.plot(dates, ddaily_7d, color="#C62828", lw=1.2, alpha=0.6)
    ax1r.set_ylabel("Daily deaths (7-day avg)", fontsize=13, color="#C62828")
    ax1r.tick_params(axis="y", labelcolor="#C62828", labelsize=11)
    ax1r.spines["right"].set_visible(True); ax1r.spines["right"].set_color("#C62828")
    ax1.set_ylabel(r"$\alpha(t)$ (day$^{-1}$)", fontsize=14, color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_title("Behavioral Protection Signal", fontsize=16, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=12, loc="upper right", framealpha=0.9)
    ax1.grid(alpha=0.15)

    # --- BOTTOM: β_eff(t) ---
    ax2.plot(dates, beff, color="#1565C0", lw=2.2, label=r"$\beta_{\rm eff}(t)$")
    ax2r = ax2.twinx()
    ax2r.fill_between(dates, 0, sd, color="#FFA726", alpha=0.2, label="Social distancing (SD)")
    ax2r.plot(dates, sd, color="#FFA726", lw=1.2, alpha=0.6)
    ax2r.set_ylabel("Social distancing index", fontsize=13, color="#E65100")
    ax2r.tick_params(axis="y", labelcolor="#E65100", labelsize=11)
    ax2r.set_ylim(0, 1.0)
    ax2r.spines["right"].set_visible(True); ax2r.spines["right"].set_color("#E65100")

    # Variant vertical lines + labels
    for vname, vdate, vprior, vramp in VARIANT_EVENTS:
        vt = pd.Timestamp(vdate)
        if vt >= dates.min() and vt <= dates.max():
            ax2.axvline(vt, color=VCOLS.get(vname,"gray"), ls="--", lw=1.8, alpha=0.7)
            # Find y position for label
            vidx = min(np.searchsorted(dates, vt), len(beff)-1)
            ylab = beff[vidx] + (ax2.get_ylim()[1] - beff[vidx]) * 0.3
            ax2.text(vt + pd.Timedelta(days=5), ylab, vname,
                     fontsize=13, fontweight="bold", color=VCOLS.get(vname,"gray"),
                     ha="left", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    ax2.set_ylabel(r"$\beta_{\rm eff}(t)$ (day$^{-1}$)", fontsize=14, color="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#1565C0")
    ax2.set_title("Effective Transmission Rate with Variant Emergence", fontsize=16, fontweight="bold")
    h3, l3 = ax2.get_legend_handles_labels()
    h4, l4 = ax2r.get_legend_handles_labels()
    ax2.legend(h3+h4, l3+l4, fontsize=12, loc="upper left", framealpha=0.9)
    ax2.grid(alpha=0.15)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate(rotation=0)

    fig.suptitle("Time-Varying Protection, Transmission, and Variant-Driven Changes",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.savefig(Path(outdir)/"fig11_alpha_beta_eff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig11_alpha_beta_eff.png")


# ============================================================
# ALSO ADD this helper that was missing:
# ============================================================

def _window_ymax(*arrays):
    """Compute a nice y-axis max from multiple arrays."""
    vals = []
    for a in arrays:
        a = np.asarray(a, float).ravel()
        a = a[np.isfinite(a)]
        if len(a) > 0:
            vals.append(np.percentile(a, 97))
    return max(vals) * 1.25 if vals else 100.0
def train_full_and_export(df, Npop, outdir, sd_lag_days=SD_LAG_DAYS):
    t0 = time.time()
    cfg = TrainCfg(max_epochs=EPOCHS_FULL, lr=LR_FULL, sd_lag=sd_lag_days,
                   validation_days=VALIDATION_DAYS, patience_epochs=PATIENCE_EPOCHS)
    res = train_sueihcdr_once(df, Npop, cfg, return_all=True, sd_future=None)
    T = len(df)

    #pd.DataFrame({
    #    "date": pd.to_datetime(df["date"]).astype(str),
    #    **{k: res["comps"][k][:T] for k in ["S","U","E","I","H","C","D","R"]},
    #    "cum_cases_pred": res["Ccum"][:T], "obs_cases": df["cases"].values,
    #}).to_csv(outdir/"compartments_counts.csv", index=False)


    # Save time-varying signals (NEW)
    pd.DataFrame({
        "date": pd.to_datetime(df["date"]).astype(str),
        "alpha_series": res["alpha_series"][:T],
        "beta_eff": res["beta_eff"][:T],
        "sd_used": res["sd_used"][:T],
        "variant_multiplier": res["variant_multiplier"][:T],
        "Rt": res["Rt"][:T],
    }).to_csv(outdir/"time_varying_signals.csv", index=False)

    # Publication figures
    _plot_fig10_compartments(df, res, Npop, outdir)
    _plot_fig11_alpha_beta(df, res, Npop, outdir)

    params_final = {k:v for k,v in res.items() if k.endswith('_final')}
    pd.Series(params_final).to_json(outdir/"parameters_final.json")
    export_pinn_diagnostics(df, res, Npop, outdir, prefix="full")

    o7 = weekly_avg_np(daily_from_cum_np(df["cases"].values[:T]))
    f7 = weekly_avg_np(daily_from_cum_np(res["Ccum"][:T]))
    mae = float(np.mean(np.abs(o7-f7))); mape = safe_mape(o7, f7)
    print(f"  FIT: MAE(7d)={mae:.1f}, MAPE(7d)={mape:.1f}% | {time.time()-t0:.1f}s → {outdir}")

    return res

# ======================== MULTI-WINDOW EVAL ========================
def tiny_validate_sd_lag(df, Npop, cut_date):
    best = None
    for sd_l in SD_LAG_GRID:
        cfg = TrainCfg(max_epochs=EPOCHS_TINY, lr=LR_FULL, sd_lag=sd_l, validation_days=7, patience_epochs=500)
        dsub = df[df["date"] <= cut_date-pd.Timedelta(days=7)].reset_index(drop=True)
        if len(dsub) < 21: continue
        try:
            r = train_sueihcdr_once(dsub, Npop, cfg, return_all=True)
            o7 = weekly_avg_np(daily_from_cum_np(dsub["cases"].values))[-7:]
            p7 = weekly_avg_np(daily_from_cum_np(r["Ccum"][:len(dsub)]))[-7:]
            mae = float(np.mean(np.abs(p7-o7))) if len(o7)==len(p7) and len(o7)>0 else np.inf
            if best is None or mae < best[0]: best = (mae, sd_l)
        except: pass
    return best[1] if best else 7

# ======================== EVALUATION & PLOTTING ========================
def generate_plots_and_metrics(full_res, df, Npop, cut_date, horizon, window_name="", lookback_days=180, outdir=None):
    if not np.issubdtype(df["date"].dtype, np.datetime64): df["date"] = pd.to_datetime(df["date"])
    cut_idx_full = get_cut_idx_full(df, cut_date)
    tr_start = pd.Timestamp(cut_date) - pd.Timedelta(days=lookback_days)
    dsub = df[(df["date"] <= pd.Timestamp(cut_date)) & (df["date"] >= tr_start)].reset_index(drop=True)
    idx_cut_slice = len(dsub)

    obs_cum = df["cases"].values[:cut_idx_full + 1]
    obs_daily_hist = daily_from_cum_np(obs_cum)
    obs7_hist = weekly_avg_np(obs_daily_hist)
    last_obs7 = float(max(0.0, obs7_hist[-1])) if obs7_hist.size > 0 else 0.0

    # --- PINN ---
    pin_abs_future = full_res["Ccum"][idx_cut_slice: idx_cut_slice + horizon]
    last_model_cum = full_res["Ccum"][idx_cut_slice - 1]
    pin_daily_raw = np.clip(np.diff(np.r_[last_model_cum, pin_abs_future]), 0, None)
    if len(pin_daily_raw) >= 7: pin_start_7d = np.mean(pin_daily_raw[:7])
    else: pin_start_7d = max(np.mean(pin_daily_raw), 1.0)
    s_pin = float(np.clip((last_obs7 / max(pin_start_7d, 1.0)), 0.5, 2.0))
    pin_daily = pin_daily_raw * s_pin

    # --- ARIMA ---
    ar_daily_smooth = forecast_daily_levels_from_cum(obs_cum, steps=horizon, recalib_window_days=lookback_days)
    ar_start_7d = max(ar_daily_smooth[0] if len(ar_daily_smooth) > 0 else 1.0, 1.0)
    s_ar = float(np.clip(last_obs7 / ar_start_7d, 0.5, 2.0))
    ar_daily_smooth = ar_daily_smooth * s_ar

    # --- Hybrid ---
    rt_hist = np.asarray(full_res.get("Rt", []))[:idx_cut_slice]
    sd_hist_sliced = dsub["sd"].values[:len(rt_hist)]
    w_arima = gating_from_regime(rt_hist, sd_hist_sliced)
    hyb_daily = (1.0 - w_arima) * pin_daily + w_arima * ar_daily_smooth

    # --- Truth ---
    true_cum_future = df["cases"].values[cut_idx_full + 1: cut_idx_full + 1 + horizon]
    H = int(horizon)
    if not (len(pin_daily)==H and len(ar_daily_smooth)==H and len(hyb_daily)==H and len(true_cum_future)==H):
        print(f"Skipping {pd.Timestamp(cut_date).date()}: length mismatch"); return None

    pin_cum_fut = obs_cum[-1] + np.cumsum(pin_daily)
    ar_cum_fut = obs_cum[-1] + np.cumsum(ar_daily_smooth)
    hyb_cum_fut = obs_cum[-1] + np.cumsum(hyb_daily)

    pin_wavg = carry_wavg_future_from_cum(obs_cum, pin_cum_fut)
    ar_wavg = carry_wavg_future_from_cum(obs_cum, ar_cum_fut)
    hyb_wavg = carry_wavg_future_from_cum(obs_cum, hyb_cum_fut)
    true_wavg = carry_wavg_future_from_cum(obs_cum, true_cum_future)

    H_metric = min(len(pin_wavg), len(ar_wavg), len(hyb_wavg), len(true_wavg))
    if H_metric == 0: return None
    pin_wavg, ar_wavg, hyb_wavg, true_wavg = pin_wavg[:H_metric], ar_wavg[:H_metric], hyb_wavg[:H_metric], true_wavg[:H_metric]

    # --- Comparison Models ---
    obs7_series = pd.Series(obs7_hist, index=df["date"].iloc[:len(obs7_hist)])
    loglin_forecast = forecast_loglinear_7d(obs7_series, current_date=cut_date, history_days=21, horizon=H_metric)
    bayes_forecast = forecast_bayesian_poly_7d(obs7_series, current_date=cut_date, history_days=21, horizon=H_metric, degree=2)

    # --- Ensemble ---
    ens3_wavg = 0.5*pin_wavg + 0.3*ar_wavg + 0.2*bayes_forecast

    # --- Metrics ---
    metrics = {
        "w_arima": w_arima,
        "mae_pin": float(np.mean(np.abs(pin_wavg - true_wavg))),
        "mae_ari": float(np.mean(np.abs(ar_wavg - true_wavg))),
        "mae_hyb": float(np.mean(np.abs(hyb_wavg - true_wavg))),
        "mae_loglin": float(np.mean(np.abs(loglin_forecast - true_wavg))),
        "mae_bayesian": float(np.mean(np.abs(bayes_forecast - true_wavg))),
        "mae_ens3": float(np.mean(np.abs(ens3_wavg - true_wavg))),
        "mape_pin": safe_mape(true_wavg, pin_wavg),
        "mape_ari": safe_mape(true_wavg, ar_wavg),
        "mape_hyb": safe_mape(true_wavg, hyb_wavg),
        "mape_loglin": safe_mape(true_wavg, loglin_forecast),
        "mape_bayesian": safe_mape(true_wavg, bayes_forecast),
        "me_pin": float(np.mean(pin_wavg - true_wavg)),
        "me_ari": float(np.mean(ar_wavg - true_wavg)),
        "me_hyb": float(np.mean(hyb_wavg - true_wavg)),
    }
    print(f"    H={H}d: MAE_PINN={metrics['mae_pin']:.1f} MAE_ARIMA={metrics['mae_ari']:.1f} MAE_Hybrid={metrics['mae_hyb']:.1f}")

    # --- Plotting ---
    if outdir is not None:
        dates_all = df["date"].values
        xmin = pd.Timestamp(cut_date) - pd.Timedelta(days=21)
        xmax = pd.Timestamp(cut_date) + pd.Timedelta(days=horizon)
        obs7_all = weekly_avg_np(daily_from_cum_np(df["cases"].values))
        fut_dates_plot = pd.date_range(pd.Timestamp(cut_date), periods=len(pin_wavg)+1, freq="D")
        pin_plot = np.r_[last_obs7, pin_wavg]; ar_plot = np.r_[last_obs7, ar_wavg]
        ll_plot = np.r_[last_obs7, loglin_forecast]; bayes_plot = np.r_[last_obs7, bayes_forecast]

        fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))
        ax.plot(dates_all[1:], obs7_all[1:], "k-", lw=2, label="Observed (7d)")
        ax.plot(fut_dates_plot, ar_plot, "-", lw=2.5, marker="o", ms=6, label="ARIMA")
        ax.plot(fut_dates_plot, pin_plot, "--", lw=2.5, marker="s", ms=6, label="PINN (proposed)")
        ax.plot(fut_dates_plot, ll_plot, ":", lw=2.5, marker="x", ms=6, label="Log-Linear")
        ax.plot(fut_dates_plot, bayes_plot, "--", lw=2.5, marker="D", ms=6, label="Bayesian Poly")
        ax.axvspan(pd.Timestamp(cut_date), xmax, color="gray", alpha=0.1)
        ax.axvline(pd.Timestamp(cut_date), color="k", ls=":", alpha=0.9)
        ax.set_xlim([xmin, xmax])
        ctx_mask = (dates_all >= xmin) & (dates_all <= xmax)
        if np.sum(ctx_mask) > 0:
            ymax = _window_ymax(obs7_all[1:][ctx_mask[1:]], ar_wavg, pin_wavg, loglin_forecast, bayes_forecast)
            ax.set_ylim(0, ymax)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.set_title(f"Forecast: {window_name} | Cut: {pd.Timestamp(cut_date).date()}")
        ax.legend(frameon=False, ncol=3); fig.autofmt_xdate(); fig.tight_layout()
        fname = Path(outdir) / "regime_plots" / f"overlay_{pd.Timestamp(cut_date).date()}_{horizon}d.png"
        fname.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(fname, dpi=220); plt.close(fig)
        print(f"    Saved Plot -> {fname.name}")

    return metrics

def run_multi_window_eval(df, Npop, outdir):
    all_res = []; max_h = max(HORIZON_LIST)
    print(f"\n{'='*20} REGIME-BASED VALIDATION (5 windows) {'='*20}")
    for regime, cut_str, lb in VALIDATION_CONFIG:
        cut = pd.Timestamp(cut_str)
        print(f"  Target: {regime} ({cut.date()}) | Lookback: {lb}d")
        if (cut-pd.Timedelta(days=lb)) < df['date'].min(): print("    SKIP: before data start"); continue
        if cut > df['date'].max()-pd.Timedelta(days=max_h): print("    SKIP: insufficient future data"); continue
        try:
            sl = tiny_validate_sd_lag(df, Npop, cut)
            tr_s = max(pd.Timestamp(cut)-pd.Timedelta(days=lb), df['date'].min())
            dsub = df[(df["date"]<=pd.Timestamp(cut))&(df["date"]>=tr_s)].reset_index(drop=True)
            if len(dsub) < 60: print("    SKIP: <60 days"); continue
            sf = project_sd_future(dsub["sd"].values, max_h)
            cfg = TrainCfg(max_epochs=EPOCHS_MAX, lr=LR_FULL, sd_lag=sl,
                           rollout_extra=max_h, validation_days=VALIDATION_DAYS, patience_epochs=PATIENCE_EPOCHS)
            fr = train_ensemble(dsub, Npop, cfg, return_all=True, sd_future=sf)
            for H in HORIZON_LIST:
                metrics = generate_plots_and_metrics(fr, df, Npop, cut, H,
                    window_name=regime, lookback_days=lb, outdir=outdir)
                if metrics:
                    metrics["window"] = regime; metrics["cut_date"] = str(cut.date()); metrics["horizon"] = H
                    all_res.append(metrics)
        except Exception as e:
            print(f"    FAIL {cut.date()}: {e}")

    if all_res:
        pd.DataFrame(all_res).to_csv(outdir/"regime_validation_metrics.csv", index=False)
        print(f"  Saved → {outdir/'regime_validation_metrics.csv'}")

# ======================== MAIN DRIVER ========================
def run_single_city(label, df, Npop, mode="fit_check"):
    od_main = Path(OUT_PATH_STR)/f"outputs_SUEIHCDR_PUBLICATION_vpaper_{label}"
    od_multi = Path(OUT_PATH_STR)/f"pinn_sueihcdr_multiwindow_vpaper_{label}"
    od_main.mkdir(exist_ok=True, parents=True); od_multi.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"CITY: {label} | T={len(df)} | Pop={int(Npop):,} | {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    res = train_full_and_export(df, Npop, od_main)
    if "learned_variant_multipliers" in res:
        for nm, v in res["learned_variant_multipliers"].items():
            print(f"  Variant {nm}: prior={v['prior']:.2f}, learned={v['learned']:.3f}")
    if mode == "full":
        run_multi_window_eval(df, Npop, od_multi)
    return res



