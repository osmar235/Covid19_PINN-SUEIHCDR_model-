from .core import *

# ============================================================
# Nature Communications revision helpers (drop-in cell)
#   1) Patch alpha future projection to match ALPH_BOUNDS
#   2) Add stronger baselines (ETS/Theta/Naive)
#   3) Add scale-normalized metrics (MASE, per-100k MAE)
#   4) Combine existing per-city regime_validation_metrics.csv
#   5) Multiplicity correction (Holm) + city-clustered tests
#   6) Optional: ablation scaffolding (no/single waning; no physics)
# ============================================================

import os, re, math, warnings, json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None

# -----------------------------
# 0) Audit info (printed only when this module is run as a script,
#     not during normal imports)
# -----------------------------
def _print_audit_banner():
    """Print configuration audit info. Called from __main__, never on import."""
    print("ALPH_BOUNDS:", ALPH_BOUNDS)
    print("VAR_BUMP_BOUNDS:", VAR_BUMP_BOUNDS)
    print("VALIDATION_CONFIG:", VALIDATION_CONFIG)
    print("NOTE: This module patches project_alpha_future() to respect ALPH_BOUNDS.\n")


# -----------------------------
# 1) PATCH: alpha future projection
# -----------------------------
def project_alpha_future(alpha_hist, steps, mode="exp_decay", tau=7, floor=None, quantile_base=0.25):
    """
    Project α(t) forward (used during forecast rollout).

    IMPORTANT PATCH:
      - default floor is ALPH_BOUNDS[0] (NOT 0.05)
      - baseline is a low quantile of recent α history
      - output is clamped to ALPH_BOUNDS exactly
    """
    if steps <= 0:
        return np.array([], float)

    ah = np.asarray(alpha_hist, float)
    if ah.size == 0:
        last = float(ALPH_BOUNDS[0])
        base = float(ALPH_BOUNDS[0])
    else:
        last = float(ah[-1])
        if ah.size >= 21:
            base = float(np.quantile(ah[-21:], quantile_base))
        else:
            base = float(np.quantile(ah, quantile_base))

    if floor is None:
        floor = float(ALPH_BOUNDS[0])
    floor = float(np.clip(floor, ALPH_BOUNDS[0], ALPH_BOUNDS[1]))
    base  = float(np.clip(max(base, floor), ALPH_BOUNDS[0], ALPH_BOUNDS[1]))
    last  = float(np.clip(last, ALPH_BOUNDS[0], ALPH_BOUNDS[1]))

    t = np.arange(1, steps + 1, dtype=float)
    out = base + (last - base) * np.exp(-t / max(float(tau), 1.0))
    return np.clip(out, ALPH_BOUNDS[0], ALPH_BOUNDS[1]).astype(float)


# -----------------------------
# 2) Metrics helpers: per-capita, MASE, Holm correction, Wilson CI
# -----------------------------
def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # ~N(0,1) 97.5th percentile
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n) / denom
    return (max(0.0, center-half), min(1.0, center+half))

def holm_adjust(pvals):
    """
    Holm-Bonferroni adjustment.
    Returns adjusted p-values in original order.
    """
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, float)
    running_max = 0.0
    for j, idx in enumerate(order):
        p = pvals[idx]
        adj_p = (m - j) * p
        running_max = max(running_max, adj_p)
        adj[idx] = min(1.0, running_max)
    return adj

def mase_scale(y_train, eps=1e-9):
    """
    MASE denominator: mean absolute naive one-step change.
    Works on ANY scale (raw counts, 7d avg, etc).
    """
    y = np.asarray(y_train, float)
    if len(y) < 2:
        return np.nan
    denom = np.mean(np.abs(np.diff(y)))
    return max(denom, eps)

def per100k(mae, Npop):
    return float(mae) / (float(Npop) / 100_000.0)


# -----------------------------
# 3) Build observed 7d series and "true future" 7d values
#    (uses your existing daily_from_cum_np, weekly_avg_np, carry_wavg_future_from_cum)
# -----------------------------
def obs7_series_from_df(df):
    # observed cumulative up to end
    cum = df["cases"].values.astype(float)
    daily = daily_from_cum_np(cum)
    obs7 = weekly_avg_np(daily)
    dates = pd.to_datetime(df["date"].values)[:len(obs7)]
    return pd.Series(obs7, index=dates)

def true_future_7d_from_df(df, cut_date, horizon):
    cut_date = pd.Timestamp(cut_date)
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    cut_idx = get_cut_idx_full(df2, cut_date)

    obs_cum_hist = df2["cases"].values[:cut_idx+1].astype(float)
    true_cum_future = df2["cases"].values[cut_idx+1: cut_idx+1+horizon].astype(float)
    if len(true_cum_future) < horizon:
        return None
    return carry_wavg_future_from_cum(obs_cum_hist, true_cum_future)

def train_series_for_origin(df, cut_date, lookback_days):
    cut = pd.Timestamp(cut_date)
    tr_start = cut - pd.Timedelta(days=int(lookback_days))
    dsub = df[(pd.to_datetime(df["date"]) <= cut) & (pd.to_datetime(df["date"]) >= tr_start)].copy()
    s = obs7_series_from_df(dsub)
    return s


# -----------------------------
# 4) Stronger baselines (fast): Naive, ETS, Theta (if available)
# -----------------------------
def forecast_naive(y_series, cut_date, horizon, method="last"):
    cut_date = pd.Timestamp(cut_date)
    y_train = y_series.loc[:cut_date].dropna()
    if len(y_train) == 0:
        return np.zeros(horizon, float)
    if method == "last":
        v = float(y_train.iloc[-1])
        return np.full(horizon, v, float)
    elif method == "mean7":
        tail = y_train.iloc[-7:] if len(y_train) >= 7 else y_train
        v = float(np.mean(tail))
        return np.full(horizon, v, float)
    else:
        raise ValueError("method must be 'last' or 'mean7'")

def forecast_ets(y_series, cut_date, horizon, history_days=60):
    """
    ETS / Holt-Winters (statsmodels). Works on 7d-avg series.
    """
    cut_date = pd.Timestamp(cut_date)
    y_train = y_series.loc[:cut_date].dropna()
    y_train = y_train.iloc[-history_days:] if len(y_train) > history_days else y_train
    if len(y_train) < 10:
        return forecast_naive(y_series, cut_date, horizon, method="last")

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mod = ExponentialSmoothing(
                y_train.values.astype(float),
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated"
            )
            fit = mod.fit(optimized=True)
            fc = fit.forecast(horizon)
        fc = np.asarray(fc, float).reshape(-1)
        fc = np.clip(fc, 0.0, None)
        if len(fc) < horizon:
            fc = np.pad(fc, (0, horizon-len(fc)), mode="edge")
        return fc[:horizon]
    except Exception:
        return forecast_naive(y_series, cut_date, horizon, method="last")

def forecast_theta(y_series, cut_date, horizon, history_days=90):
    """
    Theta model baseline (statsmodels >= 0.12 typically).
    Falls back to ETS if unavailable.
    """
    cut_date = pd.Timestamp(cut_date)
    y_train = y_series.loc[:cut_date].dropna()
    y_train = y_train.iloc[-history_days:] if len(y_train) > history_days else y_train
    if len(y_train) < 20:
        return forecast_ets(y_series, cut_date, horizon, history_days=min(history_days, 60))

    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tm = ThetaModel(y_train.values.astype(float), period=1)  # period=1 since we smoothed weekly already
            fit = tm.fit()
            fc = fit.forecast(horizon)
        fc = np.asarray(fc, float).reshape(-1)
        fc = np.clip(fc, 0.0, None)
        if len(fc) < horizon:
            fc = np.pad(fc, (0, horizon-len(fc)), mode="edge")
        return fc[:horizon]
    except Exception:
        return forecast_ets(y_series, cut_date, horizon, history_days=min(history_days, 60))

def baseline_mae_suite(df, cut_date, horizon, lookback_days_for_mase=180):
    """
    Compute baseline MAEs against the same "true_wavg" definition you use.
    Returns dict with mae_naive_last, mae_ets, mae_theta, mase_scale (for later use).
    """
    y = obs7_series_from_df(df)
    y_train_for_scale = train_series_for_origin(df, cut_date, lookback_days_for_mase)
    scale = mase_scale(y_train_for_scale.values)

    true_wavg = true_future_7d_from_df(df, cut_date, horizon)
    if true_wavg is None:
        return None

    pred_naive = forecast_naive(y, cut_date, horizon, method="last")
    pred_ets   = forecast_ets(y, cut_date, horizon, history_days=60)
    pred_theta = forecast_theta(y, cut_date, horizon, history_days=90)

    out = {
        "mase_scale": float(scale),
        "mae_naive_last": float(np.mean(np.abs(pred_naive - true_wavg))),
        "mae_ets": float(np.mean(np.abs(pred_ets - true_wavg))),
        "mae_theta": float(np.mean(np.abs(pred_theta - true_wavg))),
    }
    return out


# -----------------------------
# 5) Load your existing per-city regime_validation_metrics.csv and augment
# -----------------------------
_LOOKBACK_BY_WINDOW = {nm: int(lb) for (nm, _, lb) in VALIDATION_CONFIG}

def _city_label_from_dirname(d):
    # expects pinn_sueihcdr_multiwindow_v2_{CITY}
    m = re.match(r"pinn_sueihcdr_multiwindow_v2_(.+)$", Path(d).name)
    return m.group(1) if m else Path(d).name

def get_city_population_fast():
    """
    Returns dict city->population using:
      - CITY_POP for world
      - covid_county_population_usafacts.csv for US cities
    """
    pop = {}
    # world
    for k,v in CITY_POP.items():
        pop[k] = float(v)

    # US via usa facts pop file (fast, single read)
    try:
        popdf = pd.read_csv(_resolve_data_path("covid_county_population_usafacts.csv"))
        for (label, county, state, sub) in US_CITIES:
            row = popdf[(popdf["County Name"]==county) & (popdf["State"]==state)]
            if not row.empty:
                pop[label] = float(row.iloc[0]["population"])
    except Exception as e:
        print("WARNING: Could not load US pop file for per-capita metrics:", e)

    return pop

# Populated lazily by _ensure_city_pop_all() so importing this module does not
# attempt to read the US population CSV (which only exists at the data root).
_CITY_POP_ALL = None


def _ensure_city_pop_all():
    """Return the city->population dict, computing it on first call.

    Once populated, ``_CITY_POP_ALL`` is the same dict returned each time, so
    in-place mutations performed elsewhere in this module continue to work.
    """
    global _CITY_POP_ALL
    if _CITY_POP_ALL is None:
        _CITY_POP_ALL = get_city_population_fast()
    return _CITY_POP_ALL

from pathlib import Path
import pandas as pd

def load_all_existing_city_metrics(out_root=OUT_PATH_STR):
    out_root = Path(out_root)
    rows = []

    # Look for files like: regime_validation_metrics_<City>.csv
    for f in out_root.glob("regime_validation_metrics_*.csv"):
        if f.is_file():
            # Extract city name from filename
            # Example: "regime_validation_metrics_SanDiego.csv"
            city = f.stem.replace("regime_validation_metrics_", "")

            dfm = pd.read_csv(f)
            dfm["city"] = city
            rows.append(dfm)

    if not rows:
        print("No regime_validation_metrics_<city>.csv found under", out_root)
        return None

    allm = pd.concat(rows, ignore_index=True)

    # Standardize types
    allm["cut_date"] = pd.to_datetime(allm["cut_date"])
    allm["horizon"] = pd.to_numeric(allm["horizon"], errors="coerce").astype(int)

    return allm

def load_city_df(label):
    # 1. Clean the label
    label_clean = str(label).strip().lower()
    
    # 2. Hardcode the US routing to bypass corrupted global variables
    hardcoded_us_map = {
        "sandiego":   ("San Diego County", "CA", "California"),
        "seattle":    ("King County", "WA", "Washington"),
        "newyork":    ("New York County", "NY", "New York"),
        "chicago":    ("Cook County", "IL", "Illinois"),
        "houston":    ("Harris County", "TX", "Texas"),
        "phoenix":    ("Maricopa County", "AZ", "Arizona"),
        "miami":      ("Miami-Dade County", "FL", "Florida"),
        "denver":     ("Denver County", "CO", "Colorado"),
        "losangeles": ("Los Angeles County", "CA", "California"),
        "sanfrancisco": ("San Francisco County", "CA", "California"),
    }

    # -- Check US Cities --
    if label_clean in hardcoded_us_map:
        county, state, sub = hardcoded_us_map[label_clean]
        df, Npop = load_us_county_series(county, state, sub)
        return df, float(Npop)

    # -- Check World Cities --
    intl_map = {k.lower(): k for k in CITY_SPECS.keys()}
    if label_clean in intl_map:
        original_key = intl_map[label_clean]
        df, _ = load_world_city_series(original_key)
        
        # Pull population securely
        Npop = CITY_POP.get(original_key, np.nan)
        if np.isnan(Npop):
            _, Npop_alt = load_world_city_series(original_key)
            Npop = Npop_alt
            
        return df, float(Npop)

    # -- Fail Safely --
    raise ValueError(f"Unknown city label: '{label}'. Check spelling in the metrics CSV.")

def augment_metrics_with_scale_and_baselines(metrics_df, cache_city_df=True, do_baselines=True):
    """
    Adds:
      - Npop
      - MAE per 100k for PINN and ARIMA
      - MASE for PINN and ARIMA (using training-window scale)
      - ETS/Theta/Naive MAEs (computed from raw data; no PINN retrain)
    """
    if metrics_df is None or len(metrics_df) == 0:
        return metrics_df

    # cache city dfs to avoid re-loading
    city_cache = {}

    aug = metrics_df.copy()
    aug["lookback_days"] = aug["window"].map(_LOOKBACK_BY_WINDOW).astype(float)

    # fill pops (lazy-load city population dict on first access)
    city_pop = _ensure_city_pop_all()
    aug["Npop"] = aug["city"].map(city_pop).astype(float)

    # some cities might be missing pop -> load once
    miss = aug[aug["Npop"].isna()]["city"].unique().tolist()
    for c in miss:
        try:
            _, Np = load_city_df(c)
            city_pop[c] = float(Np)
        except Exception:
            city_pop[c] = np.nan
    aug["Npop"] = aug["city"].map(city_pop).astype(float)

    # compute MASE scale per row (needs training series)
    mase_scales = []
    mae_naive = []
    mae_ets = []
    mae_theta = []

    for i, r in aug.iterrows():
        city = r["city"]
        cut = r["cut_date"]
        H = int(r["horizon"])
        lb = int(r["lookback_days"]) if np.isfinite(r["lookback_days"]) else 180

        if cache_city_df and city in city_cache:
            df_city, Np = city_cache[city]
        else:
            df_city, Np = load_city_df(city)
            if cache_city_df:
                city_cache[city] = (df_city, Np)

        # MASE scale
        try:
            y_train = train_series_for_origin(df_city, cut, lb)
            sc = mase_scale(y_train.values)
        except Exception:
            sc = np.nan
        mase_scales.append(sc)

        # baselines
        if do_baselines:
            try:
                b = baseline_mae_suite(df_city, cut, H, lookback_days_for_mase=lb)
            except Exception:
                b = None
            if b is None:
                mae_naive.append(np.nan); mae_ets.append(np.nan); mae_theta.append(np.nan)
            else:
                mae_naive.append(b["mae_naive_last"])
                mae_ets.append(b["mae_ets"])
                mae_theta.append(b["mae_theta"])
        else:
            mae_naive.append(np.nan); mae_ets.append(np.nan); mae_theta.append(np.nan)

    aug["mase_scale"] = mase_scales

    # per 100k MAE
    aug["mae_pin_per100k"] = aug.apply(lambda r: per100k(r["mae_pin"], r["Npop"]) if np.isfinite(r["Npop"]) else np.nan, axis=1)
    aug["mae_ari_per100k"] = aug.apply(lambda r: per100k(r["mae_ari"], r["Npop"]) if np.isfinite(r["Npop"]) else np.nan, axis=1)

    # MASE
    aug["mase_pin"] = aug["mae_pin"] / aug["mase_scale"]
    aug["mase_ari"] = aug["mae_ari"] / aug["mase_scale"]

    # add baseline MAEs
    aug["mae_naive_last"] = mae_naive
    aug["mae_ets"] = mae_ets
    aug["mae_theta"] = mae_theta

    # baseline MASE too
    aug["mase_naive_last"] = aug["mae_naive_last"] / aug["mase_scale"]
    aug["mase_ets"] = aug["mae_ets"] / aug["mase_scale"]
    aug["mase_theta"] = aug["mae_theta"] / aug["mase_scale"]

    return aug

def summarize_pairwise(df, a_col="mae_pin", b_col="mae_ari", label_a="PINN", label_b="ARIMA"):
    d = df[[a_col, b_col, "city", "window", "horizon"]].dropna()
    if len(d) == 0:
        return None

    wins = int(np.sum(d[a_col] < d[b_col]))
    n = int(len(d))
    win_pct = 100.0 * wins / n
    ci_lo, ci_hi = wilson_ci(wins, n)
    ci_lo *= 100.0; ci_hi *= 100.0

    # naive paired Wilcoxon across evaluations
    try:
        p = float(wilcoxon(d[a_col] - d[b_col]).pvalue)
    except Exception:
        p = np.nan

    # city-clustered: average within city, then Wilcoxon across cities
    city_agg = d.groupby("city").apply(lambda g: float(np.mean(g[a_col] - g[b_col]))).reset_index(name="mean_diff")
    try:
        p_city = float(wilcoxon(city_agg["mean_diff"].values).pvalue)
    except Exception:
        p_city = np.nan

    out = {
        "n_eval": n,
        "wins": wins,
        "win_pct": win_pct,
        "win_ci95": f"[{ci_lo:.1f}%, {ci_hi:.1f}%]",
        "wilcoxon_eval_p": p,
        "wilcoxon_city_p": p_city,
        "label_a": label_a,
        "label_b": label_b,
    }
    return out

def regime_pvals_with_holm(df, a_col="mae_pin", b_col="mae_ari"):
    """
    Per-horizon regime tests + Holm correction.
    Uses evaluation-level Wilcoxon within each (regime,horizon).
    """
    out_rows = []
    for H in sorted(df["horizon"].dropna().unique()):
        subH = df[df["horizon"] == H]
        regs = sorted(subH["window"].dropna().unique())
        pvals = []
        tmp = []
        for reg in regs:
            d = subH[subH["window"] == reg][[a_col, b_col]].dropna()
            if len(d) < 6:
                p = np.nan
            else:
                try:
                    p = float(wilcoxon(d[a_col] - d[b_col]).pvalue)
                except Exception:
                    p = np.nan
            pvals.append(p)
            tmp.append((H, reg, len(d), p))
        adj = holm_adjust([p if np.isfinite(p) else 1.0 for p in pvals]) if len(pvals) else []
        for (H, reg, n, p), pa in zip(tmp, adj):
            out_rows.append({"horizon": H, "regime": reg, "n": n, "p_raw": p, "p_holm": pa})
    return pd.DataFrame(out_rows)

# -----------------------------
# 6) OPTIONAL: Ablation scaffolding (targeted runs)
# -----------------------------
@contextmanager
def temp_globals(**kwargs):
    """
    Temporarily override global constants (e.g., ETA_BOUNDS, OMEGA_BOUNDS, W_PHYS).
    Useful for ablations without rewriting your training function.
    """
    g = globals()
    old = {}
    for k,v in kwargs.items():
        old[k] = g.get(k, None)
        g[k] = v
    try:
        yield
    finally:
        for k,v in old.items():
            g[k] = v

def run_targeted_ablation(
    city_label,
    regimes=("Winter20", "BA5_Waning"),
    horizons=(7,14),
    n_models=3,
    ablation_name="no_waning",
    outdir=None
):
    """
    Runs a SMALL ablation suite for one city on selected regimes/horizons.
    This WILL retrain the PINN on those windows (but only a few runs).

    ablation_name:
      - "baseline"     : your current model
      - "no_waning"    : omega=0 and eta=0
      - "no_bio"       : eta=0 only
      - "no_behav"     : omega=0 only
      - "no_physics"   : W_PHYS=0 (keeps other regularizers)
    """
    df, Npop = load_city_df(city_label)
    df["date"] = pd.to_datetime(df["date"])

    reg_map = {nm: (pd.Timestamp(cut), int(lb)) for nm, cut, lb in VALIDATION_CONFIG}
    picks = [(r, reg_map[r][0], reg_map[r][1]) for r in regimes if r in reg_map]

    if outdir is None:
        outdir = Path(OUT_PATH_STR) / "ablations_natcomm" / city_label / ablation_name
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # choose overrides
    overrides = {}
    if ablation_name == "no_waning":
        overrides.update({"OMEGA_BOUNDS": (0.0, 0.0), "ETA_BOUNDS": (0.0, 0.0)})
    elif ablation_name == "no_bio":
        overrides.update({"ETA_BOUNDS": (0.0, 0.0)})
    elif ablation_name == "no_behav":
        overrides.update({"OMEGA_BOUNDS": (0.0, 0.0)})
    elif ablation_name == "no_physics":
        overrides.update({"W_PHYS": 0.0})
    elif ablation_name == "baseline":
        overrides = {}
    else:
        raise ValueError("Unknown ablation_name")

    results = []

    with temp_globals(**overrides):
        print(f"\n=== ABLATION {ablation_name} | city={city_label} | overrides={overrides} ===")
        for (reg, cut, lb) in picks:
            if cut > df["date"].max() - pd.Timedelta(days=max(horizons)):
                print("  SKIP", reg, "insufficient future data")
                continue
            tr_start = max(cut - pd.Timedelta(days=lb), df["date"].min())
            dsub = df[(df["date"]<=cut) & (df["date"]>=tr_start)].reset_index(drop=True)
            if len(dsub) < 60:
                print("  SKIP", reg, "<60 days")
                continue

            max_h = max(horizons)
            sd_future = project_sd_future(dsub["sd"].values, max_h)

            cfg = TrainCfg(
                max_epochs=EPOCHS_MAX, lr=LR_FULL, sd_lag=SD_LAG_DAYS,
                rollout_extra=max_h, validation_days=VALIDATION_DAYS, patience_epochs=PATIENCE_EPOCHS
            )

            # train ensemble (small n_models for speed)
            preds = []
            last_res = None
            for i in range(n_models):
                torch.manual_seed(10_000 + i*111)
                rr = train_sueihcdr_once(dsub, Npop, cfg, return_all=True, sd_future=sd_future)
                preds.append(rr["Ccum"])
                last_res = rr
            stack = np.vstack(preds)
            # use median like your pipeline
            fr = dict(last_res)
            fr["Ccum"] = np.median(stack, axis=0)

            for H in horizons:
                m = generate_plots_and_metrics(fr, df, Npop, cut, H, window_name=reg, lookback_days=lb, outdir=outdir)
                if m:
                    m.update({"city": city_label, "ablation": ablation_name, "window": reg, "cut_date": str(cut.date()), "horizon": int(H)})
                    results.append(m)

    out_df = pd.DataFrame(results)
    out_df.to_csv(outdir / "ablation_metrics.csv", index=False)
    print("Saved:", outdir / "ablation_metrics.csv")
    return out_df


# -----------------------------
# 7) RUN: compile + augment + summarize (no retraining)
# -----------------------------
def run_full_summary():
    """Compile existing per-city metrics, augment, save, and print summaries.

    Performs file I/O (reads Resultados_Cidades_02152026/, writes CSVs). Called
    from ``python -m covid_pinn.stats_summary``; importing this module does not
    execute it.
    """
    metrics = load_all_existing_city_metrics(
        Path(OUT_PATH_STR) / "Resultados_Cidades_02152026"
    )

    if metrics is None:
        print("No existing metrics found. If you haven't run mode=full per city, run that first.")
        return None

    print("Loaded existing evaluation rows:", len(metrics))
    aug = augment_metrics_with_scale_and_baselines(metrics, cache_city_df=True, do_baselines=True)

    # Save augmented results
    out_aug = Path(OUT_PATH_STR) / "natcomm_revision_metrics_augmented.csv"
    aug.to_csv(out_aug, index=False)
    print("Saved augmented metrics:", out_aug)

    # Overall summaries (raw MAE and MASE)
    s1 = summarize_pairwise(aug, "mae_pin", "mae_ari", "PINN", "ARIMA")
    s2 = summarize_pairwise(aug, "mase_pin", "mase_ari", "PINN", "ARIMA")
    s3 = summarize_pairwise(aug, "mae_pin", "mae_ets", "PINN", "ETS")
    s4 = summarize_pairwise(aug, "mae_ari", "mae_ets", "ARIMA", "ETS")
    print("\n=== OVERALL (raw MAE) PINN vs ARIMA ===\n", s1)
    print("\n=== OVERALL (MASE)   PINN vs ARIMA ===\n", s2)
    print("\n=== OVERALL (raw MAE) PINN vs ETS ===\n", s3)
    print("\n=== OVERALL (raw MAE) ARIMA vs ETS ===\n", s4)

    # Regime p-values with Holm correction (helps address multiple testing critique)
    reg_tbl = regime_pvals_with_holm(aug, a_col="mae_pin", b_col="mae_ari")
    out_reg = Path(OUT_PATH_STR) / "natcomm_regime_pvals_holm.csv"
    reg_tbl.to_csv(out_reg, index=False)
    print("\nSaved regime p-values (Holm corrected):", out_reg)
    print(reg_tbl)

    return {"metrics": metrics, "augmented": aug, "regime_pvals": reg_tbl}


if __name__ == "__main__":
    _print_audit_banner()
    run_full_summary()
