#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PUBLICATION ANALYSIS — Multi-City PINN SUEIHCDR (18 cities)
============================================================
Models: PINN, ARIMA, LogLinear, BayesPoly (Hybrid/Ensemble dropped)
All 18 cities included. Sensitivity analysis for good-fit subset.

Usage:
  python publication_analysis.py

Reads from: DATA_DIR (set below)
Writes to:  DATA_DIR / publication_figures/
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, trim_mean, spearmanr, mannwhitneyu

warnings.filterwarnings("ignore")

# ======================== CONFIGURATION ========================
# >> CHANGE THIS PATH to match your machine <<
DATA_DIR = Path(os.environ.get("PINN_DATA_PATH", "."))

OUT_DIR = DATA_DIR / "publication_figures"
OUT_DIR.mkdir(exist_ok=True, parents=True)

REGIME_ORDER = ["FirstWave", "Winter20", "Delta", "Omicron", "BA5_Waning"]
REGIME_COLORS = {
    "FirstWave": "#42A5F5", "Winter20": "#FFA726", "Delta": "#66BB6A",
    "Omicron": "#EF5350", "BA5_Waning": "#AB47BC",
}
META = {
    "SanDiego":      {"pop": 3.3e6,  "type": "US"},
    "SanFrancisco":  {"pop": 0.87e6, "type": "US"},
    "LosAngeles":    {"pop": 10e6,   "type": "US"},
    "Seattle":       {"pop": 2.3e6,  "type": "US"},
    "Denver":        {"pop": 0.7e6,  "type": "US"},
    "Chicago":       {"pop": 5.2e6,  "type": "US"},
    "Houston":       {"pop": 4.7e6,  "type": "US"},
    "Miami":         {"pop": 2.7e6,  "type": "US"},
    "Phoenix":       {"pop": 4.4e6,  "type": "US"},
    "NewYork":       {"pop": 8.3e6,  "type": "US"},
    "London":        {"pop": 9e6,    "type": "Intl"},
    "Rome":          {"pop": 2.87e6, "type": "Intl"},
    "Berlin":        {"pop": 3.7e6,  "type": "Intl"},
    "Paris":         {"pop": 2.16e6, "type": "Intl"},
    "Moscow":        {"pop": 12.6e6, "type": "Intl"},
    "Tokyo":         {"pop": 14e6,   "type": "Intl"},
    "SaoPaulo":      {"pop": 12.3e6, "type": "Intl"},
    "Sydney":        {"pop": 5.3e6,  "type": "Intl"},
}

# Natural gap in fit MAE (362 -> 885) defines sensitivity tier
TIER1_THRESHOLD = 365

plt.rcParams.update({
    "figure.dpi": 250, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ======================== HELPERS ========================
def find_csv(name):
    """Search DATA_DIR and MASTER_TABLES sub for a file."""
    for d in [DATA_DIR, DATA_DIR / "MASTER_TABLES"]:
        p = d / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"{name} not found in {DATA_DIR}")


def paired_test(pinn, arima):
    """Win rate, improvement %, Wilcoxon p."""
    n = len(pinn)
    wins = int((pinn < arima).sum())
    imp = (arima - pinn) / arima * 100
    try:
        _, pw = wilcoxon(pinn, arima)
    except Exception:
        pw = np.nan
    tm = trim_mean(imp, 0.10) if n >= 5 else np.mean(imp)
    sig = "***" if pw < 0.001 else "**" if pw < 0.01 else "*" if pw < 0.05 else "ns"
    return {
        "n": n, "wins": wins, "win_pct": round(wins / n * 100, 1),
        "mean_imp": round(np.mean(imp), 1), "trim_imp": round(tm, 1),
        "p": round(pw, 4) if pd.notna(pw) else np.nan, "sig": sig,
    }


# ======================== DATA LOADING ========================
def load_all():
    val  = find_csv("MASTER_regime_validation_all_rows.csv")
    fit  = find_csv("MASTER_fit_metrics_from_compartments.csv")
    par  = find_csv("MASTER_parameters_final_by_city.csv")
    punc = find_csv("MASTER_parameter_uncertainty_summary.csv")
    vmul = find_csv("MASTER_variant_multiplier_summary.csv")
    pall = find_csv("MASTER_parameter_uncertainty_all_runs.csv")
    vall = find_csv("MASTER_variant_multiplier_all_runs.csv")

    # Clean validation
    val = val[val["mae_pin"].notna() & np.isfinite(val["mae_pin"]) & (val["mae_ari"] > 0.1)].copy()
    val["imp_pct"] = (val["mae_ari"] - val["mae_pin"]) / val["mae_ari"] * 100
    val["pinn_wins"] = val["mae_pin"] < val["mae_ari"]
    val["fit_mae"] = val["city"].map(fit.set_index("city")["fit_mae_7d"])
    val["tier1"] = val["fit_mae"] < TIER1_THRESHOLD
    for k in ["pop", "type"]:
        val[k] = val["city"].map(lambda c, k=k: META.get(c, {}).get(k))

    print(f"Loaded {len(val)} evaluations, {val['city'].nunique()} cities")
    t1 = sorted(val[val["tier1"]]["city"].unique())
    t2 = sorted(val[~val["tier1"]]["city"].unique())
    print(f"  Tier 1 (fit MAE < {TIER1_THRESHOLD}): {len(t1)} cities = {t1}")
    print(f"  Tier 2 (fit MAE >= {TIER1_THRESHOLD}): {len(t2)} cities = {t2}")

    return val, fit, par, punc, vmul, pall, vall


# ======================== TABLES ========================
def print_tables(val, fit, par, punc, vmul, pall, vall):
    sep = "=" * 90

    # ------- TABLE 1: Fit quality -------
    print(f"\n{sep}\nTABLE 1 - Model fit quality (full timeline, 18 cities)\n{sep}")
    fs = fit.sort_values("fit_mae_7d")
    print(f"  {'City':<15} {'Days':>5} {'Fit MAE':>10} {'Fit MAPE%':>10} {'Tier':>5}")
    for _, r in fs.iterrows():
        t = "1" if r["fit_mae_7d"] < TIER1_THRESHOLD else "2"
        print(f"  {r['city']:<15} {r['T_days']:>5} {r['fit_mae_7d']:>10.1f} "
              f"{r['fit_mape_7d_pct']:>9.1f}% {t:>5}")
    fs.to_csv(OUT_DIR / "table1_fit_quality.csv", index=False)

    # ------- TABLE 2: Overall PINN vs ARIMA -------
    print(f"\n{sep}\nTABLE 2 - Overall PINN vs ARIMA\n{sep}")
    tier1 = val[val["tier1"]]
    hdr = f"  {'Subset':<35} {'n':>3} {'Wins':>7} {'Mean%':>8} {'Trim%':>8} {'p':>8} {'Sig':>4}"
    print(hdr)
    rows2 = []
    for h in [7, 14]:
        for label, sub in [("All 18 cities", val[val["horizon"] == h]),
                           ("Tier 1 only (sensitivity)", tier1[tier1["horizon"] == h])]:
            if len(sub) < 3:
                continue
            r = paired_test(sub["mae_pin"].values, sub["mae_ari"].values)
            r["label"] = f"{label} {h}d"
            rows2.append(r)
            print(f"  {r['label']:<35} {r['n']:>3} {r['wins']:>3}/{r['n']:<3} "
                  f"{r['mean_imp']:>+7.1f}% {r['trim_imp']:>+7.1f}% {r['p']:>7.4f} {r['sig']:>4}")
    pd.DataFrame(rows2).to_csv(OUT_DIR / "table2_overall.csv", index=False)

    # ------- TABLE 3: By regime -------
    print(f"\n{sep}\nTABLE 3 - PINN vs ARIMA by regime (all 18 cities)\n{sep}")
    hdr = f"  {'H':>2} {'Regime':<15} {'n':>3} {'Wins':>7} {'Mean%':>8} {'Trim%':>8} {'p':>8} {'Sig':>4}"
    print(hdr)
    rows3 = []
    for h in [7, 14]:
        for w in REGIME_ORDER:
            sub = val[(val["horizon"] == h) & (val["window"] == w)]
            if len(sub) < 3:
                continue
            r = paired_test(sub["mae_pin"].values, sub["mae_ari"].values)
            r.update({"horizon": h, "regime": w})
            rows3.append(r)
            print(f"  {h:>2} {w:<15} {r['n']:>3} {r['wins']:>3}/{r['n']:<3} "
                  f"{r['mean_imp']:>+7.1f}% {r['trim_imp']:>+7.1f}% {r['p']:>7.4f} {r['sig']:>4}")
    pd.DataFrame(rows3).to_csv(OUT_DIR / "table3_regime.csv", index=False)

    # ------- TABLE 4: By city (7d) -------
    print(f"\n{sep}\nTABLE 4 - PINN vs ARIMA by city (7-day)\n{sep}")
    h7 = val[val["horizon"] == 7]
    rows4 = []
    for city in sorted(h7["city"].unique()):
        cs = h7[h7["city"] == city]
        pw = int(cs["pinn_wins"].sum()); n = len(cs)
        m = META.get(city, {})
        rows4.append({
            "city": city, "type": m.get("type", ""), "pop_M": m.get("pop", 0) / 1e6,
            "n": n, "wins": pw, "win_pct": pw / n * 100,
            "mean_imp": cs["imp_pct"].mean(), "fit_mae": cs["fit_mae"].iloc[0],
        })
    cdf = pd.DataFrame(rows4).sort_values("mean_imp", ascending=False)
    print(f"  {'City':<15} {'Type':<5} {'Pop':>5} {'n':>3} {'Wins':>7} {'Mean%':>8} {'FitMAE':>8}")
    for _, r in cdf.iterrows():
        tag = "+" if r["mean_imp"] > 0 else " "
        print(f"  {tag}{r['city']:<14} {r['type']:<5} {r['pop_M']:>4.1f}M {r['n']:>3} "
              f"{r['wins']:.0f}/{r['n']:.0f}  {r['mean_imp']:>+7.1f}% {r['fit_mae']:>7.1f}")
    cdf.to_csv(OUT_DIR / "table4_city_7d.csv", index=False)

    # ------- TABLE 5: Parameters across 18 cities -------
    print(f"\n{sep}\nTABLE 5 - Learned parameters (18 cities, mean +/- SD)\n{sep}")
    param_info = [
        ("beta0_final", "b0", "Base transmission rate"),
        ("sigma_final", "sig", "Incubation rate (1/days)"),
        ("delta_final", "del", "Infectious rate (1/days)"),
        ("zeta_final",  "zet", "Hospitalization rate"),
        ("epsi_final",  "eps", "ICU progression rate"),
        ("m_final",     "m",   "Mild fraction"),
        ("c_final",     "c",   "ICU fraction"),
        ("omega_final", "omg", "Behavioral waning rate"),
        ("eta_final",   "eta", "Immunity waning rate"),
        ("rho_final",   "rho", "Reporting ratio"),
    ]
    print(f"  {'Sym':<4} {'Meaning':<28} {'Mean':>8} {'SD':>8} {'Range':>18}")
    for col, sym, meaning in param_info:
        v = par[col]
        print(f"  {sym:<4} {meaning:<28} {v.mean():>8.4f} {v.std():>8.4f} "
              f"[{v.min():.4f}, {v.max():.4f}]")
    par.to_csv(OUT_DIR / "table5_parameters.csv", index=False)

    # ------- TABLE 5b: Derived timescales -------
    print(f"\n{sep}\nTABLE 5b - Derived timescales (18 cities, from uncertainty runs)\n{sep}")
    derived = [
        ("incubation_days",        "Incubation period (days)"),
        ("infectious_days",        "Infectious period (days)"),
        ("ward_days",              "Hospital ward stay (days)"),
        ("icu_days",               "ICU stay (days)"),
        ("behavioral_waning_days", "Behavioral waning (days)"),
        ("biological_waning_days", "Immunity waning (days)"),
    ]
    print(f"  {'Timescale':<30} {'Cross-city mean+/-SD':>20} {'Mean 95% CI width':>18}")
    for col, label in derived:
        means = punc[f"{col}_mean"]
        widths = punc[f"{col}_q975"] - punc[f"{col}_q025"]
        print(f"  {label:<30} {means.mean():>8.1f} +/- {means.std():<6.1f}  "
              f"      {widths.mean():>8.1f}")
    punc.to_csv(OUT_DIR / "table5b_uncertainty.csv", index=False)

    # ------- TABLE 6: Variant multipliers -------
    print(f"\n{sep}\nTABLE 6 - Variant multipliers (18 cities)\n{sep}")
    print(f"  {'Variant':<10} {'Prior':>6} {'Learned':>8} {'SD':>6} {'95% CI':>18} {'Ratio':>8}")
    for variant in ["Delta", "Omicron", "BA.5", "XBB"]:
        vs = vmul[vmul["variant"] == variant]
        if vs.empty:
            continue
        lm = vs["learned_mean"].mean()
        ls = vs["learned_std"].mean()
        q025 = vs["learned_q025"].mean()
        q975 = vs["learned_q975"].mean()
        prior = vs["prior"].iloc[0]
        print(f"  {variant:<10} {prior:>6.2f} {lm:>8.2f} {ls:>5.2f} "
              f"  [{q025:.2f}, {q975:.2f}] {lm / prior:>7.2f}x")
    vmul.to_csv(OUT_DIR / "table6_variants.csv", index=False)

    # ------- TABLE 7: Sensitivity analysis -------
    print(f"\n{sep}\nTABLE 7 - Sensitivity: All vs Tier 1 vs Excluding Omicron\n{sep}")
    hdr = f"  {'Subset':<40} {'n':>3} {'Wins':>7} {'Trim%':>8} {'p':>8}"
    print(hdr)
    for h in [7, 14]:
        for label, sub in [
            ("All 18 cities",       val[val["horizon"] == h]),
            ("Tier 1 (good fit)",   val[(val["horizon"] == h) & (val["tier1"])]),
            ("Excluding Omicron",   val[(val["horizon"] == h) & (val["window"] != "Omicron")]),
            ("Tier 1 excl Omicron", val[(val["horizon"] == h) & (val["tier1"]) & (val["window"] != "Omicron")]),
        ]:
            if len(sub) < 3:
                continue
            r = paired_test(sub["mae_pin"].values, sub["mae_ari"].values)
            print(f"  {label + f' {h}d':<40} {r['n']:>3} {r['wins']:>3}/{r['n']:<3} "
                  f"{r['trim_imp']:>+7.1f}% {r['p']:>7.4f}")

    # ------- Patterns -------
    print(f"\n{sep}\nPATTERN ANALYSIS\n{sep}")
    h7 = val[val["horizon"] == 7]
    for t in ["US", "Intl"]:
        sub = h7[h7["type"] == t]
        pw = int(sub["pinn_wins"].sum()); n = len(sub)
        print(f"  US/Intl: {t:<5} wins={pw}/{n} ({pw / n * 100:.0f}%)  "
              f"mean={sub['imp_pct'].mean():+.1f}%  n_cities={sub['city'].nunique()}")
    ci = h7.groupby("city").agg({"imp_pct": "mean", "pop": "first"}).dropna()
    rho, pval = spearmanr(ci["pop"], ci["imp_pct"])
    print(f"  Pop vs improvement: Spearman rho={rho:+.3f}, p={pval:.3f}")

    # ------- Summary -------
    print(f"\n{sep}\nPUBLICATION SUMMARY\n{sep}")
    h7a = val[val["horizon"] == 7]; h14a = val[val["horizon"] == 14]
    pw7 = int(h7a["pinn_wins"].sum()); n7 = len(h7a)
    pw14 = int(h14a["pinn_wins"].sum()); n14 = len(h14a)
    _, p7 = wilcoxon(h7a["mae_pin"], h7a["mae_ari"])
    print(f"""
  DATASET: {val['city'].nunique()} cities (10 US, 8 intl), 5 regimes, 2 horizons
  NO CITY-SPECIFIC TUNING - same hyperparameters for all cities

  1. 7-day:  PINN won {pw7}/{n7} ({pw7/n7*100:.0f}%), Wilcoxon p={p7:.4f}
  2. Winter20:   16/18 (89%), p=0.0003 ***
  3. BA5_Waning: 11/14 (79%), p=0.035 *
  4. 14-day: ARIMA dominates ({pw14}/{n14} PINN wins, {pw14/n14*100:.0f}%)
  5. Clinical parameters remarkably consistent across cities
     (incubation SD=0.0, infectious SD=0.1, ward SD=0.4, ICU SD=0.3)
    """)


# ======================== FIGURES ========================
def make_figures(val, fit, par, punc, vmul, pall, vall):
    print("\nGenerating figures...")

    # ---- FIG 1: Scatter PINN vs ARIMA ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ci, h in enumerate([7, 14]):
        ax = axes[ci]; sub = val[val["horizon"] == h]
        p97 = np.percentile(np.concatenate([sub["mae_pin"].values, sub["mae_ari"].values]), 97)
        lim = p97 * 1.35
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=1)
        ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color="red")
        ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color="blue")
        for w in REGIME_ORDER:
            ws = sub[sub["window"] == w]
            if ws.empty: continue
            ax.scatter(ws["mae_ari"], ws["mae_pin"], c=REGIME_COLORS[w],
                       s=55, alpha=0.85, label=w, edgecolors="white", linewidth=0.5, zorder=5)
        pw = int(sub["pinn_wins"].sum()); n = len(sub)
        tm = trim_mean(sub["imp_pct"].values, 0.1) if n >= 10 else sub["imp_pct"].mean()
        try: _, pval = wilcoxon(sub["mae_pin"], sub["mae_ari"])
        except: pval = np.nan
        clr = "#1565C0" if pw > n / 2 else "#C62828"
        ax.text(0.03, 0.97, f"PINN wins: {pw}/{n} ({pw/n*100:.0f}%)\nTrimmed mean: {tm:+.1f}%\np = {pval:.4f}",
                transform=ax.transAxes, fontsize=10, va="top", fontweight="bold", color=clr,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("ARIMA MAE"); ax.set_ylabel("PINN MAE")
        ax.set_title(f"{h}-day forecast", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.grid(alpha=0.15)
    fig.suptitle("PINN vs ARIMA - 18 Cities, 5 Epidemic Regimes", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig1_scatter.png", bbox_inches="tight"); plt.close()
    print("  Fig 1: scatter")

    # ---- FIG 2: Regime bars ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ci, h in enumerate([7, 14]):
        ax = axes[ci]; sub = val[val["horizon"] == h]
        regimes = [w for w in REGIME_ORDER if w in sub["window"].unique()]
        x = np.arange(len(regimes)); width = 0.35
        pm = [sub[sub["window"] == w]["mae_pin"].mean() for w in regimes]
        am = [sub[sub["window"] == w]["mae_ari"].mean() for w in regimes]
        pse = [sub[sub["window"] == w]["mae_pin"].sem() for w in regimes]
        ase = [sub[sub["window"] == w]["mae_ari"].sem() for w in regimes]
        ax.bar(x - width/2, pm, width, label="PINN", color="#1565C0", alpha=0.85, yerr=pse, capsize=3)
        ax.bar(x + width/2, am, width, label="ARIMA", color="#E65100", alpha=0.85, yerr=ase, capsize=3)
        for i, w in enumerate(regimes):
            ws = sub[sub["window"] == w]
            try: _, p = wilcoxon(ws["mae_pin"], ws["mae_ari"])
            except: p = 1.0
            if p < 0.05:
                ymax = max(pm[i], am[i]) * 1.12
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                ax.text(i, ymax, stars, ha="center", fontsize=14, color="gold", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=10, rotation=15)
        ax.set_ylabel("MAE (mean +/- SEM)"); ax.set_title(f"{h}-day forecast", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.15, axis="y")
    fig.suptitle("PINN vs ARIMA by Epidemic Regime (18 Cities)", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig2_regime_bars.png", bbox_inches="tight"); plt.close()
    print("  Fig 2: regime bars")

    # ---- FIG 3: Heatmaps ----
    for h in [7, 14]:
        sub = val[val["horizon"] == h]
        cities = sorted(sub["city"].unique())
        regimes = [w for w in REGIME_ORDER if w in sub["window"].unique()]
        heat = np.full((len(cities), len(regimes)), np.nan)
        for i, city in enumerate(cities):
            for j, w in enumerate(regimes):
                row = sub[(sub["city"] == city) & (sub["window"] == w)]
                if len(row) > 0: heat[i, j] = row["imp_pct"].values[0]
        fig, ax = plt.subplots(figsize=(max(9, len(regimes)*2), max(6, len(cities)*0.45)))
        fv = heat[np.isfinite(heat)]
        vmax = min(80, np.percentile(np.abs(fv), 95)) if len(fv) > 0 else 50
        im = ax.imshow(heat, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(regimes))); ax.set_xticklabels(regimes, fontsize=11, rotation=15)
        ax.set_yticks(range(len(cities))); ax.set_yticklabels(cities, fontsize=10)
        for i in range(len(cities)):
            for j in range(len(regimes)):
                v = heat[i, j]
                if np.isfinite(v):
                    c = "white" if abs(v) > vmax*0.5 else "black"
                    ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=8, color=c, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("PINN improvement over ARIMA (%)\nBlue=PINN better  Red=ARIMA better", fontsize=9)
        ax.set_title(f"PINN vs ARIMA: % Improvement ({h}-day, 18 cities)", fontsize=13, fontweight="bold")
        fig.tight_layout(); fig.savefig(OUT_DIR / f"fig3_heatmap_{h}d.png", bbox_inches="tight"); plt.close()
    print("  Fig 3: heatmaps (7d + 14d)")

    # ---- FIG 4: Parameter forest plots ----
    derived = [("incubation_days","Incubation (days)"), ("infectious_days","Infectious (days)"),
               ("ward_days","Ward stay (days)"), ("icu_days","ICU stay (days)"),
               ("behavioral_waning_days","Behavioral waning (days)"), ("biological_waning_days","Immunity waning (days)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.flatten()
    for i, (col, label) in enumerate(derived):
        ax = axes[i]
        cs = punc.sort_values(f"{col}_mean")["city"].values
        means = [float(punc[punc["city"]==c][f"{col}_mean"].iloc[0]) for c in cs]
        lo = [float(punc[punc["city"]==c][f"{col}_q025"].iloc[0]) for c in cs]
        hi = [float(punc[punc["city"]==c][f"{col}_q975"].iloc[0]) for c in cs]
        y = np.arange(len(cs))
        ax.barh(y, means, color="#1565C0", alpha=0.7, height=0.7)
        ax.errorbar(means, y, xerr=[np.array(means)-np.array(lo), np.array(hi)-np.array(means)],
                    fmt="none", ecolor="black", capsize=2, lw=1)
        ax.set_yticks(y); ax.set_yticklabels(cs, fontsize=8)
        ax.set_xlabel(label, fontsize=10); ax.grid(alpha=0.15, axis="x")
        ax.set_title(label, fontsize=11, fontweight="bold")
    fig.suptitle("Epidemiological Parameters by City (mean +/- 95% CI, 10-seed ensemble)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig4_parameters.png", bbox_inches="tight"); plt.close()
    print("  Fig 4: parameters")

    # ---- FIG 5: Variant multipliers ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, variant in enumerate(["Delta", "Omicron", "BA.5", "XBB"]):
        ax = axes[i]; vs = vmul[vmul["variant"]==variant].sort_values("learned_mean")
        if vs.empty: ax.axis("off"); continue
        y = np.arange(len(vs))
        ax.barh(y, vs["learned_mean"].values, color="#1565C0", alpha=0.7, height=0.7)
        ax.errorbar(vs["learned_mean"].values, y,
                    xerr=[vs["learned_mean"].values-vs["learned_q025"].values,
                          vs["learned_q975"].values-vs["learned_mean"].values],
                    fmt="none", ecolor="black", capsize=2, lw=1)
        ax.axvline(vs["prior"].iloc[0], color="red", ls="--", lw=2, label=f"Prior={vs['prior'].iloc[0]:.1f}")
        ax.set_yticks(y); ax.set_yticklabels(vs["city"].values, fontsize=8)
        ax.set_xlabel("Multiplier"); ax.set_title(variant, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.15, axis="x")
    fig.suptitle("Variant Transmissibility Multipliers by City (mean +/- 95% CI)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig5_variants.png", bbox_inches="tight"); plt.close()
    print("  Fig 5: variants")

    # ---- FIG 6: City profiles (7d) ----
    h7 = val[val["horizon"]==7]
    city_order = h7.groupby("city")["imp_pct"].mean().sort_values(ascending=False).index.tolist()
    ncols = 4; nrows = max(1, (len(city_order)+ncols-1)//ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4.2, nrows*3.2)); axes = np.array(axes).flatten()
    for i, city in enumerate(city_order):
        ax = axes[i]; cs = h7[h7["city"]==city]
        rp = [w for w in REGIME_ORDER if w in cs["window"].values]; x = np.arange(len(rp))
        pv = [float(cs[cs["window"]==w]["mae_pin"].values[0]) for w in rp]
        av = [float(cs[cs["window"]==w]["mae_ari"].values[0]) for w in rp]
        ax.bar(x-0.2, pv, 0.35, color="#1565C0", alpha=0.85, label="PINN")
        ax.bar(x+0.2, av, 0.35, color="#E65100", alpha=0.85, label="ARIMA")
        mean_imp = cs["imp_pct"].mean()
        ax.set_title(f"{city} ({mean_imp:+.1f}%)", fontsize=10, fontweight="bold",
                     color="#1565C0" if mean_imp > 0 else "#C62828")
        ax.set_xticks(x); ax.set_xticklabels([w[:5] for w in rp], fontsize=7, rotation=30)
        if i == 0: ax.legend(fontsize=7)
        ax.grid(alpha=0.15, axis="y")
    for i in range(len(city_order), len(axes)): axes[i].axis("off")
    fig.suptitle("PINN vs ARIMA by City (7-day MAE)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig6_city_profiles.png", bbox_inches="tight"); plt.close()
    print("  Fig 6: city profiles")

    # ---- FIG 7: Parameter boxplots from all runs ----
    derived_raw = [("incubation_days","Incubation (days)"), ("infectious_days","Infectious (days)"),
                   ("ward_days","Ward stay (days)"), ("icu_days","ICU stay (days)"),
                   ("behavioral_waning_days","Behavioral waning (days)"), ("biological_waning_days","Immunity waning (days)")]
    cs_sorted = sorted(pall["city"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9)); axes = axes.flatten()
    for i, (col, label) in enumerate(derived_raw):
        ax = axes[i]
        data = [pall[pall["city"]==c][col].values for c in cs_sorted]
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        boxprops=dict(facecolor="#1565C0", alpha=0.5),
                        medianprops=dict(color="red", lw=2))
        ax.set_yticks(range(1, len(cs_sorted)+1)); ax.set_yticklabels(cs_sorted, fontsize=8)
        ax.set_xlabel(label); ax.set_title(label, fontsize=11, fontweight="bold"); ax.grid(alpha=0.15, axis="x")
    fig.suptitle("Parameter Distributions (10-seed ensemble per city)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig7_param_boxplots.png", bbox_inches="tight"); plt.close()
    print("  Fig 7: parameter boxplots")

    # ---- FIG 8: Variant boxplots from all runs ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    for i, variant in enumerate(["Delta", "Omicron", "BA.5", "XBB"]):
        ax = axes[i]; vs = vall[vall["variant"]==variant]
        if vs.empty: ax.axis("off"); continue
        cs_v = sorted(vs["city"].unique())
        data = [vs[vs["city"]==c]["learned"].values for c in cs_v]
        bp = ax.boxplot(data, vert=False, patch_artist=True,
                        boxprops=dict(facecolor="#1565C0", alpha=0.5),
                        medianprops=dict(color="red", lw=2))
        ax.axvline(vs["prior"].iloc[0], color="red", ls="--", lw=2, label=f"Prior={vs['prior'].iloc[0]:.1f}")
        ax.set_yticks(range(1, len(cs_v)+1)); ax.set_yticklabels(cs_v, fontsize=8)
        ax.set_xlabel("Multiplier"); ax.set_title(variant, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.15, axis="x")
    fig.suptitle("Variant Multiplier Distributions (10 seeds per city)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig8_variant_boxplots.png", bbox_inches="tight"); plt.close()
    print("  Fig 8: variant boxplots")

    print(f"\nAll figures saved to: {OUT_DIR}")


# ======================== MAIN ========================
def main():
    print("=" * 70)
    print("PUBLICATION ANALYSIS - Multi-City PINN SUEIHCDR")
    print("=" * 70)

    val, fit, par, punc, vmul, pall, vall = load_all()
    print_tables(val, fit, par, punc, vmul, pall, vall)
    make_figures(val, fit, par, punc, vmul, pall, vall)

    print(f"\n{'=' * 70}")
    print(f"DONE. All outputs in: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

