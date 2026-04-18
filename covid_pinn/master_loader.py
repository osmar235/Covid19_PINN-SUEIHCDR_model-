import os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, binomtest

# =========================
# INPUT / OUTPUT (paths only; directories created lazily in run_master_analysis())
# =========================
MASTER = Path(os.environ.get("PINN_DATA_PATH", "."))
OUTDIR = Path(os.environ.get("PINN_DATA_PATH", "."))

# =========================
# MODEL COLUMN MAP
# =========================
models = {
    "PINN":"mae_pin",
    "ARIMA":"mae_ari",
    "LogLinear":"mae_loglin",
    "BayesPoly":"mae_bayesian",
    "ETS":"mae_ets_manual",
    "Prophet":"mae_prophet",
}

# =========================
# HELPERS
# =========================
def holm_adj(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    adj = np.maximum.accumulate((m - np.arange(m)) * ps)
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out


# =========================
# MAIN ENTRY POINT
# =========================
def run_master_analysis(master_path: Path | None = None, outdir: Path | None = None):
    """Load the master metrics CSV, compute summary tables + figures, and write outputs.

    All file-system side effects (directory creation, reading the master CSV,
    writing tables + figures) happen inside this function. Importing this
    module does NOT execute it.

    Parameters
    ----------
    master_path : Path, optional
        Path to the master CSV. Defaults to the module-level ``MASTER``.
    outdir : Path, optional
        Output directory. Defaults to the module-level ``OUTDIR``.
    """
    master_path = Path(master_path) if master_path is not None else MASTER
    outdir = Path(outdir) if outdir is not None else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    # =========================
    # LOAD
    # =========================
    df = pd.read_csv(master_path)
    df["cut_date"] = pd.to_datetime(df["cut_date"], errors="coerce")
    df["horizon"]  = pd.to_numeric(df["horizon"], errors="coerce")
    df = df.dropna(subset=["city","window","cut_date","horizon"]).copy()
    df["horizon"] = df["horizon"].astype(int)

    for col in models.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mean_true"] = pd.to_numeric(df["mean_true"], errors="coerce")

    # Stabilize scale-normalization for early-wave near-zero series
    den = np.maximum(df["mean_true"].values.astype(float), 1.0)

    for name, col in models.items():
        if col in df.columns:
            df[f"rel_{name}"] = df[col] / den

    for name, col in models.items():
        if name == "ARIMA" or col not in df.columns:
            continue
        df[f"imp_{name}_vs_ARIMA"] = (df["mae_ari"] - df[col]) / df["mae_ari"]
        df[f"win_{name}"] = (df[col] < df["mae_ari"]).astype(int)

    # =========================
    # TABLE 1: OVERALL BY HORIZON (includes ETS/Prophet if present)
    # =========================
    rows = []
    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h]
        for name, col in models.items():
            if col not in sub.columns:
                continue
            x = sub[col].dropna()
            if len(x) == 0:
                continue
            row = {
                "horizon": h,
                "model": name,
                "n": int(len(x)),
                "MAE_mean": float(x.mean()),
                "MAE_median": float(x.median()),
                "relMAE_mean": float(sub[f"rel_{name}"].dropna().mean()) if f"rel_{name}" in sub.columns else np.nan,
                "relMAE_median": float(sub[f"rel_{name}"].dropna().median()) if f"rel_{name}" in sub.columns else np.nan,
            }
            if name != "ARIMA" and f"win_{name}" in sub.columns:
                w = sub.loc[x.index, f"win_{name}"].dropna()
                row["wins_vs_ARIMA"] = int(w.sum()) if len(w) else np.nan
                row["win_rate_vs_ARIMA"] = float(w.mean()) if len(w) else np.nan
            rows.append(row)

    table_overall = pd.DataFrame(rows).sort_values(["horizon", "model"])
    table_overall.to_csv(outdir / "table_overall_by_horizon.csv", index=False)
    print("Wrote:", outdir / "table_overall_by_horizon.csv")

    # =========================
    # TABLE 2: REGIME P-VALUES (PINN vs ARIMA) + HOLM
    # =========================
    pv = []
    for h in sorted(df["horizon"].unique()):
        for reg in sorted(df["window"].unique()):
            sub = df[(df.horizon == h) & (df.window == reg)].dropna(subset=["mae_pin", "mae_ari"])
            if len(sub) < 5:
                continue
            dif = (sub["mae_pin"] - sub["mae_ari"]).values
            try:
                p_w = float(wilcoxon(dif, alternative="two-sided").pvalue)
            except Exception:
                p_w = np.nan
            wins = int((sub["mae_pin"] < sub["mae_ari"]).sum())
            n = int(len(sub))
            p_s = float(binomtest(wins, n, 0.5, alternative="two-sided").pvalue)
            pv.append({
                "horizon": h, "regime": reg, "n": n,
                "wins_PINN": wins, "win_rate_PINN": wins / n,
                "mean_improvement_vs_ARIMA": float(((sub["mae_ari"] - sub["mae_pin"]) / sub["mae_ari"]).mean()),
                "p_wilcoxon": p_w, "p_sign": p_s,
            })

    pv = pd.DataFrame(pv)
    pv["p_wilcoxon_holm"] = np.nan
    for h in pv["horizon"].unique():
        idx = pv.index[pv["horizon"] == h]
        pv.loc[idx, "p_wilcoxon_holm"] = holm_adj(pv.loc[idx, "p_wilcoxon"].values)

    pv = pv.sort_values(["horizon", "regime"])
    pv.to_csv(outdir / "table_regime_pvals_holm_PINN_vs_ARIMA.csv", index=False)
    print("Wrote:", outdir / "table_regime_pvals_holm_PINN_vs_ARIMA.csv")

    # =========================
    # FIGURE: IMPROVEMENT BOXPLOTS BY REGIME (PINN vs ARIMA)
    # =========================
    canonical = ["FirstWave", "Winter20", "Delta", "Omicron", "BA5_Waning"]

    for h in sorted(df["horizon"].unique()):
        sub = df[df.horizon == h].dropna(subset=["mae_pin", "mae_ari"]).copy()
        sub["imp"] = (sub["mae_ari"] - sub["mae_pin"]) / sub["mae_ari"]
        regimes = [r for r in canonical if r in sub["window"].unique()] + [r for r in sub["window"].unique() if r not in canonical]
        data = [sub.loc[sub.window == r, "imp"].dropna().values for r in regimes]

        plt.figure(figsize=(9, 4))
        plt.boxplot(data, labels=regimes, showfliers=False)
        plt.axhline(0, linestyle="--")
        plt.ylabel("Improvement vs ARIMA (fraction)")
        plt.title(f"PINN improvement vs ARIMA by regime (h={h}d)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        outp = outdir / f"fig_pinn_improvement_boxplot_{h}d.png"
        plt.savefig(outp, dpi=220)
        plt.close()
        print("Wrote:", outp)

    # =========================
    # FIGURE: CITY BARS (Figure-6-like) for 7-day horizon (PINN vs ARIMA)
    # =========================
    h = 7
    sub = df[df.horizon == h].dropna(subset=["mae_pin", "mae_ari"]).copy()
    city_mean = sub.groupby("city")[["mae_pin", "mae_ari"]].mean()
    city_mean["imp"] = (city_mean["mae_ari"] - city_mean["mae_pin"]) / city_mean["mae_ari"]
    city_mean = city_mean.sort_values("imp", ascending=False)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(city_mean))
    w = 0.38
    plt.bar(x - w / 2, city_mean["mae_pin"].values, w, label="PINN")
    plt.bar(x + w / 2, city_mean["mae_ari"].values, w, label="ARIMA")
    plt.xticks(x, city_mean.index, rotation=45, ha="right")
    plt.ylabel("MAE (cases/day)")
    plt.title("City-level mean MAE across regimes (7-day horizon)")
    plt.legend()
    plt.tight_layout()
    outp = outdir / "fig_city_bars_pinn_vs_arima_7d.png"
    plt.savefig(outp, dpi=220)
    plt.close()
    print("Wrote:", outp)

    # Save enriched master (handy for later tables/plots)
    df.to_csv(outdir / "MASTER_with_derived.csv", index=False)
    print("Wrote:", outdir / "MASTER_with_derived.csv")

    print("\nDone. Outputs in:", outdir)

    return {
        "df": df,
        "table_overall": table_overall,
        "regime_pvals": pv,
    }


if __name__ == "__main__":
    run_master_analysis()
