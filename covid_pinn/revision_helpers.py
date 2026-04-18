import os
# ============================================================
# MASTER LOADER — ALL CITIES (outputs + uncertainty + regime)
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ---------------------------
# SET YOUR ROOT PATHS HERE
# ---------------------------
ROOT = Path(os.environ.get("PINN_DATA_PATH", "."))
RESULTS_DIR = ROOT / "Resultados_Cidades_02152026"

# Where to save master tables (directory created lazily by build_master_tables())
OUTDIR = RESULTS_DIR / "MASTER_TABLES"

# ---------------------------
# HELPERS
# ---------------------------
def daily_from_cum(cum):
    cum = np.asarray(cum, dtype=float)
    if cum.size == 0:
        return cum
    d = np.diff(np.r_[cum[0], cum])
    return np.clip(d, 0, None)

def weekly_avg(x, k=7):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    return pd.Series(x).rolling(k, min_periods=1).mean().to_numpy()

def safe_mape(y_true, y_pred, floor=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom)) * 100.0

def city_from_outputs_folder(folder_name: str):
    # e.g. outputs_SUEIHCDR_PUBLICATION_v2_Moscow -> Moscow
    # robust: take last chunk after final underscore
    if "_" not in folder_name:
        return None
    return folder_name.rsplit("_", 1)[-1].strip()

def city_from_uncert_folder(folder_name: str):
    # e.g. parameter_uncertainty_Moscow -> Moscow
    prefix = "parameter_uncertainty_"
    if not folder_name.startswith(prefix):
        return None
    return folder_name[len(prefix):].strip()

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------
# MAIN ENTRY POINT
# ---------------------------
def build_master_tables():
    """Discover city output folders, collect per-city results, and save master tables.

    This function performs all file-system side effects (directory creation,
    reading per-city CSV/JSON outputs, writing master CSVs). It is invoked
    from run scripts and from ``python -m covid_pinn.revision_helpers``;
    importing this module does NOT execute it.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # DISCOVER CITIES
    # ---------------------------
    output_folders = sorted([p for p in ROOT.glob("outputs_SUEIHCDR_PUBLICATION_v2_*") if p.is_dir()])
    cities = []
    for p in output_folders:
        c = city_from_outputs_folder(p.name)
        if c:
            cities.append(c)

    cities = sorted(set(cities))

    print(f"[INFO] ROOT: {ROOT}")
    print(f"[INFO] RESULTS_DIR: {RESULTS_DIR}")
    print(f"[INFO] Found output folders: {len(output_folders)}")
    print(f"[INFO] Found cities: {cities}")

    if len(cities) == 0:
        print("\n[ERROR] No cities found. Check that your folders are like:")
        print(r"  C:\...\CODES_04032021\outputs_SUEIHCDR_PUBLICATION_v2_Moscow")
        print("and that ROOT is set correctly (set PINN_DATA_PATH env var).")
        return None

    # ---------------------------
    # COLLECT TABLES
    # ---------------------------
    rows_params_final = []
    rows_fit = []
    regime_all = []
    uncert_runs_all = []
    variant_runs_all = []

    for city in cities:
        # ---- model outputs folder ----
        outdir_city = ROOT / f"outputs_SUEIHCDR_PUBLICATION_v3_{city}"
        p_comp = outdir_city / f"compartments_counts_v3_{city}.csv"
        p_par  = outdir_city / f"parameters_final_v3_{city}.json"

        if not outdir_city.exists():
            print(f"[WARN] Missing outputs folder for {city}: {outdir_city}")
            continue

        # 1) parameters_final.json
        if p_par.exists():
            try:
                js = load_json(p_par)
                row = {"city": city, "source_folder": str(outdir_city)}
                # flatten: keep scalar-like values
                for k, v in js.items():
                    # sometimes json saves numbers as strings; try to coerce
                    try:
                        row[k] = float(v)
                    except Exception:
                        row[k] = v
                rows_params_final.append(row)
            except Exception as e:
                print(f"[WARN] Failed reading {p_par} for {city}: {e}")
        else:
            print(f"[WARN] Missing parameters_final.json for {city}: {p_par}")

        # 2) compartments_counts.csv → fit metrics
        if p_comp.exists():
            try:
                dfc = pd.read_csv(p_comp)
                # expected columns from your exporter:
                # date, cum_cases_pred, obs_cases, plus compartments
                if "obs_cases" not in dfc.columns or "cum_cases_pred" not in dfc.columns:
                    print(f"[WARN] {city}: compartments_counts.csv missing obs_cases/cum_cases_pred columns")
                else:
                    obs_cum = dfc["obs_cases"].astype(float).to_numpy()
                    pred_cum = dfc["cum_cases_pred"].astype(float).to_numpy()

                    obs_daily7 = weekly_avg(daily_from_cum(obs_cum), 7)
                    pred_daily7 = weekly_avg(daily_from_cum(pred_cum), 7)

                    # align lengths (should match)
                    n = min(len(obs_daily7), len(pred_daily7))
                    obs_daily7 = obs_daily7[:n]
                    pred_daily7 = pred_daily7[:n]

                    mae7 = float(np.mean(np.abs(obs_daily7 - pred_daily7))) if n else np.nan
                    mape7 = safe_mape(obs_daily7, pred_daily7) if n else np.nan
                    me7 = float(np.mean(pred_daily7 - obs_daily7)) if n else np.nan

                    # also store last date + total period
                    rowf = {
                        "city": city,
                        "T_days": int(len(dfc)),
                        "date_start": str(pd.to_datetime(dfc["date"]).iloc[0].date()) if "date" in dfc.columns and len(dfc) else None,
                        "date_end": str(pd.to_datetime(dfc["date"]).iloc[-1].date()) if "date" in dfc.columns and len(dfc) else None,
                        "fit_mae_7d": mae7,
                        "fit_mape_7d_pct": mape7,
                        "fit_me_7d": me7,
                        "source_file": str(p_comp),
                    }
                    rows_fit.append(rowf)
            except Exception as e:
                print(f"[WARN] Failed reading {p_comp} for {city}: {e}")
        else:
            print(f"[WARN] Missing compartments_counts.csv for {city}: {p_comp}")

        # 3) regime validation metrics file (stored in Resultados_Cidades_02152026)
        p_reg = RESULTS_DIR / f"regime_validation_metrics_{city}.csv"
        if p_reg.exists():
            try:
                dfr = pd.read_csv(p_reg)
                dfr["city"] = city
                dfr["source_file"] = str(p_reg)
                regime_all.append(dfr)
            except Exception as e:
                print(f"[WARN] Failed reading {p_reg}: {e}")
        else:
            # not fatal
            pass

        # 4) parameter uncertainty folder
        udir = ROOT / f"parameter_uncertainty_{city}"
        p_u = udir / "parameter_uncertainty_results.csv"
        p_v = udir / "variant_multiplier_results.csv"

        if udir.exists():
            if p_u.exists():
                try:
                    dfu = pd.read_csv(p_u)
                    dfu["city"] = city
                    dfu["source_file"] = str(p_u)
                    uncert_runs_all.append(dfu)
                except Exception as e:
                    print(f"[WARN] Failed reading {p_u}: {e}")
            else:
                print(f"[WARN] Missing parameter_uncertainty_results.csv for {city}: {p_u}")

            if p_v.exists():
                try:
                    dfv = pd.read_csv(p_v)
                    dfv["city"] = city
                    dfv["source_file"] = str(p_v)
                    variant_runs_all.append(dfv)
                except Exception as e:
                    print(f"[WARN] Failed reading {p_v}: {e}")
            else:
                print(f"[WARN] Missing variant_multiplier_results.csv for {city}: {p_v}")
        else:
            # not fatal
            pass

    # ---------------------------
    # BUILD MASTER TABLES (safe on empty)
    # ---------------------------
    df_params_final = pd.DataFrame(rows_params_final)
    df_fit_metrics  = pd.DataFrame(rows_fit)

    df_regime_all = pd.concat(regime_all, ignore_index=True) if len(regime_all) else pd.DataFrame()
    df_uncert_runs = pd.concat(uncert_runs_all, ignore_index=True) if len(uncert_runs_all) else pd.DataFrame()
    df_variant_runs = pd.concat(variant_runs_all, ignore_index=True) if len(variant_runs_all) else pd.DataFrame()

    if not df_params_final.empty and "city" in df_params_final.columns:
        df_params_final = df_params_final.sort_values("city").reset_index(drop=True)
    if not df_fit_metrics.empty and "city" in df_fit_metrics.columns:
        df_fit_metrics = df_fit_metrics.sort_values("city").reset_index(drop=True)

    # -------------- regime summaries --------------
    df_regime_summary = pd.DataFrame()
    if not df_regime_all.empty:
        # Typical columns in your regime_validation_metrics: window, cut_date, horizon, mae_pin, mae_ari, mae_hyb, etc.
        # We'll compute per-city mean over windows for each horizon.
        cols = [c for c in df_regime_all.columns if c.startswith("mae_") or c.startswith("mape_") or c in ["w_arima","me_pin","me_ari","me_hyb"]]
        grp = df_regime_all.groupby(["city","horizon"], as_index=False)[cols].mean(numeric_only=True)
        df_regime_summary = grp.sort_values(["city","horizon"]).reset_index(drop=True)

    # -------------- uncertainty summaries --------------
    df_uncert_summary = pd.DataFrame()
    if not df_uncert_runs.empty:
        # summarize numeric parameters per city
        ignore_cols = {"run_id","seed","elapsed_sec","source_file"}
        num_cols = [c for c in df_uncert_runs.columns
                    if c not in ignore_cols and c != "city" and pd.api.types.is_numeric_dtype(df_uncert_runs[c])]

        # mean/std/CI
        def q025(x): return x.quantile(0.025)
        def q975(x): return x.quantile(0.975)

        agg = df_uncert_runs.groupby("city")[num_cols].agg(["mean","std",q025,q975])
        # flatten multiindex columns
        agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
        df_uncert_summary = agg.reset_index().sort_values("city").reset_index(drop=True)

    # -------------- variant multiplier summaries --------------
    df_variant_summary = pd.DataFrame()
    if not df_variant_runs.empty:
        # expected columns: variant, prior, learned, ratio
        keep = [c for c in ["city","variant","prior","learned","ratio"] if c in df_variant_runs.columns]
        dv = df_variant_runs[keep].copy()
        if "learned" in dv.columns:
            df_variant_summary = dv.groupby(["city","variant"], as_index=False).agg(
                prior=("prior","first") if "prior" in dv.columns else ("variant","size"),
                learned_mean=("learned","mean"),
                learned_std=("learned","std"),
                learned_q025=("learned", lambda x: x.quantile(0.025)),
                learned_q975=("learned", lambda x: x.quantile(0.975)),
            ).sort_values(["city","variant"]).reset_index(drop=True)

    # ---------------------------
    # SAVE EVERYTHING
    # ---------------------------
    df_params_final.to_csv(OUTDIR / "MASTER_parameters_final_by_city.csv", index=False)
    df_fit_metrics.to_csv(OUTDIR / "MASTER_fit_metrics_from_compartments.csv", index=False)
    df_regime_all.to_csv(OUTDIR / "MASTER_regime_validation_all_rows.csv", index=False)
    df_regime_summary.to_csv(OUTDIR / "MASTER_regime_validation_summary_mean.csv", index=False)
    df_uncert_runs.to_csv(OUTDIR / "MASTER_parameter_uncertainty_all_runs.csv", index=False)
    df_uncert_summary.to_csv(OUTDIR / "MASTER_parameter_uncertainty_summary.csv", index=False)
    df_variant_runs.to_csv(OUTDIR / "MASTER_variant_multiplier_all_runs.csv", index=False)
    df_variant_summary.to_csv(OUTDIR / "MASTER_variant_multiplier_summary.csv", index=False)

    print("\n[OK] DONE")
    print(f"Saved master tables to: {OUTDIR}")

    print("\nQuick sanity:")
    print("  params_final rows:", len(df_params_final))
    print("  fit_metrics rows:", len(df_fit_metrics))
    print("  regime rows:", len(df_regime_all))
    print("  uncert runs rows:", len(df_uncert_runs))
    print("  variant runs rows:", len(df_variant_runs))

    return {
        "params_final": df_params_final,
        "fit_metrics": df_fit_metrics,
        "regime_all": df_regime_all,
        "regime_summary": df_regime_summary,
        "uncert_runs": df_uncert_runs,
        "uncert_summary": df_uncert_summary,
        "variant_runs": df_variant_runs,
        "variant_summary": df_variant_summary,
    }


if __name__ == "__main__":
    build_master_tables()
