from .core import *

# ============================================================
# PARAMETER UNCERTAINTY (drop-in for your unified runner)
# - Runs N times per selected city (fit only, NO multi-window eval)
# - Saves:
#     parameter_uncertainty_results.csv
#     parameter_summary_for_table1.csv
#     variant_multiplier_results.csv
#     variant_multiplier_summary.csv
# - Integrates cleanly with your existing:
#     load_us_county_series / load_world_city_series
#     TrainCfg / train_sueihcdr_once
#     US_CITIES / WORLD_CITIES / OUT_PATH_STR / DEVICE / LR_FULL / SD_LAG_DAYS ...
# ============================================================

import json

# -----------------------------
# Uncertainty config
# -----------------------------
UNC_N_RUNS   = 10        # 10–20 is usually enough
UNC_BASESEED = 1337
UNC_SEED_STRIDE = 100
UNC_EPOCHS   = EPOCHS_FULL     # reuse
UNC_LR       = LR_FULL
UNC_SD_LAG   = SD_LAG_DAYS
UNC_MODE_DEFAULT = "fit_check" # this will *only* do fit-type training for uncertainty

def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _param_row_from_res(res, run_id, seed, elapsed):
    row = {
        "run_id": run_id,
        "seed": seed,
        "elapsed_sec": float(elapsed),

        # core parameters
        "beta0": float(res.get("beta0_final", np.nan)),
        "sigma": float(res.get("sigma_final", np.nan)),
        "delta": float(res.get("delta_final", np.nan)),
        "zeta":  float(res.get("zeta_final",  np.nan)),
        "epsi":  float(res.get("epsi_final",  np.nan)),
        "m":     float(res.get("m_final",     np.nan)),
        "c":     float(res.get("c_final",     np.nan)),
        "omega": float(res.get("omega_final", np.nan)),
        "eta":   float(res.get("eta_final",   np.nan)),
        "rho":   float(res.get("rho_final",   np.nan)),
    }

    # derived durations (days)
    def inv(x): 
        return (1.0/x) if (np.isfinite(x) and x > 0) else np.nan

    row["incubation_days"]          = inv(row["sigma"])
    row["infectious_days"]          = inv(row["delta"])
    row["ward_days"]                = inv(row["zeta"])
    row["icu_days"]                 = inv(row["epsi"])
    row["behavioral_waning_days"]   = inv(row["omega"])
    row["biological_waning_days"]   = inv(row["eta"])
    return row

def _summarize_numeric(df: pd.DataFrame, cols):
    out = []
    for col, symbol, unit in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) == 0:
            continue
        out.append({
            "parameter": col,
            "symbol": symbol,
            "unit": unit,
            "n": int(len(x)),
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "ci_2.5": float(x.quantile(0.025)),
            "ci_97.5": float(x.quantile(0.975)),
            "min": float(x.min()),
            "max": float(x.max()),
        })
    return pd.DataFrame(out)

def run_parameter_uncertainty_for_city(label, df, Npop, n_runs=UNC_N_RUNS, base_seed=UNC_BASESEED):
    """
    Run N fits with different seeds for ONE city, save results to a dedicated folder.
    This does NOT run multi-window evaluation (fast).
    """
    outdir = Path(OUT_PATH_STR) / f"parameter_uncertainty_{label}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"PARAMETER UNCERTAINTY — {label}")
    print(f"Runs: {n_runs} | base_seed={base_seed} | device={DEVICE}")
    print("="*70)

    all_params = []
    all_variants = []

    for run_id in range(n_runs):
        seed = int(base_seed + run_id * UNC_SEED_STRIDE)
        _set_all_seeds(seed)

        print(f"\n[UNC] {label}  run {run_id+1}/{n_runs}  seed={seed}")
        t0 = time.time()

        cfg = TrainCfg(
            max_epochs=UNC_EPOCHS,
            lr=UNC_LR,
            sd_lag=UNC_SD_LAG,
            rollout_extra=0,
            validation_days=VALIDATION_DAYS,
            patience_epochs=PATIENCE_EPOCHS
        )

        try:
            res = train_sueihcdr_once(df, Npop, cfg, return_all=True, sd_future=None)
            elapsed = time.time() - t0
            print(f"  done in {elapsed/60:.1f} min")

            # store parameters
            all_params.append(_param_row_from_res(res, run_id, seed, elapsed))

            # store variant multipliers
            if "learned_variant_multipliers" in res and isinstance(res["learned_variant_multipliers"], dict):
                for vname, vals in res["learned_variant_multipliers"].items():
                    all_variants.append({
                        "run_id": run_id,
                        "seed": seed,
                        "variant": vname,
                        "prior": float(vals.get("prior", np.nan)),
                        "learned": float(vals.get("learned", np.nan)),
                        "ratio": float(vals.get("ratio", np.nan)),
                    })
                    print(f"    {vname}: learned={vals.get('learned', np.nan):.3f}  prior={vals.get('prior', np.nan):.2f}")

        except Exception as e:
            print(f"  ERROR in run {run_id+1}: {e}")
            traceback.print_exc()

    # --- save raw results ---
    params_df = pd.DataFrame(all_params)
    variants_df = pd.DataFrame(all_variants)

    params_df.to_csv(outdir / "parameter_uncertainty_results.csv", index=False)
    if len(variants_df):
        variants_df.to_csv(outdir / "variant_multiplier_results.csv", index=False)

    # --- summary for Table 1 ---
    param_cols = [
        ("beta0", "β₀", "day⁻¹"),
        ("sigma", "σ", "day⁻¹"),
        ("delta", "δ", "day⁻¹"),
        ("zeta",  "ζ", "day⁻¹"),
        ("epsi",  "ε", "day⁻¹"),
        ("m",     "m", ""),
        ("c",     "c", ""),
        ("omega", "ω", "day⁻¹"),
        ("eta",   "η", "day⁻¹"),
        ("rho",   "ρ", ""),

        ("incubation_days",        "1/σ", "days"),
        ("infectious_days",        "1/δ", "days"),
        ("ward_days",              "1/ζ", "days"),
        ("icu_days",               "1/ε", "days"),
        ("behavioral_waning_days", "1/ω", "days"),
        ("biological_waning_days", "1/η", "days"),
    ]

    summary_df = _summarize_numeric(params_df, param_cols)
    summary_df.to_csv(outdir / "parameter_summary_for_table1.csv", index=False)

    # --- variant summary ---
    if len(variants_df):
        v = variants_df.copy()
        def q025(x): return x.quantile(0.025)
        def q975(x): return x.quantile(0.975)
        v_sum = (
            v.groupby("variant")
             .agg(prior=("prior","first"),
                  learned_mean=("learned","mean"),
                  learned_std=("learned","std"),
                  learned_q025=("learned", q025),
                  learned_q975=("learned", q975))
             .reset_index()
        )
        v_sum.to_csv(outdir / "variant_multiplier_summary.csv", index=False)

    # --- quick console table ---
    print("\n" + "-"*70)
    print(f"SUMMARY (Table 1 ready) — {label}")
    print("-"*70)
    if len(summary_df):
        for _, r in summary_df.iterrows():
            sym = r["symbol"]; unit = r["unit"]
            mean = r["mean"]; sd = r["std"]
            lo = r["ci_2.5"]; hi = r["ci_97.5"]
            if unit == "days":
                print(f"{sym:<10} mean={mean:>8.1f}  sd={sd:>8.1f}  95%CI=[{lo:>7.1f},{hi:>7.1f}]")
            else:
                print(f"{sym:<10} mean={mean:>8.4f}  sd={sd:>8.4f}  95%CI=[{lo:>7.4f},{hi:>7.4f}]")
    else:
        print("No successful runs to summarize (summary_df empty).")

    print("\nSaved to:", outdir)
    return outdir, params_df, summary_df, variants_df


# ============================================================
# OPTIONAL: integrate into your existing CLI main()
# Add a new mode: "uncertainty"
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["fit_check","full","uncertainty"])
    parser.add_argument("--cities", default=None, help="Comma-separated city labels")
    parser.add_argument("--skip-us", action="store_true")
    parser.add_argument("--skip-world", action="store_true")
    parser.add_argument("--unc-runs", type=int, default=UNC_N_RUNS)
    parser.add_argument("--unc-base-seed", type=int, default=UNC_BASESEED)
    args, unknown = parser.parse_known_args()

    filt = set(c.strip() for c in args.cities.split(",")) if args.cities else None
    print(f"Device: {DEVICE} | Threads: {N_THREADS} | Mode: {args.mode}")
    summary = []

    if args.mode == "uncertainty":
        # Uncertainty = run ONLY the fit multiple times (no multiwindow eval)
        if not args.skip_us:
            for label, county, state, sub in US_CITIES:
                if filt and label not in filt: continue
                try:
                    df, Npop = load_us_county_series(county, state, sub)
                    t0 = time.time()
                    run_parameter_uncertainty_for_city(label, df, Npop, n_runs=args.unc_runs, base_seed=args.unc_base_seed)
                    summary.append({"city":label,"type":"US","status":"OK_UNC","time_min":(time.time()-t0)/60})
                except Exception as e:
                    print(f"\n!! FAILED UNC {label}: {e}")
                    summary.append({"city":label,"type":"US","status":f"FAIL_UNC: {e}","time_min":0})

        if not args.skip_world:
            for city in WORLD_CITIES:
                if filt and city not in filt: continue
                try:
                    df, Npop = load_world_city_series(city)
                    t0 = time.time()
                    run_parameter_uncertainty_for_city(city, df, Npop, n_runs=args.unc_runs, base_seed=args.unc_base_seed)
                    summary.append({"city":city,"type":"World","status":"OK_UNC","time_min":(time.time()-t0)/60})
                except Exception as e:
                    print(f"\n!! FAILED UNC {city}: {e}")
                    summary.append({"city":city,"type":"World","status":f"FAIL_UNC: {e}","time_min":0})

    else:
        # Your original behavior for fit_check / full
        if not args.skip_us:
            for label, county, state, sub in US_CITIES:
                if filt and label not in filt: continue
                try:
                    df, Npop = load_us_county_series(county, state, sub)
                    t0 = time.time()
                    run_single_city(label, df, Npop, args.mode)
                    summary.append({"city":label,"type":"US","status":"OK","time_min":(time.time()-t0)/60})
                except Exception as e:
                    print(f"\n!! FAILED {label}: {e}")
                    summary.append({"city":label,"type":"US","status":f"FAIL: {e}","time_min":0})

        if not args.skip_world:
            for city in WORLD_CITIES:
                if filt and city not in filt: continue
                try:
                    df, Npop = load_world_city_series(city)
                    t0 = time.time()
                    run_single_city(city, df, Npop, args.mode)
                    summary.append({"city":city,"type":"World","status":"OK","time_min":(time.time()-t0)/60})
                except Exception as e:
                    print(f"\n!! FAILED {city}: {e}")
                    summary.append({"city":city,"type":"World","status":f"FAIL: {e}","time_min":0})

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    sdf = pd.DataFrame(summary); print(sdf.to_string(index=False))
    sdf.to_csv(Path(OUT_PATH_STR)/"run_summary.csv", index=False)

#if __name__ == "__main__":
#    main()

