#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 18B: Real-time 8-week scenario projections, 2020 to 2021 (PHASE18B_PROTOCOL.md).
No new fitting. Reads the official Phase 17B fits (phase17b_layer3b/params_states.csv,
trajectories.csv, forecasts_weekly.csv), rebuilds each Saturday origin's causal inputs
with B.Problem3B (exactly as layer3b_sero.py and scenarios_phase18.py do), and runs
8-week projections from the fitted state on t0 using B.simulate3.

Scenarios posed at each origin t0 (2020-12-12 to 2021-10-02, 43 origins per county):
  P0: status quo (transmission and vaccination pace as known on t0)
  P1: transmission 20 % lower from t0 + 7 days
  P2: transmission 20 % higher from t0 + 7 days
  P3: vaccination pace 1.5 times the pace on t0 (origins with doses only)
  P4: vaccination pace 0.5 times the pace on t0 (origins with doses only)
  P0_HINDSIGHT (E3): P0 with the doses actually given (known only later)
"""

import argparse
import datetime as dt
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import layer2_admissions as L2
import layer3b_sero as B

ORIGINS_START = "2020-12-12"
ORIGINS_END = "2021-10-02"
HWEEKS_18B = 8
CHAINS = [("L3B", 1), ("L3B", 0), ("L3B_NOBEHAV", 1)]
PRIMARY = ("L3B", 1)
SCENARIOS = ["P0", "P1", "P2", "P3", "P4", "P0_HINDSIGHT"]
ENV_SCENARIOS = ["P0", "P1", "P2", "P3", "P4"]
STATE = ["S", "U_v", "U_b", "E", "I", "R_pre", "R_O1", "R_O2", "CUM"]


def sha(p):
    return B.sha(p)


def build_origin_problem(D, county, name, delay, t0, r, om, cdc, vax_cum, rounds, ca):
    """
    Rebuild the exact Problem3B object at origin t0 covering the 8 target weeks
    ahead (h = 1..8 after the last causally known admission week on t0).
    """
    behav = not name.endswith("_NOBEHAV")
    K = L2.known_inputs(D, t0, delay)
    last_ws = K["wk"].dropna().index.max()
    target_weeks = [last_ws + pd.Timedelta(days=7 * h) for h in range(1, HWEEKS_18B + 1)]
    end_day = max(target_weeks[-1] + pd.Timedelta(days=6), t0 + pd.Timedelta(days=7 * HWEEKS_18B - 1))
    n_extra = (end_day - (t0 - B.DAY)).days

    Pr = B.Problem3B(
        K,
        D,
        t0,
        np.zeros(9),
        behav,
        n_extra,
        om,
        cdc,
        B.booster_weight(vax_cum, t0),
        rho_sero=B.rho_sero_at(t0, rounds, ca)[0],
    )
    return K, Pr, last_ws, target_weeks, end_day


def run_origin_scenarios(Pr, D, r, t0, target_weeks):
    """
    Simulate from t0 (index i0 = L2.WINDOW in Pr.days) to end_day under P0..P4 and P0_HINDSIGHT.
    Also verifies that integrating from w0 reproduces x_t0 and the h=1..4 forecasts exactly.
    """
    i0 = L2.WINDOW
    n_proj = Pr.n - i0  # days from t0 to end_day inclusive
    days_proj = Pr.days[i0:]

    # Starting state x_t0 at t0 from params_states.csv
    x_t0 = np.array([float(r[f"{k}_t0"]) for k in STATE], dtype=float)

    beta_last = float(r["beta_last"])
    ab = float(r["a_b"])
    omb = float(r["omega_b"])
    ihr = float(r["ihr"])
    rho = float(r["rho"])
    ve = float(Pr.ve)
    omv = float(Pr.omv)
    f_dict = dict(Pr.f)
    N = float(Pr.N)

    # Causal baseline inputs from t0 onward (already held constant by Problem.__init__)
    beta_base = np.full(n_proj, beta_last, dtype=float)
    dose_base = Pr.dose_d[i0:Pr.n].copy()
    s_base = Pr.s_d[i0:Pr.n].copy()
    sh_proj = tuple(a[i0:Pr.n].copy() for a in Pr.sh)

    has_doses = bool(np.any(dose_base > 0.0))

    # Hindsight realized doses (7-day rolling mean or daily series in D["doses"])
    doses_full = D["doses"]
    dose_hindsight = np.array(
        [float(doses_full.get(d, 0.0)) for d in days_proj], dtype=float
    )

    scen_specs = {
        "P0": (beta_base, dose_base),
        "P1": (
            np.where(np.arange(n_proj) >= 7, 0.8 * beta_last, beta_last),
            dose_base,
        ),
        "P2": (
            np.where(np.arange(n_proj) >= 7, 1.2 * beta_last, beta_last),
            dose_base,
        ),
        "P3": (beta_base, 1.5 * dose_base),
        "P4": (beta_base, 0.5 * dose_base),
        "P0_HINDSIGHT": (beta_base, dose_hindsight),
    }

    # For delay=1, target_weeks[0] starts at last_ws + 7d = t0 - 6d (6 days before t0!).
    # Wait: in layer2/layer3b, target_weeks[0] can start before t0 when delay=1 (since last_ws is
    # the Sunday >= 13 days before Saturday t0, so last_ws + 7d is the Sunday 6 days before t0).
    # Let's reconstruct the full trajectory from w0 so any days of week h=1 before t0 come from
    # the fitted window trajectory X[:i0], and days from t0 onward come from the scenario projection!
    logk = np.log(np.array(json.loads(r["beta_knots"]), float))
    beta_win = L2.beta_series(logk, i0)
    # Reconstruct window admission series up to t0 - 1 from the fitted parameters
    # We also have x_t0 at index i0. To get exact window days w0 .. t0-1, we can either read
    # trajectories.csv or integrate from t0 onward and prepend the fitted window from trajectories.csv.
    out = {}
    for sc, (b_vec, d_vec) in scen_specs.items():
        X_proj, fE_proj, fI_proj = B.simulate3(
            x_t0, b_vec, d_vec, s_base, sh_proj, ab, omb, omv, ve, f_dict
        )
        adm_proj = fI_proj * N * ihr
        cases_proj = fE_proj * N * rho
        inf_proj = fE_proj * N
        pop_err = float(np.max(np.abs(X_proj[:, :8].sum(axis=1) - 1.0)))
        out[sc] = {
            "days": days_proj,
            "X": X_proj,
            "adm": pd.Series(adm_proj, index=days_proj),
            "cases": pd.Series(cases_proj, index=days_proj),
            "inf": pd.Series(inf_proj, index=days_proj),
            "pop_err": pop_err,
            "has_doses": has_doses,
        }

    return out, has_doses, float(dose_base[0]) if len(dose_base) else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 18B real-time scenario projections")
    ap.add_argument("--root", default=".")
    ap.add_argument("--adm-dir", default="phase15_admissions")
    ap.add_argument(
        "--vax",
        default="data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv",
    )
    ap.add_argument("--fits", default="phase17b_layer3b")
    ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
    ap.add_argument("--chains", default="all", help="smoke tests only")
    ap.add_argument("--max-origins", type=int, default=0, help="smoke tests only")
    ap.add_argument("--out", default="phase18b_projections_v2")
    ap.add_argument("--overwrite", action="store_true")
    A = ap.parse_args(argv)

    out = Path(A.out)
    if out.exists() and any(out.iterdir()) and not A.overwrite:
        sys.exit(f"{out} exists and is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run_log.txt", "w", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t_start_all = time.time()
    inputs = set()
    root = A.root
    counties = [c.strip() for c in A.counties.split(",") if c.strip()]
    chains = (
        CHAINS
        if A.chains == "all"
        else [c for c in CHAINS if f"{c[0]}:{c[1]}" in A.chains.split(",")]
    )
    vax_path = str(Path(root) / A.vax)

    pp = Path(root) / A.fits / "params_states.csv"
    tp = Path(root) / A.fits / "trajectories.csv"
    fp = Path(root) / A.fits / "forecasts_weekly.csv"
    inputs.update([str(pp), str(tp), str(fp)])

    Pf = pd.read_csv(pp)
    Pf["origin_dt"] = pd.to_datetime(Pf["origin"])
    Tf = pd.read_csv(
        tp,
        usecols=["county", "model", "delay", "origin", "date"]
        + STATE
        + ["adm_model", "cases_model"],
    )
    Tf["origin_dt"] = pd.to_datetime(Tf["origin"])
    Tf["date"] = pd.to_datetime(Tf["date"])
    Ff = pd.read_csv(fp)

    om, cdc = B.load_variants(root, inputs)
    rounds = B.load_commercial_rounds(root, inputs)
    ca = B.load_ca_cases(root, inputs)

    log(
        f"Phase 18B | fits '{A.fits}' | counties {counties} | "
        f"origins {ORIGINS_START}..{ORIGINS_END} | chains {chains}"
    )

    weekly_rows = []
    summary_rows = []
    repro_checks = []

    for county in counties:
        D = L2.load_county(root, county, A.adm_dir, vax_path, inputs)
        vc = B.load_vax_cum(vax_path, county, inputs)
        N = float(D["N"])
        wk_obs_series = D["wk"].dropna()

        for name, delay in chains:
            t_c0 = time.time()
            sub_p = Pf[
                (Pf["county"] == county)
                & (Pf["model"] == name)
                & (Pf["delay"] == delay)
                & (Pf["origin_dt"] >= pd.Timestamp(ORIGINS_START))
                & (Pf["origin_dt"] <= pd.Timestamp(ORIGINS_END))
            ].sort_values("origin_dt")

            if A.max_origins:
                sub_p = sub_p.iloc[: int(A.max_origins)]

            max_rel_p0 = 0.0
            max_rel_state = 0.0
            max_pop_err = 0.0
            n_doses_origins = 0

            for _, r in sub_p.iterrows():
                t0 = pd.Timestamp(r["origin_dt"])
                t0_str = str(t0.date())

                K, Pr, last_ws, target_weeks, end_day = build_origin_problem(
                    D, county, name, delay, t0, r, om, cdc, vc, rounds, ca
                )

                # Extract fitted trajectory on window [w0, t0 - 1] from trajectories.csv
                tr_orig = Tf[
                    (Tf["county"] == county)
                    & (Tf["model"] == name)
                    & (Tf["delay"] == delay)
                    & (Tf["origin_dt"] == t0)
                ].set_index("date")

                win_days = Pr.days[: L2.WINDOW]
                adm_win = tr_orig.loc[win_days, "adm_model"]
                # Verify state at t0 in trajectories.csv vs params_states.csv
                x_t0_traj = tr_orig.loc[t0, STATE].to_numpy(float)
                x_t0_param = np.array([float(r[f"{k}_t0"]) for k in STATE], float)
                rel_st = float(
                    np.max(np.abs(x_t0_traj - x_t0_param))
                    / max(1e-12, np.max(np.abs(x_t0_param)))
                )
                max_rel_state = max(max_rel_state, rel_st)

                # Run 8-week scenario projections from t0
                scen_out, has_doses, dose_t0 = run_origin_scenarios(
                    Pr, D, r, t0, target_weeks
                )
                if has_doses:
                    n_doses_origins += 1

                for sc in SCENARIOS:
                    max_pop_err = max(max_pop_err, scen_out[sc]["pop_err"])

                # Combine window admissions (for any part of h=1 before t0) with projected admissions
                adm_full_by_scen = {}
                inf_full_by_scen = {}
                for sc in SCENARIOS:
                    adm_full_by_scen[sc] = pd.concat([adm_win, scen_out[sc]["adm"]])
                    inf_full_by_scen[sc] = scen_out[sc]["inf"]

                # Compute weekly means on the 8 target_weeks (exactly as l3_origin does)
                p0_wk = {
                    h: float(
                        adm_full_by_scen["P0"]
                        .loc[ws : ws + pd.Timedelta(days=6)]
                        .mean()
                    )
                    for h, ws in enumerate(target_weeks, 1)
                }

                # Check P0 against saved forecasts_weekly.csv for h = 1..4
                fc_orig = Ff[
                    (Ff["county"] == county)
                    & (Ff["model"] == name)
                    & (Ff["delay"] == delay)
                    & (Ff["origin"] == t0_str)
                ]
                for _, fr in fc_orig.iterrows():
                    h_i = int(fr["h"])
                    if h_i in p0_wk and pd.notna(fr["pred"]):
                        rel_p = abs(p0_wk[h_i] - float(fr["pred"])) / max(
                            1e-12, abs(float(fr["pred"]))
                        )
                        max_rel_p0 = max(max_rel_p0, rel_p)

                # Naive projection: last known observed admission week on t0 held constant
                naive_val = float(K["wk"].dropna().iloc[-1])
                # Ex-ante growth vs decline from last two causally known weeks
                wk_known = K["wk"].dropna()
                ex_ante_phase = (
                    "growth"
                    if (len(wk_known) >= 2 and float(wk_known.iloc[-1]) > float(wk_known.iloc[-2]))
                    else "decline"
                )

                # 8-week totals (over the 56 projected days [t0, t0 + 55d])
                days_56 = pd.date_range(t0, periods=7 * HWEEKS_18B, freq="D")
                p0_tot_adm = float(scen_out["P0"]["adm"].loc[days_56].sum())
                p0_tot_inf = float(scen_out["P0"]["inf"].loc[days_56].sum())
                p0_peak_wk = int(max(p0_wk, key=p0_wk.get))
                p0_peak_val = float(p0_wk[p0_peak_wk])

                for sc in SCENARIOS:
                    sc_wk = {
                        h: float(
                            adm_full_by_scen[sc]
                            .loc[ws : ws + pd.Timedelta(days=6)]
                            .mean()
                        )
                        for h, ws in enumerate(target_weeks, 1)
                    }
                    for h, ws in enumerate(target_weeks, 1):
                        obs_val = float(wk_obs_series.get(ws, np.nan))
                        realized_phase = (
                            "growth"
                            if (pd.notna(obs_val) and obs_val > naive_val)
                            else "decline"
                        )
                        # Infections in week h (for days >= t0)
                        ws_clip = max(ws, t0)
                        we_clip = ws + pd.Timedelta(days=6)
                        inf_wk = float(
                            inf_full_by_scen[sc].loc[ws_clip:we_clip].sum()
                        )
                        weekly_rows.append(
                            {
                                "county": county,
                                "model": name,
                                "delay": int(delay),
                                "origin": t0_str,
                                "scenario": sc,
                                "h": int(h),
                                "week_start": str(ws.date()),
                                "pred": sc_wk[h],
                                "pred_P0": p0_wk[h],
                                "diff_vs_P0": sc_wk[h] - p0_wk[h],
                                "pct_vs_P0": 100.0
                                * (sc_wk[h] - p0_wk[h])
                                / max(1e-12, p0_wk[h]),
                                "obs": obs_val,
                                "naive_pred": naive_val,
                                "inf_week": inf_wk,
                                "has_doses": has_doses,
                                "dose_rate_t0": dose_t0,
                                "ex_ante_phase": ex_ante_phase,
                                "realized_phase": realized_phase,
                            }
                        )

                    tot_adm = float(scen_out[sc]["adm"].loc[days_56].sum())
                    tot_inf = float(scen_out[sc]["inf"].loc[days_56].sum())
                    pk_wk = int(max(sc_wk, key=sc_wk.get))
                    pk_val = float(sc_wk[pk_wk])

                    summary_rows.append(
                        {
                            "county": county,
                            "model": name,
                            "delay": int(delay),
                            "origin": t0_str,
                            "scenario": sc,
                            "has_doses": has_doses,
                            "dose_rate_t0": dose_t0,
                            "total_adm_8w": tot_adm,
                            "diff_total_adm_vs_P0": tot_adm - p0_tot_adm,
                            "pct_total_adm_vs_P0": 100.0
                            * (tot_adm - p0_tot_adm)
                            / max(1e-12, p0_tot_adm),
                            "peak_h": pk_wk,
                            "peak_shift_weeks_vs_P0": pk_wk - p0_peak_wk,
                            "peak_weekly_adm": pk_val,
                            "pct_peak_adm_vs_P0": 100.0
                            * (pk_val - p0_peak_val)
                            / max(1e-12, p0_peak_val),
                            "total_inf_8w": tot_inf,
                            "pct_total_inf_vs_P0": 100.0
                            * (tot_inf - p0_tot_inf)
                            / max(1e-12, p0_tot_inf),
                        }
                    )

            repro_checks.append(
                {
                    "county": county,
                    "model": name,
                    "delay": int(delay),
                    "n_origins": int(len(sub_p)),
                    "n_origins_with_doses": int(n_doses_origins),
                    "max_rel_p0_vs_saved_h1_4": float(max_rel_p0),
                    "max_rel_state_t0": float(max_rel_state),
                    "max_pop_err": float(max_pop_err),
                }
            )

            w_tmp = pd.DataFrame(weekly_rows)
            w_sub = w_tmp[
                (w_tmp["county"] == county)
                & (w_tmp["model"] == name)
                & (w_tmp["delay"] == delay)
                & (w_tmp["scenario"].isin(ENV_SCENARIOS))
            ]
            piv = w_sub.pivot_table(
                index=["origin", "h", "obs"], columns="scenario", values="pred"
            ).reset_index()
            piv["lo"] = piv[ENV_SCENARIOS].min(axis=1)
            piv["hi"] = piv[ENV_SCENARIOS].max(axis=1)
            piv["in_env"] = (piv["obs"] >= piv["lo"]) & (piv["obs"] <= piv["hi"])
            c14 = float(piv[piv["h"] <= 4]["in_env"].mean())
            c58 = float(piv[piv["h"] >= 5]["in_env"].mean())
            log(
                f"  {county} {name} d{delay}: {len(sub_p)} origins ({n_doses_origins} w/ doses), "
                f"E1 cov w1-4 {c14:.3f}, w5-8 {c58:.3f}, "
                f"P0 rel diff {max_rel_p0:.2e} ({time.time() - t_c0:.1f} s)"
            )

    W = pd.DataFrame(weekly_rows)
    S = pd.DataFrame(summary_rows)

    # E1: Envelope coverage
    e1_rows = []
    for (county, name, delay), grp in W[W["scenario"].isin(ENV_SCENARIOS)].groupby(
        ["county", "model", "delay"], sort=False
    ):
        piv = grp.pivot_table(
            index=["origin", "h", "obs"], columns="scenario", values="pred"
        ).reset_index()
        piv["lo"] = piv[ENV_SCENARIOS].min(axis=1)
        piv["hi"] = piv[ENV_SCENARIOS].max(axis=1)
        piv["in_env"] = (piv["obs"] >= piv["lo"]) & (piv["obs"] <= piv["hi"])
        c14 = float(piv[piv["h"] <= 4]["in_env"].mean())
        c58 = float(piv[piv["h"] >= 5]["in_env"].mean())
        c18 = float(piv["in_env"].mean())
        e1_rows.append(
            {
                "county": county,
                "model": name,
                "delay": int(delay),
                "n_origins": int(piv["origin"].nunique()),
                "cov_w1_4": c14,
                "cov_w5_8": c58,
                "cov_w1_8": c18,
                "E1_pass": bool(c14 >= 0.50),
            }
        )
    E1 = pd.DataFrame(e1_rows)

    # E2: Status quo drift by week ahead
    e2_rows = []
    p0_df = W[W["scenario"] == "P0"].copy()
    p0_df["log_p0_obs"] = np.log(
        np.maximum(1e-9, p0_df["pred"]) / np.maximum(1e-9, p0_df["obs"])
    )
    p0_df["log_naive_obs"] = np.log(
        np.maximum(1e-9, p0_df["naive_pred"]) / np.maximum(1e-9, p0_df["obs"])
    )
    for (county, name, delay, h), grp in p0_df.groupby(
        ["county", "model", "delay", "h"], sort=False
    ):
        for pcol in ("realized_phase", "ex_ante_phase"):
            for pval in ("all", "growth", "decline"):
                sub = grp if pval == "all" else grp[grp[pcol] == pval]
                if len(sub) == 0:
                    continue
                e2_rows.append(
                    {
                        "county": county,
                        "model": name,
                        "delay": int(delay),
                        "h": int(h),
                        "phase_definition": pcol,
                        "phase": pval,
                        "n_pairs": int(len(sub)),
                        "median_log_p0_over_obs": float(sub["log_p0_obs"].median()),
                        "iqr_log_p0_over_obs": float(
                            sub["log_p0_obs"].quantile(0.75)
                            - sub["log_p0_obs"].quantile(0.25)
                        ),
                        "median_ratio_p0_over_obs": float(
                            np.exp(sub["log_p0_obs"].median())
                        ),
                        "median_log_naive_over_obs": float(
                            sub["log_naive_obs"].median()
                        ),
                        "median_ratio_naive_over_obs": float(
                            np.exp(sub["log_naive_obs"].median())
                        ),
                    }
                )
    E2 = pd.DataFrame(e2_rows)

    # E3: Hindsight vaccination vs status quo P0
    e3_rows = []
    p0_h = (
        W[W["scenario"].isin(["P0", "P0_HINDSIGHT"])]
        .pivot_table(
            index=["county", "model", "delay", "origin", "h", "obs"],
            columns="scenario",
            values="pred",
        )
        .reset_index()
    )
    for (county, name, delay), grp in p0_h.groupby(
        ["county", "model", "delay"], sort=False
    ):
        for wlabel, cond in [
            ("w1_4", grp["h"] <= 4),
            ("w5_8", grp["h"] >= 5),
            ("w1_8", grp["h"] >= 1),
        ]:
            s = grp[cond]
            mae_p0 = float(np.mean(np.abs(s["P0"] - s["obs"])))
            mae_hi = float(np.mean(np.abs(s["P0_HINDSIGHT"] - s["obs"])))
            mlog_p0 = float(
                np.median(
                    np.log(np.maximum(1e-9, s["P0"]) / np.maximum(1e-9, s["obs"]))
                )
            )
            mlog_hi = float(
                np.median(
                    np.log(
                        np.maximum(1e-9, s["P0_HINDSIGHT"])
                        / np.maximum(1e-9, s["obs"])
                    )
                )
            )
            mpct = float(
                np.median(
                    100.0
                    * (s["P0_HINDSIGHT"] - s["P0"])
                    / np.maximum(1e-9, s["P0"])
                )
            )
            e3_rows.append(
                {
                    "county": county,
                    "model": name,
                    "delay": int(delay),
                    "window": wlabel,
                    "mae_P0": mae_p0,
                    "mae_P0_HINDSIGHT": mae_hi,
                    "median_log_P0_over_obs": mlog_p0,
                    "median_log_HINDSIGHT_over_obs": mlog_hi,
                    "median_pct_diff_HINDSIGHT_vs_P0": mpct,
                }
            )
    E3 = pd.DataFrame(e3_rows)

    W.to_csv(out / "projections_weekly.csv", index=False)
    S.to_csv(out / "projections_summary.csv", index=False)
    E1.to_csv(out / "envelope_e1.csv", index=False)
    E2.to_csv(out / "drift_e2.csv", index=False)
    E3.to_csv(out / "hindsight_vax_e3.csv", index=False)
    pd.DataFrame(repro_checks).to_csv(out / "repro_check.csv", index=False)

    e1_prim = E1[(E1["model"] == PRIMARY[0]) & (E1["delay"] == PRIMARY[1])]
    n_pass_e1 = int(e1_prim["E1_pass"].sum())
    e1_met = bool(n_pass_e1 >= 2)
    max_repro_p0 = float(max(c["max_rel_p0_vs_saved_h1_4"] for c in repro_checks))
    max_pop = float(max(c["max_pop_err"] for c in repro_checks))

    crit = {
        "rules": {
            "E1": "Share of (origin, week) pairs where observed weekly admissions fall in [min(P0..P4), max(P0..P4)] >= 50% for weeks 1 to 4 in >= 2 of 3 counties",
            "E2": "Median log(P0 / observed) by week ahead, split by growth and decline weeks, vs naive persistence",
            "E3": "P0 rerun with realized doses actually given (P0_HINDSIGHT)",
        },
        "E1_counties_passed": n_pass_e1,
        "E1_met": e1_met,
        "E1_primary": e1_prim.to_dict(orient="records"),
        "repro_check": {
            "max_rel_p0_vs_saved_h1_4": max_repro_p0,
            "max_abs_pop_error": max_pop,
            "by_chain": repro_checks,
        },
    }
    (out / "criteria.json").write_text(json.dumps(crit, indent=2))

    log("\nE1 envelope coverage (primary chain L3B d1):")
    log(
        e1_prim[
            ["county", "n_origins", "cov_w1_4", "cov_w5_8", "cov_w1_8", "E1_pass"]
        ].to_string(index=False)
    )
    log(
        f"repro check: max_rel_p0_vs_saved_h1_4={max_repro_p0:.3e} | "
        f"population error {max_pop:.3e} | E1_met={e1_met}"
    )

    me = Path(__file__).resolve()
    import scipy
    import statsmodels

    man = {
        "script": me.name,
        "script_sha256": sha(me),
        "layer3b_sha256": sha(Path(B.__file__)),
        "layer2_sha256": sha(Path(L2.__file__)),
        "command": " ".join([sys.executable] + sys.argv),
        "args": vars(A),
        "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t_start_all, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "inputs_sha256": {p: sha(p) for p in sorted(inputs)},
        "outputs_sha256": {
            f.name: sha(f)
            for f in sorted(out.iterdir())
            if f.is_file() and f.name not in ("run_log.txt", "run_manifest.json")
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    log(f"\nSaved to {out}/")
    logf.close()


if __name__ == "__main__":
    main()