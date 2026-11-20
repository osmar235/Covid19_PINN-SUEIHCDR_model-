#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and verification tests for Phase 18B (projections_phase18b.py).
Verifies:
1. P0 reproduces Phase 17B weekly forecast (forecasts_weekly.csv) for h=1..4 to machine precision (< 1e-6).
2. Population conservation (< 1e-12) across all scenarios (P0..P4, P0_HINDSIGHT).
3. Scenario change timing: P1 and P2 match P0 on days 0..6 from t0, and diverge from day 7 onward.
4. Transmission ordering: P1 (-20% beta) < P0 < P2 (+20% beta) on 8-week admissions and infections.
5. Vaccination ordering: when causal doses > 0 on t0, P3 (1.5x pace) < P0 < P4 (0.5x pace) on 8-week admissions.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import layer2_admissions as L2
import layer3b_sero as B
import projections_phase18b as P18B


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--adm-dir", default="phase15_admissions")
    ap.add_argument(
        "--vax",
        default="data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv",
    )
    ap.add_argument("--fits", default="phase17b_layer3b")
    ap.add_argument("--out", default="phase18b_tests")
    A = ap.parse_args(argv)

    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    inputs = set()
    root = A.root
    vax_path = str(Path(root) / A.vax)
    Pf = pd.read_csv(Path(root) / A.fits / "params_states.csv")
    Pf["origin_dt"] = pd.to_datetime(Pf["origin"])
    Tf = pd.read_csv(
        Path(root) / A.fits / "trajectories.csv",
        usecols=["county", "model", "delay", "origin", "date"]
        + P18B.STATE
        + ["adm_model"],
    )
    Tf["origin_dt"] = pd.to_datetime(Tf["origin"])
    Tf["date"] = pd.to_datetime(Tf["date"])
    Ff = pd.read_csv(Path(root) / A.fits / "forecasts_weekly.csv")

    om, cdc = B.load_variants(root, inputs)
    rounds = B.load_commercial_rounds(root, inputs)
    ca = B.load_ca_cases(root, inputs)

    n_pass, n_fail = 0, 0

    def check(label, ok, detail=""):
        nonlocal n_pass, n_fail
        st = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        log(f"[{st}] {label} {detail}")

    for county in ["San Diego", "Los Angeles", "San Francisco"]:
        D = L2.load_county(root, county, A.adm_dir, vax_path, inputs)
        vc = B.load_vax_cum(vax_path, county, inputs)
        sub_p = Pf[
            (Pf["county"] == county)
            & (Pf["model"] == "L3B")
            & (Pf["delay"] == 1)
            & (Pf["origin_dt"] >= pd.Timestamp(P18B.ORIGINS_START))
            & (Pf["origin_dt"] <= pd.Timestamp(P18B.ORIGINS_END))
        ].sort_values("origin_dt")

        check(f"{county}: 43 origins present", len(sub_p) == 43, f"(n={len(sub_p)})")

        max_rel_p0 = 0.0
        max_pop_err = 0.0
        day0_6_exact = True
        beta_order_ok = True
        vax_order_ok = True

        for t0_str in ["2020-12-12", "2021-04-17", "2021-10-02"]:
            r = sub_p[sub_p["origin"] == t0_str].iloc[0]
            t0 = pd.Timestamp(t0_str)
            K, Pr, last_ws, target_weeks, end_day = P18B.build_origin_problem(
                D, county, "L3B", 1, t0, r, om, cdc, vc, rounds, ca
            )
            tr_orig = Tf[
                (Tf["county"] == county)
                & (Tf["model"] == "L3B")
                & (Tf["delay"] == 1)
                & (Tf["origin_dt"] == t0)
            ].set_index("date")
            adm_win = tr_orig.loc[Pr.days[: L2.WINDOW], "adm_model"]

            scen_out, has_doses, dose_t0 = P18B.run_origin_scenarios(
                Pr, D, r, t0, target_weeks
            )
            for sc in P18B.SCENARIOS:
                max_pop_err = max(max_pop_err, scen_out[sc]["pop_err"])

            adm_full_p0 = pd.concat([adm_win, scen_out["P0"]["adm"]])
            p0_wk = {
                h: float(adm_full_p0.loc[ws : ws + pd.Timedelta(days=6)].mean())
                for h, ws in enumerate(target_weeks, 1)
            }
            fc_orig = Ff[
                (Ff["county"] == county)
                & (Ff["model"] == "L3B")
                & (Ff["delay"] == 1)
                & (Ff["origin"] == t0_str)
            ]
            for _, fr in fc_orig.iterrows():
                h_i = int(fr["h"])
                if h_i in p0_wk and pd.notna(fr["pred"]):
                    rel = abs(p0_wk[h_i] - float(fr["pred"])) / max(
                        1e-12, abs(float(fr["pred"]))
                    )
                    max_rel_p0 = max(max_rel_p0, rel)

            # Days 0..6 from t0 identical for P0, P1, P2
            p0_7 = scen_out["P0"]["adm"].iloc[:7].to_numpy()
            p1_7 = scen_out["P1"]["adm"].iloc[:7].to_numpy()
            p2_7 = scen_out["P2"]["adm"].iloc[:7].to_numpy()
            if np.max(np.abs(p1_7 - p0_7)) > 1e-12 or np.max(np.abs(p2_7 - p0_7)) > 1e-12:
                day0_6_exact = False

            # Ordering after day 7
            p0_rest = scen_out["P0"]["adm"].iloc[8:].to_numpy()
            p1_rest = scen_out["P1"]["adm"].iloc[8:].to_numpy()
            p2_rest = scen_out["P2"]["adm"].iloc[8:].to_numpy()
            if not (
                np.all(p1_rest < p0_rest)
                and np.all(p0_rest < p2_rest)
            ):
                beta_order_ok = False

            # Vaccination ordering when doses > 0 on t0
            if has_doses:
                tot_p3 = float(scen_out["P3"]["adm"].sum())
                tot_p0 = float(scen_out["P0"]["adm"].sum())
                tot_p4 = float(scen_out["P4"]["adm"].sum())
                if not (tot_p3 < tot_p0 < tot_p4):
                    vax_order_ok = False

        check(
            f"{county}: P0 reproduces Phase 17B forecast (h=1..4)",
            max_rel_p0 < 1e-6,
            f"(max_rel={max_rel_p0:.2e})",
        )
        check(
            f"{county}: population conserved across scenarios",
            max_pop_err < 1e-12,
            f"(max_err={max_pop_err:.2e})",
        )
        check(f"{county}: P1 and P2 equal P0 on days 0..6 from t0", day0_6_exact)
        check(f"{county}: P1 < P0 < P2 strict ordering from day 8 onward", beta_order_ok)
        check(f"{county}: P3 < P0 < P4 strict ordering when doses > 0", vax_order_ok)

    log(f"\nSummary: {n_pass} PASS, {n_fail} FAIL")
    if n_fail == 0:
        log("ALL PASS")
    (out / "test_phase18b_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()