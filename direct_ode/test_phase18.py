#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for scenarios_phase18.py (Phase 18). Run from the project root after phase17b_layer3b exists:
    python test_phase18.py .
Prints only PASS or FAIL with check sizes, never scenario outcomes.
1. Stitched reconstruction: one week simulated from the fitted state (with the E and I corrections) reproduces
   the Phase 17B fitted states and admissions (relative 1e-4; the beta knots are stored to 5 decimals), and
   the rebuilt inputs equal those recorded in params_states.csv (three weeks, three counties).
2. Population conserved in every scenario path (all counties, primary chain, both periods).
3. V0 gives at least as many infections and admissions as S0; V1 at most as many infections as S0 in
   Los Angeles and San Diego.
4. B0 equals S0 when a_b is already 0 and U_b is already empty.
"""
import sys
import numpy as np
import pandas as pd
import layer2_admissions as L2
import layer3b_sero as B
import scenarios_phase18 as S

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok_all &= bool(cond)


root = sys.argv[1] if len(sys.argv) > 1 else "."
vax = root + "/data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv"
inputs = set()
Pf, Tf = S.load_fits(root, inputs)
om, cdc = B.load_variants(root, inputs)
rounds = B.load_commercial_rounds(root, inputs); ca = B.load_ca_cases(root, inputs)
Dfast = L2.load_county(root, S.FASTEST, "phase15_admissions", vax, inputs)
for county in ["San Diego", "Los Angeles", "San Francisco"]:
    D = L2.load_county(root, county, "phase15_admissions", vax, inputs); vc = B.load_vax_cum(vax, county, inputs); N = D["N"]
    for w0 in ("2020-12-12", "2021-06-05", "2021-11-27"):
        w = S.build_weeks(D, county, "L3B", 1, Pf, Tf, pd.Timestamp(w0), pd.Timestamp(w0), om, cdc, vc, rounds, ca)[0]
        X, fE, fI = B.simulate3(w["x_fit"][0], w["beta"], w["dose"], w["s"], w["sh"], w["a_b"], w["omega_b"], w["omv"], w["ve"], w["f"])
        rs = np.abs(X - w["x_fit"]).max() / np.abs(w["x_fit"]).max()
        ra = np.abs(fI * N * w["ihr"] - w["adm_fit"]).max() / np.abs(w["adm_fit"]).max()
        check(f"stitch {county} {w0}", rs < 1e-4 and ra < 1e-4 and all(abs(u - v) < 1e-12 for u, v in w["check"].values()),
              f"rel state {rs:.1e}, rel adm {ra:.1e}")
    for period in ("vaccination", "behaviour"):
        a, b = (pd.Timestamp(x) for x in S.PERIODS[period])
        weeks = S.build_weeks(D, county, "L3B", 1, Pf, Tf, a, b, om, cdc, vc, rounds, ca)
        paths = S.scenario_paths(weeks, N, period, doses_fast=Dfast["doses"] if county != S.FASTEST else None, doses_own=D["doses"])
        err = max(float((p["pop_sum"] - 1).abs().max()) for p in paths.values())
        check(f"population conserved {county} {period}", err < 1e-9, f"{err:.1e}")
        if period == "vaccination":
            s0, v0 = paths["S0"], paths["V0"]
            check(f"V0 >= S0 {county}", v0["new_infectious"].sum() >= s0["new_infectious"].sum() and v0["adm"].sum() >= s0["adm"].sum())
            if county != S.FASTEST:
                check(f"V1 <= S0 infections {county}", paths["V1"]["new_infectious"].sum() <= s0["new_infectious"].sum())
            else:
                check("no V1 for the fastest county", "V1" not in paths)
        else:
            wz = [dict(w, a_b=0.0) for w in weeks]
            x0 = wz[0]["x_fit"][0].copy(); x0[0] += x0[2]; x0[2] = 0.0
            wz[0] = dict(wz[0], x_fit=np.vstack([x0, wz[0]["x_fit"][1:]]))
            pz = S.scenario_paths(wz, N, period)
            check(f"B0 equals S0 without behaviour {county}", np.allclose(pz["B0"][S.STATE].values, pz["S0"][S.STATE].values, rtol=0, atol=1e-15))
print("\nALL PASS" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
