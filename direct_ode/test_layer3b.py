#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for layer3b_sero.py (Phase 17B). Run from the project root:
    python test_layer3b.py . phase15_admissions
1. rho_sero: equals California cumulative cases per person at the round midpoint / seroprevalence of the
   latest round known 42 days after the end of collection; None before the first round is known.
2. The rho prior term uses log sd 0.3 and is centred on rho_sero.
3. Causality: everything Phase 17 tested, plus serosurvey rounds not yet known at t0 and California cases
   after t0 - 1 replaced by random values; forecast and carried state unchanged (two origins, three counties).
"""
import sys, math
import numpy as np
import pandas as pd
import layer2_admissions as L2
import layer3b_sero as B

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok_all &= bool(cond)


root = sys.argv[1] if len(sys.argv) > 1 else "."
adm_dir = sys.argv[2] if len(sys.argv) > 2 else "phase15_admissions"
vax = root + "/data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv"
rounds = B.load_commercial_rounds(root, set()); ca = B.load_ca_cases(root, set())
check("30 all ages rounds", len(rounds) == 30, str(len(rounds)))
t = pd.Timestamp("2021-03-06"); v, k = B.rho_sero_at(t, rounds, ca)
r = rounds[rounds["round"] == k].iloc[0]
check("latest known round", r["end"] + pd.Timedelta(days=42) <= t and
      (rounds[rounds["round"] == k + 1].empty or rounds[rounds["round"] == k + 1].iloc[0]["end"] + pd.Timedelta(days=42) > t), f"round {k}")
check("rho_sero formula", abs(v - float(ca[ca.index <= r["mid"]].iloc[-1]) / r["sero"]) < 1e-12, f"{v:.4f}")
check("no round known before 2020-09-16", B.rho_sero_at(pd.Timestamp("2020-09-15"), rounds, ca)[0] is None)

om, cdc = B.load_variants(root, set())
rng = np.random.default_rng(13)
for county in ["San Diego", "Los Angeles", "San Francisco"]:
    D = L2.load_county(root, county, adm_dir, vax, set()); vc = B.load_vax_cum(vax, county, set())
    for t0s in ("2021-01-09", "2022-01-08"):
        t0 = pd.Timestamp(t0s)
        x = B.initial_state3(D, t0 - pd.Timedelta(days=56), 0.4)
        rs = B.rho_sero_at(t0, rounds, ca)[0]
        r1 = B.l3_origin(D, t0, 1, True, x, None, 1, 40, (0, 0, 0), om, cdc, vc, rs)
        if county == "San Diego" and t0s == "2021-01-09":
            K = r1["K"]; last_ws = K["wk"].dropna().index.max(); end_day = last_ws + pd.Timedelta(days=34)
            P = B.Problem3B(K, D, t0, x, True, (end_day - (t0 - L2.DAY)).days, om, cdc, B.booster_weight(vc, t0), rho_sero=rs)
            th = r1["theta"].copy(); res = P.residuals(th)
            check("rho prior term with sd 0.3", abs(res[-5] - (th[9] - math.log(rs)) / 0.3) < 1e-12)
        D2 = {k_: (v_.copy() if hasattr(v_, "copy") else v_) for k_, v_ in D.items()}
        for k_ in ("cum_cases", "cum_deaths"):
            m = D2[k_].index > t0 - L2.DAY; D2[k_].loc[m] = rng.uniform(0, 1e7, int(m.sum()))
        m = D2["doses"].index > t0 - pd.Timedelta(days=7); D2["doses"].loc[m] = rng.uniform(0, 0.05, int(m.sum()))
        m = D2["wk"].index + pd.Timedelta(days=6) > t0 - pd.Timedelta(days=7); D2["wk"].loc[m] = rng.uniform(0, 500, int(m.sum()))
        om2 = om.copy(); m = om2.index > t0 - pd.Timedelta(days=14); om2.loc[m] = rng.uniform(0, 1, int(m.sum()))
        cdc2 = cdc.copy(); cdc2["share"] = cdc2["share"].astype(float); m = cdc2["creation_date"] > t0
        cdc2.loc[m, "share"] = rng.uniform(0, 1, int(m.sum()))
        vc2 = vc.astype(float); m = vc2.index > t0 - pd.Timedelta(days=7); vc2.loc[m, "boost"] = rng.uniform(0, 1e7, int(m.sum()))
        rounds2 = rounds.copy(); m = rounds2["known_from"] > t0; rounds2.loc[m, "sero"] = rng.uniform(0.01, 0.9, int(m.sum()))
        ca2 = ca.copy(); m = ca2.index > t0 - L2.DAY; ca2.loc[m] = rng.uniform(0, 1, int(m.sum()))
        rs2 = B.rho_sero_at(t0, rounds2, ca2)[0]
        r2 = B.l3_origin(D2, t0, 1, True, x, None, 1, 40, (0, 0, 0), om2, cdc2, vc2, rs2)
        same = abs(rs - rs2) < 1e-15 and all(abs(r1["pred"][h][1] - r2["pred"][h][1]) < 1e-9 for h in r1["pred"]) \
            and np.allclose(r1["x_next"], r2["x_next"])
        check(f"causal {county} {t0s}", same)
print("\nALL PASS" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
