#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for layer3_escape.py (Phase 17). Run from the project root:
    python test_layer3.py . phase15_admissions
1. Unit: population conserved with all leaks on; cumulative infections grow by the daily new infectious flux;
   ever infected by compartments equals 1 - S - U_v - U_b.
2. Shares: p_pre + p_O1 + p_O2 = 1 and all in [0, 1].
3. Causality: cases and deaths after t0 - 1, doses after t0 - 7, unpublished admission weeks, CDPH variant
   data after t0 - 14 and CDC variant rows created after t0 are replaced by random values; the L3 forecast
   and the carried state must be unchanged (two origins, three counties).
4. Escape values switch on at their public dates.
"""
import sys
import numpy as np
import pandas as pd
import layer2_admissions as L2
import layer3_escape as L3

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok_all &= bool(cond)


n = 150
x0 = np.array([0.4, 0.2, 0.03, 0.003, 0.004, 0.3, 0.05, 0.013, 0.4])
f = {"V": 0.3, "pre_O1": 0.38, "pre_O2": 0.69, "O1_O2": 0.11}
sh = (np.linspace(0.5, 0.0, n), np.linspace(0.4, 0.3, n), np.linspace(0.1, 0.7, n))
X, fE, fI = L3.simulate3(x0, np.full(n, 0.3), np.full(n, 0.002), np.linspace(0.2, 0.7, n), sh, 0.002, 1 / 60, 1 / 191, 0.9, f)
check("population conserved", np.allclose(X[:, :8].sum(axis=1), 1.0, atol=1e-9), f"{np.abs(X[:, :8].sum(axis=1) - 1).max():.1e}")
check("cumulative infections", np.allclose(np.diff(X[:, 8]), fE, atol=1e-12))
check("ever infected identity", np.allclose(X[:, 3:8].sum(axis=1), 1 - X[:, 0] - X[:, 1] - X[:, 2], atol=1e-9))

root = sys.argv[1] if len(sys.argv) > 1 else "."
adm_dir = sys.argv[2] if len(sys.argv) > 2 else "phase15_admissions"
vax = root + "/data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv"
om, cdc = L3.load_variants(root, set())
for t0s in ("2022-01-08", "2022-07-02"):
    t0 = pd.Timestamp(t0s); days = pd.date_range(t0 - pd.Timedelta(days=56), t0 + pd.Timedelta(days=40))
    a, b, c = L3.shares_at(om, cdc, t0, days)
    check(f"shares sum to 1 at {t0s}", np.allclose(a + b + c, 1) and (np.minimum.reduce([a, b, c]) >= 0).all())

rng = np.random.default_rng(11)
for county in ["San Diego", "Los Angeles", "San Francisco"]:
    D = L2.load_county(root, county, adm_dir, vax, set()); vc = L3.load_vax_cum(vax, county, set())
    for t0s in ("2022-01-08", "2022-07-02"):
        t0 = pd.Timestamp(t0s)
        x = L3.initial_state3(D, t0 - pd.Timedelta(days=56), 1 / 4.6)
        r1 = L3.l3_origin(D, t0, 1, True, x, None, 1, 40, (0, 0, 0), om, cdc, vc)
        D2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in D.items()}
        for k in ("cum_cases", "cum_deaths"):
            m = D2[k].index > t0 - L2.DAY; D2[k].loc[m] = rng.uniform(0, 1e7, int(m.sum()))
        m = D2["doses"].index > t0 - pd.Timedelta(days=7); D2["doses"].loc[m] = rng.uniform(0, 0.05, int(m.sum()))
        m = D2["wk"].index + pd.Timedelta(days=6) > t0 - pd.Timedelta(days=7); D2["wk"].loc[m] = rng.uniform(0, 500, int(m.sum()))
        om2 = om.copy(); m = om2.index > t0 - pd.Timedelta(days=L3.CDPH_LAG); om2.loc[m] = rng.uniform(0, 1, int(m.sum()))
        cdc2 = cdc.copy(); cdc2["share"] = cdc2["share"].astype(float); m = cdc2["creation_date"] > t0; cdc2.loc[m, "share"] = rng.uniform(0, 1, int(m.sum()))
        vc2 = vc.astype(float); m = vc2.index > t0 - pd.Timedelta(days=7); vc2.loc[m, "boost"] = rng.uniform(0, 1e7, int(m.sum()))
        r2 = L3.l3_origin(D2, t0, 1, True, x, None, 1, 40, (0, 0, 0), om2, cdc2, vc2)
        same = all(abs(r1["pred"][h][1] - r2["pred"][h][1]) < 1e-9 for h in r1["pred"]) and np.allclose(r1["x_next"], r2["x_next"])
        check(f"causal {county} {t0s}", same)

check("f_V before and after 2021-12-14", L3.fval([(L3.F_V_DATE, 1.0)], pd.Timestamp("2021-12-13")) == 0 and
      L3.fval([(L3.F_V_DATE, 1.0)], pd.Timestamp("2021-12-14")) == 1.0)
check("f_pre_O1 switches on 2022-01-06", L3.fval(L3.F_R_PRE_O1, pd.Timestamp("2022-01-05")) == 0 and
      L3.fval(L3.F_R_PRE_O1, pd.Timestamp("2022-01-06")) == 0.38)
check("f_pre_O2 and f_O1_O2 switch on 2022-07-12", L3.fval(L3.F_R_PRE_O2, pd.Timestamp("2022-07-11")) == 0 and
      L3.fval(L3.F_R_PRE_O2, pd.Timestamp("2022-07-12")) == 0.69 and L3.fval(L3.F_R_O1_O2, pd.Timestamp("2022-07-12")) == 0.11)
print("\nALL PASS" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
