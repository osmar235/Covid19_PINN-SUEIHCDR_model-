#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for layer1c_recalibration.py (Phase 15). Run from the project root:
    python test_layer1c.py . phase15_admissions
1. Unit: one simulated admission reproduces the analytic mean stay, in hospital death probability and ICU
   days (as the Phase 14 unit test).
2. Causality: for one origin per county, all census after t0 - 1 and all admission weeks not yet published
   are replaced by garbage; every non oracle forecast and the comparators must be unchanged.
3. Admission rule: PERSIST and TREND values after the last known week follow the declared formulas.
"""
import math, sys
import numpy as np
import pandas as pd
import layer1c_recalibration as L

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok_all &= bool(cond)


# 1. unit test of the exact daily map
p = {"L_w": 6.0, "p_wc": 0.12, "p_wd": 0.05, "L_c": 9.0, "p_cd": 0.35, "L_p": 5.0, "q": 0.05}
Phi, Gam = L.transition(p)
z = Gam * 1.0; census_days = 0.0; icu_days = 0.0
# integrate one unit admission over day 0 exactly with fine steps for the census-day total
n_fine = 20000; dtf = 1.0 / 20
from scipy.linalg import expm
M = np.zeros((5, 5)); kw, kc, kp = 1 / p["L_w"], 1 / p["L_c"], 1 / p["L_p"]
M[0, 0] = -kw; M[1, 0] = p["p_wc"] * kw; M[1, 1] = -kc; M[2, 1] = (1 - p["p_cd"]) * kc; M[2, 2] = -kp
M[3, 0] = p["p_wd"] * kw; M[3, 1] = p["p_cd"] * kc
Ef = expm(M[:4, :4] * dtf)
x = np.array([1 - p["q"], p["q"], 0.0, 0.0])
for _ in range(n_fine):                      # trapezoid rule
    xn = Ef @ x
    census_days += 0.5 * ((x[0] + x[1] + x[2]) + (xn[0] + xn[1] + xn[2])) * dtf
    icu_days += 0.5 * (x[1] + xn[1]) * dtf; x = xn
dq = L.derived(p)
check("mean stay", abs(census_days - dq["mean_hosp_stay"]) < 1e-3, f"{census_days:.4f} vs {dq['mean_hosp_stay']:.4f}")
check("death probability", abs(x[3] - dq["p_death_hosp"]) < 1e-4, f"{x[3]:.5f} vs {dq['p_death_hosp']:.5f}")
check("ICU days", abs(icu_days - dq["p_icu"] * p["L_c"]) < 1e-3, f"{icu_days:.4f} vs {dq['p_icu'] * p['L_c']:.4f}")
# steady state share
ss = np.linalg.solve(np.eye(3) - Phi[:3, :3], Gam[:3])
check("steady ward share", abs(ss[0] / (ss[0] + ss[2]) - L.steady_fw(p)) < 1e-6)

# 2. causality
root = sys.argv[1] if len(sys.argv) > 1 else "."
audit = sys.argv[2] if len(sys.argv) > 2 else "phase15_admissions"
inputs = set()
rng = np.random.default_rng(1)
for county in ["San Diego", "Los Angeles", "San Francisco"]:
    census, wk, oracle = L.load_county(root, audit, "covid19hospitalbycounty.csv", county, inputs)
    t0 = pd.Timestamp("2021-07-31")
    d1, p1 = L.run_origin(census, wk, oracle, t0, [56], 1, 30, 0, True)
    c2 = census.copy(); c2.loc[c2.index > t0 - L.DAY] = rng.uniform(0, 5000, size=(int((c2.index > t0 - L.DAY).sum()), 2))
    w2 = wk.copy(); late = w2.index + pd.Timedelta(days=6) > t0 - pd.Timedelta(days=7)
    w2.loc[late] = rng.uniform(0, 500, size=int(late.sum()))
    d2, p2 = L.run_origin(c2, w2, oracle, t0, [56], 1, 30, 0, True)
    A, B = pd.DataFrame(d1), pd.DataFrame(d2)
    for m in ["H4_PERSIST", "H4_TREND", "ARIMA", "NAIVE"]:
        sa = A[(A["model"] == m) & (A["delay"].isin([1, -1]))][["ward_pred", "icu_pred"]].values
        sb = B[(B["model"] == m) & (B["delay"].isin([1, -1]))][["ward_pred", "icu_pred"]].values
        check(f"causal {county} {m} (delay 1)", sa.shape == sb.shape and np.allclose(sa, sb, equal_nan=True))
    # delay 0 may use the week ending on t0 but nothing later and no census after t0 - 1
    w3 = wk.copy(); late0 = w3.index + pd.Timedelta(days=6) > t0
    w3.loc[late0] = rng.uniform(0, 500, size=int(late0.sum()))
    c3 = c2
    d3, _ = L.run_origin(c3, w3, oracle, t0, [56], 1, 30, 0, False)
    C = pd.DataFrame(d3)
    for m in ["H4_PERSIST", "H4_TREND"]:
        sa = A[(A["model"] == m) & (A["delay"] == 0)][["ward_pred", "icu_pred"]].values
        sc = C[(C["model"] == m) & (C["delay"] == 0)][["ward_pred", "icu_pred"]].values
        check(f"causal {county} {m} (delay 0)", sa.shape == sc.shape and np.allclose(sa, sc, equal_nan=True))

# 3. admission rule
census, wk, oracle = L.load_county(root, audit, "covid19hospitalbycounty.csv", "San Diego", inputs)
t0 = pd.Timestamp("2021-07-31")
ap, ip = L.causal_admissions(wk, t0, 1, "PERSIST", t0 - pd.Timedelta(days=20), t0 + pd.Timedelta(days=10))
at, it = L.causal_admissions(wk, t0, 1, "TREND", t0 - pd.Timedelta(days=20), t0 + pd.Timedelta(days=10))
E = pd.Timestamp(ip["last_known_week_end"])
check("last known week ends t0 - 7", E == t0 - pd.Timedelta(days=7), ip["last_known_week_end"])
days = pd.date_range(t0 - pd.Timedelta(days=20), t0 + pd.Timedelta(days=10))
after = days > E
check("persist after last week", np.allclose(ap[after], ip["A_last"]))
r = it["trend_r"]
check("trend formula", np.allclose(at[after], it["A_last"] * np.exp(r * np.array([(d - E).days for d in days[after]]))),
      f"r = {r:.4f}, cap = {L.TREND_CAP:.4f}")
check("before last week equal", np.allclose(ap[~after], at[~after]))
print("\nALL PASS" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
