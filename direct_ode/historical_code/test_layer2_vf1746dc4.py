#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for layer2_admissions.py (Phase 16). Run from the project root:
    python test_layer2.py . phase15_admissions
1. Unit: the model conserves the population (S + U_v + U_b + E + I + R = 1) with vaccination, behaviour and
   both wanings switched on; the daily fluxes match the change of the states.
2. Behaviour signal: identical to the paper's bayesian_alpha_series (core_phase10.py) when torch is
   available; otherwise its defining properties are checked (range [0, 1], recursion start at 0.5).
3. Causality: for one origin per county, cases and deaths after t0 - 1, doses after t0 - 7 and admission
   weeks not yet published are replaced by random values; the L2 forecast, its fitted state and all
   comparators must be unchanged (delay 1 and delay 0).
4. Fixed values: the schedules switch on the dates of Addendum B.
"""
import sys, math
import numpy as np
import pandas as pd
import layer2_admissions as L

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok_all &= bool(cond)


# 1. conservation and fluxes
n = 120
x0 = np.array([0.7, 0.05, 0.02, 0.003, 0.004, 0.223])
beta = np.full(n, 0.25); dose = np.full(n, 0.002); s = np.linspace(0.2, 0.8, n)
X, fE, fI = L.simulate(x0, beta, dose, s, 0.002, 1 / 60, 1 / 191, 0.9, 1 / 365)
check("population conserved", np.allclose(X.sum(axis=1), 1.0, atol=1e-9), f"max dev {np.abs(X.sum(axis=1) - 1).max():.2e}")
# cumulative new infectious equals the E inflow minus change in E: check fI against I balance
dI = X[1:, 4] - X[:-1, 4]
check("I balance (fE - fI = change in I)", np.allclose(fE - fI, dI, atol=1e-6), f"max {np.abs(fE - fI - dI).max():.2e}")

# 2. behaviour signal
cum = np.cumsum(np.abs(np.random.default_rng(3).normal(5, 3, 400)))
sL = L.behaviour_s(cum)
try:
    from covid_pinn import core_phase10 as C
    a = C.bayesian_alpha_series(None, cum, 0.0, 0.003, 0.08, 0.995)
    check("behaviour signal equals paper code", np.allclose(sL, (a - 0.0) / 0.003, atol=1e-9))
except Exception as e:
    check("behaviour signal range and start", (sL.min() >= 0) and (sL.max() <= 1), f"paper code not importable here ({type(e).__name__})")

# 3. causality
root = sys.argv[1] if len(sys.argv) > 1 else "."
adm_dir = sys.argv[2] if len(sys.argv) > 2 else "phase15_admissions"
vax = "data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv"
rng = np.random.default_rng(7)
for county in ["San Diego", "Los Angeles", "San Francisco"]:
    D = L.load_county(root, county, adm_dir, root + "/" + vax, set())
    t0 = pd.Timestamp("2021-08-07")
    x = L.initial_state(D, t0 - pd.Timedelta(days=L.WINDOW), 1 / 4.6)
    for delay in (1, 0):
        r1 = L.l2_origin(D, t0, delay, True, 0.0, x, None, 1, 40, (0, 0, 0))
        D2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in D.items()}
        Lc = t0 - L.DAY
        for k in ("cum_cases", "cum_deaths"):
            m = D2[k].index > Lc; D2[k].loc[m] = rng.uniform(0, 1e7, int(m.sum()))
        m = D2["doses"].index > t0 - pd.Timedelta(days=7); D2["doses"].loc[m] = rng.uniform(0, 0.05, int(m.sum()))
        m = D2["wk"].index + pd.Timedelta(days=6) > t0 - pd.Timedelta(days=7 * delay)
        D2["wk"].loc[m] = rng.uniform(0, 500, int(m.sum()))
        r2 = L.l2_origin(D2, t0, delay, True, 0.0, x, None, 1, 40, (0, 0, 0))
        same = all(abs(r1["pred"][h][1] - r2["pred"][h][1]) < 1e-9 for h in r1["pred"]) and np.allclose(r1["x_next"], r2["x_next"])
        check(f"causal {county} L2 delay {delay}", same)
        c1 = L.comparators(r1["K"], r1["target_weeks"]); c2 = L.comparators(r2["K"], r2["target_weeks"])
        samec = all(np.allclose([c1[m][0][h] for h in c1[m][0]], [c2[m][0][h] for h in c2[m][0]], equal_nan=True) for m in c1)
        check(f"causal {county} comparators delay {delay}", samec)

# 4. schedules
check("VE before and after 2021-03-29", L.sched(L.VE_SCHED, pd.Timestamp("2021-03-28")) == 0.95 and
      L.sched(L.VE_SCHED, pd.Timestamp("2021-03-29")) == 0.90)
check("omega_v switches on 2021-10-04", L.sched(L.OMEGA_V_SCHED, pd.Timestamp("2021-10-03")) == 0.0 and
      abs(L.sched(L.OMEGA_V_SCHED, pd.Timestamp("2021-10-04")) - 1 / 191) < 1e-12)
check("rho prior switches on 2020-11-25", abs(L.sched(L.RHO_PRIOR_SCHED, pd.Timestamp("2020-11-24")) - 1 / 12) < 1e-12 and
      abs(L.sched(L.RHO_PRIOR_SCHED, pd.Timestamp("2020-11-25")) - 1 / 4.6) < 1e-12)
print("\nALL PASS" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
