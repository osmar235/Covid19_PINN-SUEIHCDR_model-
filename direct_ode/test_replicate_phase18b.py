#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for replicate_phase18b.py (independent replication of Phase 18B). Run from the project root:
    python test_replicate_phase18b.py .
Prints only PASS or FAIL with check sizes, never projection outcomes against data.
1. P0 on the Phase 17B grid reproduces the Phase 17B forecast for weeks 1 to 4 (relative 1e-9), and on the
   after_t0 grid weeks 1 to 3 equal Phase 17B weeks 2 to 4; the rebuilt inputs match.
2. Scenario changes start on the stated day: P1 and P2 equal P0 on days 0 to 6 and differ from day 8;
   P3 and P4 differ from P0 from day 1 when the held vaccination pace is above 0.
3. Population conserved in every path.
4. Total admissions P1 <= P0 <= P2 and P3 <= P0 <= P4.
"""
import sys
import numpy as np
import pandas as pd
import layer2_admissions as L2
import layer3b_sero as B
import replicate_phase18b as PJ

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok_all &= bool(cond)


root = sys.argv[1] if len(sys.argv) > 1 else "."
vax = root + "/data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv"
inputs = set()
Pf, Tf, Ff = PJ.load_fits(root, inputs)
om, cdc = B.load_variants(root, inputs)
rounds = B.load_commercial_rounds(root, inputs); ca = B.load_ca_cases(root, inputs)
for county in ["San Diego", "Los Angeles", "San Francisco"]:
    D = L2.load_county(root, county, "phase15_admissions", vax, inputs); vc = B.load_vax_cum(vax, county, inputs)
    for t0s in ("2020-12-12", "2021-03-06", "2021-08-07", "2021-10-02"):
        t0 = pd.Timestamp(t0s)
        o = PJ.origin_inputs(D, county, t0, Pf, Tf, om, cdc, vc, rounds, ca)
        paths = PJ.scenarios(o, actual_doses=D["doses"])
        f17 = Ff[(Ff["county"] == county) & (Ff["origin"] == t0)].set_index("week_start")["pred"]
        wb = PJ.weekly(paths["P0"], t0, "p17b", o["adm_before_t0"]).set_index("week_start")["pred"]
        wa = PJ.weekly(paths["P0"], t0, "after_t0").set_index("week_start")["pred"]
        jb = pd.concat([wb.rename("p0"), f17.rename("f17")], axis=1, sort=True).dropna()
        ja = pd.concat([wa.rename("p0"), f17.rename("f17")], axis=1, sort=True).dropna()
        rb = float((jb["p0"] / jb["f17"] - 1).abs().max()); ra = float((ja["p0"] / ja["f17"] - 1).abs().max())
        check(f"P0 = Phase 17B forecast {county} {t0s}", len(jb) == 4 and len(ja) == 3 and rb < 1e-9 and ra < 1e-9
              and all(abs(u - v) < 1e-12 for u, v in o["check"].values()), f"rel {max(rb, ra):.1e}")
        a0 = paths["P0"]["adm"].values
        same = np.allclose(paths["P1"]["adm"].values[:7], a0[:7], rtol=0, atol=0) and np.allclose(paths["P2"]["adm"].values[:7], a0[:7], rtol=0, atol=0)
        diff = (paths["P1"]["adm"].values[8:] < a0[8:]).all() and (paths["P2"]["adm"].values[8:] > a0[8:]).all()
        check(f"transmission change starts on day 7 {county} {t0s}", same and diff)
        err = max(float((p["pop_sum"] - 1).abs().max()) for p in paths.values())
        check(f"population conserved {county} {t0s}", err < 1e-9, f"{err:.1e}")
        t = {k: float(p["adm"].sum()) for k, p in paths.items()}
        mono = t["P1"] <= t["P0"] <= t["P2"]
        if o["dose"].max() > 0:
            mono &= t["P3"] <= t["P0"] <= t["P4"]
            mono &= bool((paths["P3"]["S"].values[1:] < paths["P0"]["S"].values[1:]).all())
        else:
            mono &= "P3" not in paths
        check(f"ordering of scenarios {county} {t0s}", mono)
print("\nALL PASS" if ok_all else "\nSOME TESTS FAILED")
sys.exit(0 if ok_all else 1)
