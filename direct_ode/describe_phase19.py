#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Descriptive summary of the official Phase 19 outputs against Phase 17B (no fitting, no new simulation).
Usage (project root):  python describe_phase19.py > phase19_describe.txt
"""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
A, B = root / "phase19_layer3t", root / "phase17b_layer3b"
for f in [A / "params_states.csv", A / "primary_error_origin.csv", A / "serology_check.csv", root / "phase19_scenarios/reconstruction.csv"]:
    print(f.relative_to(root), hashlib.sha256(f.read_bytes()).hexdigest()[:16])

print("\n1. Reconstruction R1 and R2 (primary chains): Phase 19 L3T against Phase 18 on L3B")
r19 = pd.read_csv(root / "phase19_scenarios/reconstruction.csv"); r18 = pd.read_csv(root / "phase18_scenarios/reconstruction.csv")
k = ["county", "period", "ratio", "peak_shift_days", "sero_within", "n_sero", "R1", "R2"]
j = r19[(r19["model"] == "L3T") & (r19["delay"] == 1)][k].merge(r18[(r18["model"] == "L3B") & (r18["delay"] == 1)][k], on=["county", "period"], suffixes=("_L3T", "_L3B"))
print(j.round(3).to_string(index=False))

print("\n2. Testing factor g used by the fits (L3T delay 1)")
p = pd.read_csv(A / "params_states.csv", parse_dates=["origin"]); p = p[(p["model"] == "L3T") & (p["delay"] == 1)]
p["g_off"] = (p["g_window_start"] == 1.0) & (p["g_t0_minus_1"] == 1.0)
print("origins with g = 1 throughout (before 2020-10-12 or no test data):", p.groupby("county")["g_off"].sum().to_dict())
p["month"] = p["origin"].dt.strftime("%Y-%m")
w = p[(p["origin"] >= "2020-11-28") & (p["origin"] <= "2021-05-22")]
print(w.groupby(["county", "month"])[["g_window_start", "g_t0_minus_1", "rho"]].median().round(3).to_string())

print("\n3. Weekly E and I corrections of fits whose first week is in each month, L3T against L3B (geometric mean)")
q = pd.read_csv(B / "params_states.csv", parse_dates=["origin"]); q = q[(q["model"] == "L3B") & (q["delay"] == 1)]
for name, d in (("L3T", p), ("L3B", q)):
    d = d.copy(); d["fw"] = (d["origin"] - pd.Timedelta(days=56)).dt.strftime("%Y-%m")
    d = d[(d["fw"] >= "2020-10") & (d["fw"] <= "2021-03")]
    print(name); print(d.groupby(["county", "fw"])[["mult_E", "mult_I"]].agg(lambda x: float(np.exp(np.log(x).mean()))).round(3).unstack(0).to_string())

print("\n4. Primary forecast error, delay 1, by period: L3T, L3B, ARIMA_ADM")
e19 = pd.read_csv(A / "primary_error_origin.csv"); e17 = pd.read_csv(B / "primary_error_origin.csv")
e = pd.concat([e19[(e19["delay"] == 1) & e19["model"].isin(["L3T", "ARIMA_ADM"])], e17[(e17["delay"] == 1) & (e17["model"] == "L3B")]])
print(e.pivot_table(index=["county", "period"], columns="model", values="err", aggfunc="mean").round(1).to_string())
print(e.pivot_table(index="county", columns="model", values="err", aggfunc="mean").round(2).to_string())

print("\n5. Serology, blood donor validation dates: mean model minus survey, L3T and L3B")
s = pd.read_csv(A / "serology_check.csv"); s = s[s["used_for"] == "validation"].copy()
s["L3B_diff"] = s["L3B_phase17b_ever_compartments"] - s["sero"]
print(s.groupby(["county", "source"])[["L3T_diff_compartments", "L3B_diff"]].mean().round(3).to_string())
