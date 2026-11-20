#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Descriptive summary of the official Phase 18 outputs (no fitting, no new simulation). Reads phase18_scenarios,
phase17b_layer3b/params_states.csv and the corrected admissions; prints the tables used in the REVIEW_LOG entry.
Usage (project root):  python describe_phase18.py > phase18_scenarios_describe.txt
"""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
O = root / "phase18_scenarios"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 300)
for f in [O / "reconstruction.csv", O / "scenario_summary.csv", O / "scenario_daily.csv", root / "phase17b_layer3b/params_states.csv"]:
    print(f.relative_to(root), hashlib.sha256(f.read_bytes()).hexdigest()[:16])

R = pd.read_csv(O / "reconstruction.csv")
print("\n1. Reconstruction (R1, R2) by county, period and chain")
print(R[["county", "period", "model", "delay", "ratio", "obs_peak_week", "S0_peak_week", "sero_within", "n_sero", "R1", "R2"]].round(3).to_string(index=False))

S = pd.read_csv(O / "scenario_summary.csv")
print("\n2. Scenario effects, percent change from S0 (primary chain L3B delay 1; range over chains in brackets)")
for (c, p), g in S[S["scenario"] != "S0"].groupby(["county", "period"]):
    for sc, h in g.groupby("scenario"):
        pr = h[(h["model"] == "L3B") & (h["delay"] == 1)].iloc[0]
        print(f"{c:14s} {p:12s} {sc}: admissions {pr['pct_admissions']:+.1f} % [{h['pct_admissions'].min():+.1f}, {h['pct_admissions'].max():+.1f}]"
              f" ({pr['diff_admissions']:+.0f}); infections {pr['pct_infections']:+.1f} %; peak weekly admissions {pr['pct_peak_weekly_adm_per_day']:+.1f} %;"
              f" ever infected at end {pr['ever_infected_end']:.2f} (S0 {pr['ever_infected_end'] - pr['diff_ever_infected_end']:.2f})")

print("\n3. Where the S0 drift arises: S0 over observed admissions by calendar month (primary chain, behaviour period)")
D = pd.read_csv(O / "scenario_daily.csv", parse_dates=["date"])
D = D[(D["model"] == "L3B") & (D["delay"] == 1) & (D["scenario"] == "S0")]
city = {"San Diego": "SanDiego", "Los Angeles": "LosAngeles", "San Francisco": "SanFrancisco"}
for c in city:
    wk = pd.read_csv(root / f"phase15_admissions/weekly_{city[c]}.csv", parse_dates=["week_start"]).set_index("week_start")["adm_per_day"]
    d = D[(D["county"] == c) & (D["period"] == "behaviour")].set_index("date")["adm"]
    rows = []
    for ws, v in wk.dropna().items():
        if ws >= d.index.min() and ws + pd.Timedelta(days=6) <= d.index.max():
            rows.append({"month": ws.strftime("%Y-%m"), "obs": v * 7, "S0": float(d.loc[ws:ws + pd.Timedelta(days=6)].sum())})
    m = pd.DataFrame(rows).groupby("month")[["obs", "S0"]].sum(); m["ratio"] = m["S0"] / m["obs"]
    print(c); print(m.round(2).to_string())

print("\n4. Weekly E and I corrections of the fits whose first week falls in each month (geometric mean, L3B delay 1)")
P = pd.read_csv(root / "phase17b_layer3b/params_states.csv", parse_dates=["origin"])
P = P[(P["model"] == "L3B") & (P["delay"] == 1)].copy()
P["first_week_month"] = (P["origin"] - pd.Timedelta(days=56)).dt.strftime("%Y-%m")
P = P[(P["first_week_month"] >= "2020-10") & (P["first_week_month"] <= "2021-11")]
g = P.groupby(["county", "first_week_month"])[["mult_E", "mult_I"]].agg(lambda x: float(np.exp(np.log(x).mean())))
print(g.round(3).unstack(0).to_string())
