#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Descriptive summary of the official Phase 17B outputs (no fitting, no new model runs). Reads
phase17b_layer3b, phase17_layer3 and phase16_layer2 and prints the tables used in the REVIEW_LOG entry.
Usage (project root):  python describe_phase17b.py > phase17b_layer3b_describe.txt
"""
import hashlib, sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
B, P17, P16 = root / "phase17b_layer3b", root / "phase17_layer3", root / "phase16_layer2"
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
for f in [B / "primary_error_origin.csv", B / "params_states.csv", B / "serology_check.csv",
          P17 / "primary_error_origin.csv", P16 / "primary_error_origin.csv"]:
    print(f.relative_to(root), hashlib.sha256(f.read_bytes()).hexdigest()[:16])

print("\n1. Serology, blood donor validation rows (exact measure, model minus survey)")
s = pd.read_csv(B / "serology_check.csv", parse_dates=["date"])
v = s[s["used_for"] == "validation"].copy()
v["L3_phase17_diff"] = v["L3_phase17_ever_compartments"] - v["sero"]
v["L2_diff"] = v["L2_ever_compartments"] - v["sero"]
print(v.groupby(["county", "source"])[["L3B_diff_compartments", "L3_phase17_diff", "L2_diff"]].agg(["mean", "min", "max"]).round(3))
print(v[v["source"] == "blood_donor_2022_CA"][["county", "date", "sero", "L3B_ever_compartments", "L3B_diff_compartments",
                                               "L3B_within", "L3_phase17_ever_compartments"]].round(3).to_string(index=False))

print("\n2. Primary error, delay 1, by period: L3B (17B), L3 (17), L2 (16), ARIMA_ADM, CASELAG")
e = pd.read_csv(B / "primary_error_origin.csv", parse_dates=["origin"]); e = e[e["delay"] == 1]
e17 = pd.read_csv(P17 / "primary_error_origin.csv", parse_dates=["origin"]); e17 = e17[(e17["delay"] == 1) & (e17["model"] == "L3")]
e16 = pd.read_csv(P16 / "primary_error_origin.csv", parse_dates=["origin"]); e16 = e16[(e16["delay"] == 1) & (e16["model"] == "L2")]
allm = pd.concat([e[e["model"].isin(["L3B", "ARIMA_ADM", "CASELAG"])], e17, e16])
w = allm.pivot_table(index=["county", "origin", "period"], columns="model", values="err").reset_index()
print(w.groupby(["county", "period"])[["L3B", "L3", "L2", "ARIMA_ADM", "CASELAG"]].mean().round(1))
print("\nall origins")
print(w.groupby("county")[["L3B", "L3", "L2", "ARIMA_ADM", "CASELAG"]].mean().round(2))
jan = (w["origin"] >= "2021-12-25") & (w["origin"] <= "2022-01-29")
print("\nexcluding origins 2021-12-25 to 2022-01-29")
print(w[~jan].groupby("county")[["L3B", "L3", "L2", "ARIMA_ADM", "CASELAG"]].mean().round(2))
print("\nshare of the total L3B error from origins 2021-12-25 to 2022-01-29")
print((w[jan].groupby("county")["L3B"].sum() / w.groupby("county")["L3B"].sum()).round(3))
print("\nL3B better than L3 / than L2 (count of origins)")
print(w.groupby("county").apply(lambda g: pd.Series({"n": len(g), "L3B<L3": int((g["L3B"] < g["L3"]).sum()),
                                                      "L3B<L2": int((g["L3B"] < g["L2"]).sum())}), include_groups=False))
print("\nten largest L3B errors")
print(w.sort_values("L3B", ascending=False).head(10)[["county", "origin", "L3B", "L3", "L2", "ARIMA_ADM"]].round(1).to_string(index=False))

print("\n3. States and rates, L3B delay 1 (median by period), with Phase 17 L3 for comparison")
cols = ["rho", "rho_sero", "ihr", "beta_last", "Rt_t0", "S_eff_t0", "R_pre_t0", "R_O1_t0", "R_O2_t0"]
p = pd.read_csv(B / "params_states.csv"); p = p[(p["model"] == "L3B") & (p["delay"] == 1)]
p["ever_t0"] = p[["E_t0", "I_t0", "R_pre_t0", "R_O1_t0", "R_O2_t0"]].sum(axis=1)
print(p.groupby(["county", "period"])[cols + ["ever_t0"]].median().round(3))
p17 = pd.read_csv(P17 / "params_states.csv"); p17 = p17[(p17["model"] == "L3") & (p17["delay"] == 1)]
p17["ever_t0"] = p17[["E_t0", "I_t0", "R_pre_t0", "R_O1_t0", "R_O2_t0"]].sum(axis=1)
print("\nPhase 17 L3")
print(p17.groupby(["county", "period"])[["rho", "ihr", "beta_last", "Rt_t0", "S_eff_t0", "ever_t0"]].median().round(3))
print("\nrho relative to rho_sero (L3B): median and range of rho / rho_sero by county")
print((p["rho"] / p["rho_sero"]).groupby(p["county"]).describe()[["50%", "min", "max"]].round(2))
print("\nfits at max_nfev:", int((p["nfev"] >= 300).sum()))
