#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare the independent replication (phase18b_replication) with the 2026-10-15 Phase 18B run
(phase18b_projections_v2, and the earlier phase18b_projections) on the Phase 17B week grid, chain L3B delay 1.
No simulation. Usage (project root):  python compare_phase18b_replication.py > phase18b_replication_compare.txt
"""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
R = pd.read_csv(root / "phase18b_replication/projections_weekly.csv")
R = R[R["grid"] == "p17b"].rename(columns={"k": "h"})
R["scenario"] = R["scenario"].replace({"H0": "P0_HINDSIGHT"})
for name in ("phase18b_projections_v2", "phase18b_projections"):
    f = root / name / "projections_weekly.csv"
    print(f"\n== {name} (sha256 {hashlib.sha256(f.read_bytes()).hexdigest()[:16]})")
    G = pd.read_csv(f)
    try:
        G = G[(G["model"] == "L3B") & (G["delay"] == 1)]
        _ = G[["county", "origin", "scenario", "h", "week_start", "pred", "obs", "naive_pred"]]
    except KeyError as e:
        print("columns differ, skipped:", e, list(G.columns)); continue
    j = G.merge(R, on=["county", "origin", "scenario", "h"], suffixes=("_g", "_r"), how="outer", indicator=True)
    print("rows only in one file:", j["_merge"].value_counts().to_dict())
    j = j[j["_merge"] == "both"]
    print("week_start equal:", bool((j["week_start_g"] == j["week_start_r"]).all()))
    rel = (j["pred_g"] / j["pred_r"] - 1).abs()
    print("max relative difference of pred by scenario:"); print(rel.groupby(j["scenario"]).max().to_string())
    ob = j[["obs_g", "obs_r"]]
    print("obs equal where both present:", bool(np.allclose(ob.dropna()["obs_g"], ob.dropna()["obs_r"])),
          "| obs missing in run / replication:", int(j["obs_g"].isna().sum()), int(j["obs_r"].isna().sum()))
    nv = (j["naive_pred"] - j["naive"]).abs().max(); print("naive max abs difference:", float(nv))
    lo_hi = j[j["scenario"].isin(["P0", "P1", "P2", "P3", "P4"])].groupby(["county", "origin", "h"]).agg(
        lo_g=("pred_g", "min"), hi_g=("pred_g", "max"), lo_r=("pred_r", "min"), hi_r=("pred_r", "max"), obs_g=("obs_g", "first"), obs_r=("obs_r", "first"))
    lo_hi["in_g"] = (lo_hi["obs_g"] >= lo_hi["lo_g"]) & (lo_hi["obs_g"] <= lo_hi["hi_g"])
    lo_hi["in_r"] = (lo_hi["obs_r"] >= lo_hi["lo_r"]) & (lo_hi["obs_r"] <= lo_hi["hi_r"])
    d = lo_hi[(lo_hi["in_g"] != lo_hi["in_r"]) | (lo_hi["obs_g"].isna() != lo_hi["obs_r"].isna())]
    print("envelope pairs classified differently or with obs present in only one file:", len(d))
    if len(d):
        print(d.reset_index().to_string(index=False))
