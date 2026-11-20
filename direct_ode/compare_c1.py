#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correction C1 (2026-10-24): compare the refitted L2 and L3 chains, and the rebuilt Phase 17B serology comparison,
with the 2026-10-12 to 2026-10-14 outputs. The L3B fits are not refitted. No fitting and no simulation here.

    python -B compare_c1.py > c1_compare.txt

Sections
  A  the ascertainment prior centre actually used at each origin (L2 and L3 c1 runs)
  B  serology comparison: the rebuild from the original trajectories reproduces the original file, and the c1 file
     keeps every L3B and survey column unchanged
  C  Phase 16 and Phase 17 criteria and mean primary errors, original against c1 (Table S4 rows)
  D  donor serology on the 21 common validation dates: L3 against L3B residual summaries and band counts,
     original against c1 (abstract, Results, Figure 2, Table S10)
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--p16", default="phase16_layer2"); ap.add_argument("--p16c", default="phase16_layer2_c1")
ap.add_argument("--p17", default="phase17_layer3"); ap.add_argument("--p17c", default="phase17_layer3_c1")
ap.add_argument("--p17b", default="phase17b_layer3b")
ap.add_argument("--sero-check", default="phase17b_serology_rebuild_check")
ap.add_argument("--sero-c1", default="phase17b_serology_c1")
ap.add_argument("--envcheck", default="c1_envcheck_l2", help="optional protocol_v1 rerun of the San Diego primary L2 chain")
A = ap.parse_args()
R = Path(A.root)
C = ["San Diego", "Los Angeles", "San Francisco"]
ok_all = True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flag(name, ok):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def js(p):
    return json.loads(Path(p).read_text())


print("Input hashes (SHA-256, first 16)")
for d in (A.p16, A.p16c, A.p17, A.p17c, A.p17b, A.sero_check, A.sero_c1):
    for f in ("params_states.csv", "summary.csv", "criteria.json", "serology_check.csv"):
        p = R / d / f
        if p.exists():
            print(f"  {d}/{f} {sha(p)[:16]}")

print("\nA. Ascertainment prior centre by origin")
for d, model in ((A.p16c, "L2"), (A.p17c, "L3")):
    P = pd.read_csv(R / d / "params_states.csv", parse_dates=["origin"])
    exp = np.where(P["origin"] >= pd.Timestamp("2020-11-25"), 1 / 7.7, 1 / 12.0)
    flag(f"{d}: rho_prior is 1/12 before 2020-11-25 and 1/7.7 from then ({len(P)} rows, models "
         f"{sorted(P['model'].unique())})", np.allclose(P["rho_prior"].values, exp, rtol=0, atol=1e-12))
P = pd.read_csv(R / A.p17b / "params_states.csv")
flag(f"{A.p17b} (not refitted): L3B rho_prior equals the serology value at every origin, so L3B never used the "
     "corrected schedule", np.allclose(P["rho_prior"], P["rho_sero"], rtol=0, atol=0) and P["rho_sero"].notna().all())

print("\nB. Serology comparison rebuilt from trajectories (L3B fits untouched)")
orig = R / A.p17b / "serology_check.csv"; chk = R / A.sero_check / "serology_check.csv"; new = R / A.sero_c1 / "serology_check.csv"
a = pd.read_csv(orig, float_precision="round_trip"); b = pd.read_csv(chk, float_precision="round_trip")
flag("rebuild from the original L2 and L3 trajectories reproduces the original serology_check.csv "
     f"({'byte for byte' if sha(orig) == sha(chk) else 'bytes differ'})", sha(orig) == sha(chk) or a.equals(b))
c = pd.read_csv(new, float_precision="round_trip")
cmp_cols = ["L3_phase17_ever_compartments", "L2_ever_compartments"]
keep = [x for x in a.columns if x not in cmp_cols]
flag(f"c1 file: columns other than {cmp_cols} identical to the original", list(a.columns) == list(c.columns) and a[keep].equals(c[keep]))
print(f"  L3 comparison column changed at {int((a[cmp_cols[0]] != c[cmp_cols[0]]).sum())} of {len(a)} rows")

print("\nC. Phase 16 (L2) and Phase 17 (L3) criteria and mean primary errors (primary chain, delay 1)")
for old, new, model in ((A.p16, A.p16c, "L2"), (A.p17, A.p17c, "L3")):
    co, cn = js(R / old / "criteria.json"), js(R / new / "criteria.json")
    so, sn = pd.read_csv(R / old / "summary.csv"), pd.read_csv(R / new / "summary.csv")
    print(f"  {model}: met flags original -> c1: " + ", ".join(
        f"{k} {co.get(k)} -> {cn.get(k)}" for k in co if k.endswith("_met")))
    for c in C:
        e = lambda s, m: float(s[(s["county"] == c) & (s["model"] == m) & (s["delay"] == 1)]["mean_err"].iloc[0])
        d1o, d1n = co["D1"][c], cn["D1"][c]
        print(f"    {c}: mean error {model} {e(so, model):.2f} -> {e(sn, model):.2f}; ARIMA_ADM {e(so, 'ARIMA_ADM'):.2f} -> "
              f"{e(sn, 'ARIMA_ADM'):.2f}; {model} minus ARIMA 95% interval [{d1o['ci95'][0]:.2f}, {d1o['ci95'][1]:.2f}] -> "
              f"[{d1n['ci95'][0]:.2f}, {d1n['ci95'][1]:.2f}]")
    if "serology" in cn:
        for c in C:
            print(f"    {c}: Phase 17 pooled survey criterion within/n {co['serology'][c]['within']}/{co['serology'][c]['n_dates']}"
                  f" -> {cn['serology'][c]['within']}/{cn['serology'][c]['n_dates']}")
    if "plausibility" in cn:
        print(f"    plausibility original {co['plausibility']}\n    plausibility c1       {cn['plausibility']}")

print("\nD. Donor validation dates (21 per county): model minus survey, L3 against L3B")
for label, d in (("original", A.p17b), ("c1", A.sero_c1)):
    s = pd.read_csv(R / d / "serology_check.csv")
    v = s[s["used_for"] == "validation"].copy()
    v["dB"] = v["L3B_ever_compartments"] - v["sero"]; v["d3"] = v["L3_phase17_ever_compartments"] - v["sero"]
    print(f"  {label}")
    for c in C:
        x = v[v["county"] == c]; m = x[["dB", "d3"]].notna().all(axis=1)
        asym = lambda q: int(((q >= -0.10) & (q <= 0.20)).sum())
        sym = lambda q, w: int((q.abs() <= w).sum())
        print(f"    {c}: n {int(m.sum())}; mean |L3 - survey| {x.loc[m, 'd3'].abs().mean():.3f}; mean |L3B - survey| "
              f"{x.loc[m, 'dB'].abs().mean():.3f}; asymmetric band L3 {asym(x['d3'])}, L3B {asym(x['dB'])}; "
              f"L3 symmetric 0.05/0.10/0.20 {sym(x['d3'], .05)}/{sym(x['d3'], .10)}/{sym(x['d3'], .20)}; "
              f"L3B {sym(x['dB'], .05)}/{sym(x['dB'], .10)}/{sym(x['dB'], .20)}")
        early = x[pd.to_datetime(x["date"]) < pd.Timestamp("2022-01-01")]
        print(f"      2020 and 2021 only ({len(early)} dates): mean |L3 - survey| {early['d3'].abs().mean():.3f}; "
              f"mean L3 - survey {early['d3'].mean():+.3f}")

print("\nE. Optional environment check: protocol_v1 rerun of the San Diego primary L2 chain against the original")
pe = R / A.envcheck / "params_states.csv"
if pe.exists():
    key = ["county", "model", "delay", "origin"]
    x = pd.read_csv(R / A.p16 / "params_states.csv"); y = pd.read_csv(pe)
    x = x.merge(y[key], on=key).sort_values(key).reset_index(drop=True); y = y.sort_values(key).reset_index(drop=True)
    cols = [c for c in y.columns if pd.api.types.is_numeric_dtype(y[c]) and c in x.columns and c not in ("nfev",)]
    d = np.abs(x[cols].values - y[cols].values)
    print(f"  rows {len(y)}; identical: {bool(np.nanmax(d) == 0)}; largest absolute difference {np.nanmax(d):.3g} "
          f"(column {cols[int(np.nanargmax(np.nanmax(d, axis=0)))]})")
    print("  Identical means this computer reproduces the original fits, so every c1 difference comes from the "
          "corrected prior. Small differences mean optimizer level noise is also present.")
else:
    print("  not run")

print("\nALL CHECKS PASS" if ok_all else "\nSOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
