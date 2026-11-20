#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correction C1 (2026-10-24): rebuild the L2 and L3 comparison columns of the Phase 17B serology check from refitted
L2 and L3 trajectories. The L3B and survey columns are copied unchanged; the L3B fits are not touched. The comparison
columns are recomputed with the same rule as layer3b_sero.py (model_at: state at the survey date from the fit at the
first origin on or after that date, primary chain, delay 1). No fitting and no simulation.

    python -B serology_compare_c1.py --phase16 phase16_layer2_c1/trajectories.csv
        --phase17 phase17_layer3_c1/trajectories.csv --out phase17b_serology_c1

Running it with the original phase16_layer2 and phase17_layer3 trajectories reproduces the original file (check).
"""
import argparse, datetime as dt, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from layer3b_sero import model_at


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--serology", default="phase17b_layer3b/serology_check.csv")
    ap.add_argument("--phase16", default="phase16_layer2_c1/trajectories.csv")
    ap.add_argument("--phase17", default="phase17_layer3_c1/trajectories.csv")
    ap.add_argument("--out", default="phase17b_serology_c1")
    A = ap.parse_args(argv)
    out = Path(A.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(A.serology, float_precision="round_trip")   # copied columns keep every digit
    # trajectories are parsed with the default reader, exactly as layer3b_sero.py parsed them
    p16 = pd.read_csv(A.phase16, usecols=["county", "model", "delay", "origin", "date", "R", "E", "I"])
    p17 = pd.read_csv(A.phase17, usecols=["county", "model", "delay", "origin", "date", "E", "I", "R_pre", "R_O1", "R_O2"])
    p16["date"] = pd.to_datetime(p16["date"]); p17["date"] = pd.to_datetime(p17["date"])
    l3, l2 = [], []
    for _, r in S.iterrows():
        d = pd.Timestamp(r["date"])
        m3 = model_at(p17, r["county"], "L3", 1, d, ["E", "I", "R_pre", "R_O1", "R_O2"])
        m2 = model_at(p16, r["county"], "L2", 1, d, ["R", "E", "I"])
        l3.append(np.nan if m3 is None else m3["R_pre"] + m3["R_O1"] + m3["R_O2"] + m3["E"] + m3["I"])
        l2.append(np.nan if m2 is None else m2["R"] + m2["E"] + m2["I"])
    S["L3_phase17_ever_compartments"] = l3
    S["L2_ever_compartments"] = l2
    S.to_csv(out / "serology_check.csv", index=False)
    man = {"script": Path(__file__).name, "script_sha256": sha(__file__), "command": " ".join([sys.executable] + sys.argv),
           "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0],
           "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
           "inputs_sha256": {p: sha(p) for p in (A.serology, A.phase16, A.phase17)},
           "outputs_sha256": {"serology_check.csv": sha(out / "serology_check.csv")}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"rows {len(S)}; L3 values {int(np.isfinite(l3).sum())}; L2 values {int(np.isfinite(l2).sum())}; saved to {out}/")


if __name__ == "__main__":
    main()
