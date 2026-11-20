#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrected weekly HHS admissions per county (Phase 15 Addendum B). Data construction only; no model.

Bug fixed (found 2026-10-11 while testing Phase 15): in epi_data_audit.py (Phase 12, PHASE12 Addendum A3) a
facility week was counted as "reporting" only when the adult admission value was not suppressed. Suppressed
values (-999999, meaning 1 to 3 admissions) are reports of a small number, not missing reports. In low
incidence weeks many facilities are suppressed, so those weeks failed the 90 % completeness rule and were
set to missing, then filled by interpolation in Phases 13 and 14 (for example San Diego June 2021: about
5 admissions per day observed, about 20 per day after interpolation).

Corrected rule: a facility reports in a week if its adult confirmed admission value is present, whether a
number or the suppression flag. A week is usable if the number of reporting facilities is at least 90 % of
the county's usual number of reporting hospitals (median over 2020-08-02 to 2022-12-31). Admission values
are unchanged: suppressed values count as 0 (adm_per_day, primary) or 2 (adm_per_day_supp2, sensitivity),
pediatric missing counts as 0, exactly as in epi_data_audit.py.

Output: <out>/weekly_<City>.csv with week_start (Sunday), adm_per_day, adm_per_day_supp2 and the counts,
plus the Phase 12 series for comparison, and a manifest. The column names match what layer1b_hospital.py
and layer1c_recalibration.py read, so --audit <out> points either script to the corrected series.

Usage (project root):  python hhs_admissions_v2.py --out phase15_admissions
"""
import argparse, hashlib, json, sys, platform, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--hhs", default="data_epi/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility.csv")
ap.add_argument("--old-audit", default="phase12_epi_audit")
ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
ap.add_argument("--out", default="phase15_admissions")
ap.add_argument("--overwrite", action="store_true")
A = ap.parse_args()
FIPS = {"San Diego": 6073, "Los Angeles": 6037, "San Francisco": 6075}
CITY = {"San Diego": "SanDiego", "Los Angeles": "LosAngeles", "San Francisco": "SanFrancisco"}
USUAL = ("2020-08-02", "2022-12-31"); COMPLETE = 0.9; SUPP = -999999
C_W, C_F = "collection_week", "fips_code"
C_AD = "previous_day_admission_adult_covid_confirmed_7_day_sum"
C_PD = "previous_day_admission_pediatric_covid_confirmed_7_day_sum"

out = Path(A.out)
if out.exists() and any(out.iterdir()) and not A.overwrite:
    raise SystemExit(f"{out} is not empty; use a new --out or --overwrite")
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "run_log.txt", "a", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


counties = [c.strip() for c in A.counties.split(",") if c.strip()]
parts = []
for ch in pd.read_csv(A.hhs, usecols=[C_W, C_F, C_AD, C_PD], chunksize=500_000, low_memory=False):
    f = pd.to_numeric(ch[C_F], errors="coerce")
    parts.append(ch[f.isin([FIPS[c] for c in counties])].assign(_f=f))
h = pd.concat(parts)
h["week"] = pd.to_datetime(h[C_W], errors="coerce")
for c in (C_AD, C_PD):
    h[c] = pd.to_numeric(h[c], errors="coerce")
    h[c + "_supp"] = h[c] == SUPP
    h[c + "_val"] = h[c].where(h[c] >= 0)
assert (h["week"].dt.dayofweek == 6).all(), "collection weeks are expected to start on Sunday"

summary = {}
for county in counties:
    g = h[h["_f"] == FIPS[county]].groupby("week")
    w = pd.DataFrame({
        "adm_adult": g[C_AD + "_val"].sum(min_count=1), "adm_ped": g[C_PD + "_val"].sum(min_count=1),
        "n_facilities": g[C_AD].size(), "n_value_adult": g[C_AD + "_val"].count(),
        "n_supp_adult": g[C_AD + "_supp"].sum(),
        "n_suppressed": g[C_AD + "_supp"].sum() + g[C_PD + "_supp"].sum()})
    w["n_reporting"] = w["n_value_adult"] + w["n_supp_adult"]
    w["adm_per_day"] = (w["adm_adult"].fillna(0.0) + w["adm_ped"].fillna(0.0)) / 7.0
    w.loc[w["n_reporting"] == 0, "adm_per_day"] = np.nan
    w["adm_per_day_supp2"] = w["adm_per_day"] + 2.0 * w["n_suppressed"] / 7.0
    sel = (w.index >= pd.Timestamp(USUAL[0])) & (w.index <= pd.Timestamp(USUAL[1]))
    usual = float(w.loc[sel, "n_reporting"].median())
    w["usual_reporting"] = usual
    w["adm_complete"] = w["n_reporting"] >= COMPLETE * usual
    for c in ("adm_per_day", "adm_per_day_supp2"):
        w.loc[~w["adm_complete"], c] = np.nan
    w.index.name = "week_start"
    # comparison with the Phase 12 series
    old_p = Path(A.old_audit) / f"weekly_{CITY[county]}.csv"
    if old_p.exists():
        old = pd.read_csv(old_p, parse_dates=["week_start"]).set_index("week_start")
        w["phase12_adm_per_day"] = old["adm_per_day"].reindex(w.index)
        both = w["adm_per_day"].notna() & w["phase12_adm_per_day"].notna()
        maxdiff = float((w.loc[both, "adm_per_day"] - w.loc[both, "phase12_adm_per_day"]).abs().max()) if both.any() else None
    else:
        maxdiff = None
    w.to_csv(out / f"weekly_{CITY[county]}.csv")
    ww = w[sel]
    summary[county] = {"usual_reporting_hospitals": usual, "weeks": int(len(ww)),
                       "missing_weeks_corrected": int(ww["adm_per_day"].isna().sum()),
                       "missing_weeks_phase12": int(ww["phase12_adm_per_day"].isna().sum()) if "phase12_adm_per_day" in ww else None,
                       "max_abs_diff_where_both_present": maxdiff,
                       "missing_weeks_list": [str(d.date()) for d in ww.index[ww["adm_per_day"].isna()]]}
    log(f"{county}: usual reporting {usual:.0f}; missing weeks {summary[county]['missing_weeks_corrected']} "
        f"(Phase 12: {summary[county]['missing_weeks_phase12']}); max difference where both present {maxdiff}")
(out / "summary.json").write_text(json.dumps(summary, indent=2))
man = {"script": Path(__file__).name, "script_sha256": sha(__file__), "command": " ".join([sys.executable] + sys.argv),
       "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version.split()[0],
       "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__,
       "inputs_sha256": {A.hhs: sha(A.hhs), **{str(p): sha(p) for p in sorted(Path(A.old_audit).glob("weekly_*.csv"))}},
       "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".json")}}
(out / "run_manifest.json").write_text(json.dumps(man, indent=2))
log(f"Saved to {out}/")
