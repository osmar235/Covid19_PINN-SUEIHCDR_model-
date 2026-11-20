#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 12, epidemiology data audit (no model fitting, no torch).

Purpose: describe, from data only, how the clinical course and the population immunity of COVID-19
changed over 2020 to 2022 in San Diego, Los Angeles and San Francisco counties, to decide which parts
of the compartmental model must change (protocol: PHASE12_PROTOCOL.md).

Core inputs (already in the project root):
  covid_confirmed_usafacts.csv, covid_deaths_usafacts.csv, covid_county_population_usafacts.csv (USAFacts)
  covid19hospitalbycounty.csv (CHHS hospital census)
Optional inputs (used when given; columns are detected and every choice is printed and saved):
  --cdph-ts   CDPH "COVID-19 Time-Series Metrics by County and State": cases by episode date and
              deaths by date of death (better epidemiological timing than USAFacts report dates)
  --vax       CDPH "Statewide COVID-19 Vaccines Administered By County" (cumulative doses by county)
  --hhs-adm   HHS "COVID-19 Reported Patient Impact and Hospital Capacity by Facility" (weekly confirmed
              COVID admissions per facility, aggregated here to county by FIPS code)

Outputs in --out:
  weekly_<county>.csv        weekly series and ratios
  period_summary.csv         medians per pre declared period and county
  feasibility_current_model.csv  share of weeks where the observed deaths per occupied ICU bed lies
                             outside what the current SUEIHCDR bounds allow (all deaths pass through ICU)
  lag_summary.csv            lag (days) maximizing the correlation cases -> hospital census -> deaths
  column_choices.json        which columns of each optional file were used
  audit_<county>.png         figure (if matplotlib is available)
  audit_manifest.json        SHA-256 of script, inputs and outputs, versions, command

Usage (project root):
  python epi_data_audit.py --out phase12_epi_audit
  python epi_data_audit.py --cdph-ts <file> --vax <file> --hhs-adm <file> --out phase12_epi_audit_full
"""
import argparse, hashlib, json, sys, platform, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-path", default=".")
ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
ap.add_argument("--start", default="2020-03-29")
ap.add_argument("--end", default="2022-12-31")
ap.add_argument("--cdph-ts", default=None)
ap.add_argument("--vax", default=None)
ap.add_argument("--hhs-adm", default=None)
ap.add_argument("--out", default="phase12_epi_audit")
ap.add_argument("--overwrite", action="store_true")
A = ap.parse_args()

# Pre declared periods (PHASE12_PROTOCOL.md section 3), fixed before looking at any audit output.
PERIODS = [("P1_FirstWave", "2020-03-29", "2020-09-30"), ("P2_Winter20", "2020-10-01", "2021-03-31"),
           ("P3_Delta", "2021-04-01", "2021-11-30"), ("P4_Omicron", "2021-12-01", "2022-04-30"),
           ("P5_BA2_BA5", "2022-05-01", "2022-12-31")]
# Current SUEIHCDR bounds (core_phase10.py): deaths only from ICU at rate f*epsi.
F_BOUNDS = (0.35, 0.65); EPSI_BOUNDS = (1 / 21, 1 / 5)
DPI_MIN, DPI_MAX = F_BOUNDS[0] * EPSI_BOUNDS[0], F_BOUNDS[1] * EPSI_BOUNDS[1]
FIPS = {"San Diego": 6073, "Los Angeles": 6037, "San Francisco": 6075}

D = Path(A.data_path)
out = Path(A.out)
if out.exists() and any(out.iterdir()) and not A.overwrite:
    raise SystemExit(f"{out} is not empty; use a new --out or --overwrite")
out.mkdir(parents=True, exist_ok=True)
inputs, choices = [], {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def pick(cols, candidates, what, required=True):
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise SystemExit(f"Could not find {what}; tried {candidates}; available columns: {list(cols)}")
    return None


def usafacts(county, kind):
    p = D / f"covid_{kind}_usafacts.csv"; inputs.append(str(p))
    w = pd.read_csv(p)
    r = w[w["countyFIPS"] == FIPS[county]].iloc[0].drop(["countyFIPS", "County Name", "State", "StateFIPS"])
    s = pd.Series(r.values.astype(float), index=pd.to_datetime(r.index))
    return s.diff().clip(lower=0)


def population(county):
    p = D / "covid_county_population_usafacts.csv"; inputs.append(str(p))
    pop = pd.read_csv(p, encoding="utf-8-sig")
    return float(pop.loc[pop["countyFIPS"] == FIPS[county], "population"].iloc[0])


def hospital(county):
    p = D / "covid19hospitalbycounty.csv"; inputs.append(str(p))
    h = pd.read_csv(p)
    h = h[h["county"] == county].copy()
    h["date"] = pd.to_datetime(h["todays_date"], format="mixed")
    h = h[h["date"] <= pd.Timestamp(A.end)].drop_duplicates("date", keep="last").set_index("date").sort_index()
    o = pd.DataFrame(index=h.index)
    o["hosp"] = h["hospitalized_covid_confirmed_patients"].astype(float)
    o["icu"] = h["icu_covid_confirmed_patients"].astype(float)
    o["ward"] = o["hosp"] - o["icu"]
    o["hosp_susp"] = h["hospitalized_suspected_covid_patients"].astype(float)
    o["icu_avail"] = h["icu_available_beds"].astype(float)
    o["all_beds"] = h["all_hospital_beds"].astype(float)
    return o


def cdph_ts(county):
    if not A.cdph_ts:
        return None
    p = Path(A.cdph_ts); inputs.append(str(p))
    t = pd.read_csv(p, low_memory=False)
    c_date = pick(t.columns, ["date"], "CDPH date column")
    c_area = pick(t.columns, ["area", "county"], "CDPH area column")
    c_cases = pick(t.columns, ["cases"], "CDPH cases (episode date)")
    c_deaths = pick(t.columns, ["deaths"], "CDPH deaths (date of death)")
    choices["cdph_ts"] = {"date": c_date, "area": c_area, "cases": c_cases, "deaths": c_deaths}
    t = t[t[c_area] == county].copy()
    t["date"] = pd.to_datetime(t[c_date], errors="coerce")
    t = t.dropna(subset=["date"]).set_index("date").sort_index()
    return pd.DataFrame({"cases_ep": pd.to_numeric(t[c_cases], errors="coerce"),
                         "deaths_dod": pd.to_numeric(t[c_deaths], errors="coerce")})


def vaccination(county, N):
    if not A.vax:
        return None
    p = Path(A.vax); inputs.append(str(p))
    v = pd.read_csv(p, low_memory=False)
    c_date = pick(v.columns, ["administered_date", "date"], "vaccine date column")
    c_cty = pick(v.columns, ["county"], "vaccine county column")
    c_full = pick(v.columns, ["cumulative_fully_vaccinated", "cumulative_primary_series_complete"],
                  "cumulative primary series column")
    c_one = pick(v.columns, ["cumulative_at_least_one_dose"], "cumulative at least one dose", required=False)
    c_boost = pick(v.columns, ["cumulative_booster_recip_count", "cumulative_booster_count"],
                   "cumulative booster column", required=False)
    choices["vax"] = {"date": c_date, "county": c_cty, "primary": c_full, "one_dose": c_one, "booster": c_boost}
    v = v[v[c_cty] == county].copy()
    v["date"] = pd.to_datetime(v[c_date], errors="coerce")
    v = v.dropna(subset=["date"]).set_index("date").sort_index()
    o = pd.DataFrame({"vax_primary_frac": pd.to_numeric(v[c_full], errors="coerce") / N})
    if c_one:
        o["vax_onedose_frac"] = pd.to_numeric(v[c_one], errors="coerce") / N
    if c_boost:
        o["vax_booster_frac"] = pd.to_numeric(v[c_boost], errors="coerce") / N
    return o


_HHS = {}
def hhs_admissions(county):
    """Weekly confirmed COVID admissions (adult + pediatric 7 day sums) summed over facilities in the county.
    Suppressed values (-999999, i.e. 1 to 3) are counted as missing and the number of suppressed facility
    weeks is reported."""
    if not A.hhs_adm:
        return None
    if not _HHS:
        p = Path(A.hhs_adm); inputs.append(str(p))
        head = pd.read_csv(p, nrows=0).columns
        c_week = pick(head, ["collection_week"], "HHS week column")
        c_fips = pick(head, ["fips_code"], "HHS fips column")
        c_ad = pick(head, ["previous_day_admission_adult_covid_confirmed_7_day_sum"], "HHS adult admissions")
        c_pd = pick(head, ["previous_day_admission_pediatric_covid_confirmed_7_day_sum"], "HHS pediatric admissions")
        choices["hhs_adm"] = {"week": c_week, "fips": c_fips, "adult": c_ad, "pediatric": c_pd}
        parts = []
        for ch in pd.read_csv(p, usecols=[c_week, c_fips, c_ad, c_pd], chunksize=500_000, low_memory=False):
            f = pd.to_numeric(ch[c_fips], errors="coerce")
            parts.append(ch[f.isin(list(FIPS.values()))].assign(_f=f))
        h = pd.concat(parts)
        h["week"] = pd.to_datetime(h[c_week], errors="coerce")
        for c in (c_ad, c_pd):
            h[c] = pd.to_numeric(h[c], errors="coerce")
            h[c + "_supp"] = h[c] == -999999
            h.loc[h[c] < 0, c] = np.nan
        _HHS["df"] = h; _HHS["cols"] = (c_ad, c_pd)
    h = _HHS["df"]; c_ad, c_pd = _HHS["cols"]
    h = h[h["_f"] == FIPS[county]]
    gb = h.groupby("week")
    g = pd.DataFrame({
        # min_count=1: a week in which no facility reported stays missing instead of becoming 0
        "adm_adult": gb[c_ad].sum(min_count=1), "adm_ped": gb[c_pd].sum(min_count=1),
        "n_facilities": gb[c_ad].size(), "n_reporting": gb[c_ad].count(),
        "n_suppressed": gb[c_ad + "_supp"].sum() + gb[c_pd + "_supp"].sum()})
    # Primary: suppressed values (1 to 3) counted as 0; sensitivity: counted as 2 (midpoint).
    g["adm_per_day"] = (g["adm_adult"] + g["adm_ped"].fillna(0.0)) / 7.0
    g["adm_per_day_supp2"] = g["adm_per_day"] + 2.0 * g["n_suppressed"] / 7.0
    g.loc[g["adm_adult"].isna(), "adm_per_day_supp2"] = np.nan
    return g


def best_lag(x, y, lags=range(0, 36)):
    """Lag L (days) maximizing corr(x(t - L), y(t)) on log scale; x, y daily series."""
    best = (np.nan, np.nan)
    lx, ly = np.log1p(x), np.log1p(y)
    for L in lags:
        a = lx.shift(L); ok = a.notna() & ly.notna()
        if ok.sum() < 30:
            continue
        r = np.corrcoef(a[ok], ly[ok])[0, 1]
        if not np.isfinite(best[1]) or r > best[1]:
            best = (L, r)
    return best


period_rows, feas_rows, lag_rows = [], [], []
for county in [c.strip() for c in A.counties.split(",") if c.strip()]:
    N = population(county)
    idx = pd.date_range(A.start, A.end, freq="D")
    d = pd.DataFrame(index=idx)
    d["cases"] = usafacts(county, "confirmed").reindex(idx)
    d["deaths"] = usafacts(county, "deaths").reindex(idx)
    d = d.join(hospital(county).reindex(idx))
    ts = cdph_ts(county)
    if ts is not None:
        d = d.join(ts.reindex(idx))
    vx = vaccination(county, N)
    if vx is not None:
        d = d.join(vx.reindex(idx).ffill())
    for c in ["cases", "deaths"] + (["cases_ep", "deaths_dod"] if ts is not None else []):
        d[c + "7"] = d[c].rolling(7, min_periods=7).mean()
    # Lags estimated on the whole series (descriptive)
    for a_, b_ in [("cases7", "hosp"), ("hosp", "deaths7")] + ([("cases_ep7", "hosp"), ("hosp", "deaths_dod7")]
                                                               if ts is not None else []):
        L, r = best_lag(d[a_], d[b_])
        lag_rows.append({"county": county, "from": a_, "to": b_, "best_lag_days": L, "corr_log": r})
    # With the CDPH file, cases by episode date and deaths by date of death are used for all ratios
    # (epidemiological timing); otherwise USAFacts report dates.
    ccol = "cases_ep7" if ts is not None else "cases7"
    lag_ch = int(next(r["best_lag_days"] for r in lag_rows if r["county"] == county and r["from"] == ccol))
    dcol = "deaths_dod7" if ts is not None else "deaths7"
    d["cases7_lagged"] = d[ccol].shift(lag_ch)
    d["cum_cases_frac"] = d["cases"].fillna(0).cumsum() / N
    # Weekly table (weeks starting Sunday, matching HHS collection weeks). Ratios are formed from weekly
    # means (ratio of means), which is robust to batch reporting days.
    w = d.resample("W-SAT").mean(numeric_only=True)
    w.index = w.index - pd.Timedelta(days=6)
    w.index.name = "week_start"
    w["icu_share"] = w["icu"] / w["hosp"]
    w["deaths_per_icu"] = w[dcol] / w["icu"]
    w["deaths_per_hosp"] = w[dcol] / w["hosp"]
    w["hosp_per_lagged_case7"] = w["hosp"] / w["cases7_lagged"]
    w["deaths_min_outside_icu"] = (w[dcol] - DPI_MAX * w["icu"]).clip(lower=0)
    w["susp_share"] = w["hosp_susp"] / (w["hosp"] + w["hosp_susp"])
    adm = hhs_admissions(county)
    if adm is not None:
        w = w.join(adm, how="left")
        # Reporting completeness: admissions are used only in weeks where the number of reporting
        # facilities is at least 90 % of the county's median over the study period (early weeks of the
        # HHS series have few reporting facilities and would inflate the implied length of stay).
        _med = float(w["n_reporting"].median())
        w["adm_complete"] = w["n_reporting"] >= 0.9 * _med
        for _c in ["adm_per_day", "adm_per_day_supp2"]:
            w.loc[~w["adm_complete"], _c] = np.nan
        w["implied_LOS_days"] = (w["hosp"] / w["adm_per_day"]).replace([np.inf, -np.inf], np.nan)  # census / inflow
        w["adm_per_lagged_case"] = (w["adm_per_day"] / w[ccol].shift(1)).replace([np.inf, -np.inf], np.nan)
    cname = county.replace(" ", "")
    w.to_csv(out / f"weekly_{cname}.csv")
    ratio_cols = [c for c in ["icu_share", "deaths_per_icu", "deaths_per_hosp", "hosp_per_lagged_case7",
                              "susp_share", "implied_LOS_days", "adm_per_lagged_case", "vax_primary_frac",
                              "vax_booster_frac", "cum_cases_frac"] if c in w.columns]
    for name, a0, b0 in PERIODS:
        sub = w.loc[a0:b0]
        row = {"county": county, "period": name, "weeks": len(sub)}
        for c in ratio_cols:
            row[f"{c}_median"] = float(sub[c].replace([np.inf, -np.inf], np.nan).median())
        period_rows.append(row)
        dpi = sub["deaths_per_icu"].replace([np.inf, -np.inf], np.nan).dropna()
        feas_rows.append({"county": county, "period": name, "weeks_with_data": int(len(dpi)),
                          "share_below_model_min": float((dpi < DPI_MIN).mean()) if len(dpi) else np.nan,
                          "share_above_model_max": float((dpi > DPI_MAX).mean()) if len(dpi) else np.nan,
                          "model_min": DPI_MIN, "model_max": DPI_MAX, "deaths_series": dcol})
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        ax[0].plot(w.index, w["cases7"], label="cases 7 d mean (USAFacts)")
        if "cases_ep7" in w: ax[0].plot(w.index, w["cases_ep7"], label="cases by episode date (CDPH)")
        ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title(f"{county}: data only audit")
        ax[1].plot(w.index, w["ward"], label="ward census"); ax[1].plot(w.index, w["icu"], label="ICU census")
        if "adm_per_day" in w: ax[1].plot(w.index, w["adm_per_day"], label="admissions per day (HHS)")
        ax[1].set_yscale("log"); ax[1].legend()
        ax[2].plot(w.index, w["icu_share"], label="ICU share of hospitalized")
        ax[2].plot(w.index, w["deaths_per_icu"], label=f"deaths per ICU patient per day ({dcol})")
        ax[2].axhspan(DPI_MIN, DPI_MAX, alpha=0.15, label="range allowed by current model")
        ax[2].set_ylim(0, 0.6); ax[2].legend()
        ax[3].plot(w.index, w["hosp_per_lagged_case7"], label=f"hospital census / {ccol} lagged {lag_ch} d (axis cut at 3)")
        for c in ["vax_primary_frac", "vax_booster_frac", "cum_cases_frac"]:
            if c in w: ax[3].plot(w.index, w[c], label=c)
        ax[3].set_ylim(0, 3); ax[3].legend()
        for name, a0, _ in PERIODS:
            for a_ in ax: a_.axvline(pd.Timestamp(a0), color="grey", lw=0.5)
        fig.tight_layout(); fig.savefig(out / f"audit_{cname}.png", dpi=110); plt.close(fig)
    except Exception as e:
        print(f"figure skipped for {county}: {type(e).__name__}: {e}")

P = pd.DataFrame(period_rows); F = pd.DataFrame(feas_rows); L = pd.DataFrame(lag_rows)
P.to_csv(out / "period_summary.csv", index=False)
F.to_csv(out / "feasibility_current_model.csv", index=False)
L.to_csv(out / "lag_summary.csv", index=False)
(out / "column_choices.json").write_text(json.dumps(choices, indent=2))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(L.round(3).to_string(index=False)); print()
print(F.round(3).to_string(index=False)); print()
print(P.round(3).T.to_string())

man = {"script": Path(__file__).name, "script_sha256": sha(__file__),
       "command": " ".join([sys.executable] + sys.argv), "args": vars(A),
       "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
       "python": sys.version.split()[0], "platform": platform.platform(),
       "numpy": np.__version__, "pandas": pd.__version__,
       "inputs_sha256": {p: sha(p) for p in sorted(set(inputs))},
       "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".png", ".json")
                          and f.name != "audit_manifest.json"}}
(out / "audit_manifest.json").write_text(json.dumps(man, indent=2))
print(f"\nSaved to {out}/")
