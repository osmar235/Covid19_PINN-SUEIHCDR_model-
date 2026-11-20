#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 18 diagnosis: why do the moving horizon fits keep correcting the winter 2020 to 2021 rise downward?
Uses only existing outputs (phase17b_layer3b) and the observed data. No fitting, no new simulation.
Hypotheses and diagnostics were declared in REVIEW_LOG.md (2026-10-14) before this script was run:
  H1 timing: the model's lag from reported cases to admissions (cases at the E to I flow, admissions at the
     I outflow, mean 7 days apart) is longer than in the data                                  -> D3, D2
  H2 a changing admissions per case ratio within a window (one rho and one ihr per window)     -> D4, D2
  H3 the beta smoothness penalty too tight at the turn of the wave                            -> D5, D1
  H4 holiday reporting of cases (Thanksgiving, Christmas, New Year)                           -> D2 by date
  D1 revision pattern: estimate of new infections and admissions on a date from fits where that date sits at
     different positions in the window, relative to the first week estimate (what the next fit carries)
  D2 fit residuals (model minus observed, log) of 7 day cases and weekly admissions, by window week and by date
  D3 lag between cases and admissions: observed best lag (weekly growth correlation; CASELAG's chosen lag per
     origin) against the same statistic on the model's own fitted series; peak dates
  D4 observed admissions per case (cases 7 days earlier) by week, against the fitted ihr / rho by origin
  D5 share of each fit's objective taken by the beta smoothness penalty, and the largest second difference
Usage (project root):  python diagnose_phase18.py --out phase18_diagnosis
"""
import argparse, hashlib, json, sys, platform, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import layer2_admissions as L2

WINTER = ("2020-10-01", "2021-03-31")          # dates studied
ORIGINS = ("2020-11-28", "2021-05-22")         # fits whose window lies in the winter (as Phase 18 behaviour period)
HOLIDAYS = ["2020-11-26", "2020-12-25", "2021-01-01"]
COUNTIES = ["San Diego", "Los Angeles", "San Francisco"]


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def lag_corr(cases_d, adm_wk, lags=range(-7, 22)):
    """Correlation of weekly log growth of admissions with weekly log growth of cases shifted by L days
    (cases summed over the Sunday week starting L days earlier)."""
    out = {}
    g_a = np.log(adm_wk.clip(lower=0.5)).diff()
    for L in lags:
        cs = pd.Series({ws: float(cases_d.loc[ws - pd.Timedelta(days=L):ws - pd.Timedelta(days=L - 6)].sum()) for ws in adm_wk.index})
        g_c = np.log(cs.clip(lower=0.5)).diff()
        j = pd.concat([g_a, g_c], axis=1).dropna()
        out[L] = float(j.corr().iloc[0, 1]) if len(j) > 4 else np.nan
    s = pd.Series(out)
    if s.isna().all():
        return None, s
    return int(s.idxmax()), s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--adm-dir", default="phase15_admissions")
    ap.add_argument("--vax", default="data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv")
    ap.add_argument("--counties", default=",".join(COUNTIES))
    ap.add_argument("--smoke", action="store_true", help="smoke tests only: October 2020 dates, two fits")
    ap.add_argument("--out", default="phase18_diagnosis")
    A = ap.parse_args(argv)
    out = Path(A.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "diagnosis.txt", "w", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    root = Path(A.root); inputs = {}
    win = (pd.Timestamp(WINTER[0]), pd.Timestamp("2020-10-31") if A.smoke else pd.Timestamp(WINTER[1]))
    org = (pd.Timestamp("2020-10-03"), pd.Timestamp("2020-10-10")) if A.smoke else tuple(pd.Timestamp(x) for x in ORIGINS)
    pp, tp, fp = root / "phase17b_layer3b/params_states.csv", root / "phase17b_layer3b/trajectories.csv", root / "phase17b_layer3b/forecasts_weekly.csv"
    for p in (pp, tp, fp):
        inputs[str(p)] = sha(p)
    P = pd.read_csv(pp, parse_dates=["origin"]); P = P[(P["model"] == "L3B") & (P["delay"] == 1)]
    T = pd.read_csv(tp, parse_dates=["origin", "date"]); T = T[(T["model"] == "L3B") & (T["delay"] == 1)]
    F = pd.read_csv(fp, parse_dates=["origin"]); F = F[(F["model"] == "CASELAG") & (F["delay"] == 1) & (F["h"] == 1)]
    T["pos"] = (T["date"] - (T["origin"] - pd.Timedelta(days=L2.WINDOW))).dt.days
    T = T.merge(P[["county", "origin", "rho", "ihr"]], on=["county", "origin"])
    T["new_inf"] = T["cases_model"] / T["rho"]            # sigma E N
    rows_d1, rows_d2, rows_d3, rows_d4, rows_d5 = [], [], [], [], []
    for county in [c.strip() for c in A.counties.split(",")]:
        D = L2.load_county(str(root), county, A.adm_dir, str(root / A.vax), set())
        cases_d = pd.Series(L2.daily_from_cum_np(D["cum_cases"].values), index=D["cum_cases"].index)
        cases7 = pd.Series(L2.weekly_avg_np(cases_d.values), index=cases_d.index)
        wk = D["wk"].dropna()
        t = T[T["county"] == county]
        # D1 revision pattern
        fin = t[(t["pos"] >= 0) & (t["pos"] <= 6)].set_index("date")[["new_inf", "adm_model"]]
        tw = t[(t["date"] >= win[0]) & (t["date"] <= win[1])].join(fin, on="date", rsuffix="_final").dropna(subset=["new_inf_final"])
        tw = tw.assign(lr_inf=np.log(tw["new_inf"] / tw["new_inf_final"]), lr_adm=np.log(tw["adm_model"] / tw["adm_model_final"]))
        grow = np.log(wk.clip(lower=0.5)).diff().shift(-1)        # growth from this week to the next
        wk_of = lambda d: d - pd.Timedelta(days=(d.dayofweek + 1) % 7)
        tw["phase"] = [("growth" if grow.get(wk_of(d), np.nan) > 0 else "decline") for d in tw["date"]]
        tw["pos_week"] = tw["pos"] // 7
        g = tw.groupby(["phase", "pos_week"])[["lr_inf", "lr_adm"]].median().reset_index(); g.insert(0, "county", county)
        rows_d1.append(g)
        # D2 residuals of the fits inside the window
        for o in sorted(t["origin"].unique()):
            o = pd.Timestamp(o)
            if not (org[0] <= o <= org[1]):
                continue
            to = t[(t["origin"] == o) & (t["pos"] < L2.WINDOW)].set_index("date")
            mc7 = pd.Series(L2.weekly_avg_np(to["cases_model"].values), index=to.index)
            for d in to.index[6:]:
                rows_d2.append({"county": county, "origin": o, "date": d, "pos": int(to.loc[d, "pos"]), "series": "cases7",
                                "resid": float(np.log1p(mc7[d]) - np.log1p(cases7[d]))})
            last_pub = o - pd.Timedelta(days=7)
            for ws, v in wk.items():
                if ws >= to.index.min() and ws + pd.Timedelta(days=6) <= min(to.index.max(), last_pub):
                    m = float(t[(t["origin"] == o) & (t["date"] >= ws) & (t["date"] <= ws + pd.Timedelta(days=6))]["adm_model"].mean())
                    rows_d2.append({"county": county, "origin": o, "date": ws, "pos": int((ws - (o - pd.Timedelta(days=L2.WINDOW))).days),
                                    "series": "admissions", "resid": float(np.log1p(m) - np.log1p(v))})
        # D3 lags and peaks
        aw = wk[(wk.index >= win[0]) & (wk.index <= win[1])]
        L_obs, s_obs = lag_corr(cases_d, aw)
        st = fin.copy(); st["cases"] = t[(t["pos"] >= 0) & (t["pos"] <= 6)].set_index("date")["cases_model"]
        madm = pd.Series({ws: float(st["adm_model"].loc[ws:ws + pd.Timedelta(days=6)].mean()) for ws in aw.index}).dropna()
        L_mod, s_mod = lag_corr(st["cases"], madm) if len(madm) > 5 else (None, None)
        cl = F[(F["county"] == county) & (F["origin"] >= org[0]) & (F["origin"] <= org[1])]["caselag_L"]
        c7w = cases7[(cases7.index >= win[0]) & (cases7.index <= win[1])]
        stc = st["cases"].rolling(7, min_periods=7).mean(); stc = stc[(stc.index >= win[0]) & (stc.index <= win[1])]
        rows_d3.append({"county": county, "best_lag_obs_days": L_obs, "corr_obs": float(s_obs.max()) if L_obs is not None else None,
                        "best_lag_model_days": L_mod, "corr_model": float(s_mod.max()) if L_mod is not None else None,
                        "caselag_L_median": float(cl.median()) if len(cl) else None,
                        "caselag_L_range": [int(cl.min()), int(cl.max())] if len(cl) else None,
                        "peak_cases7_obs": str(c7w.idxmax().date()), "peak_adm_obs_week": str(aw.idxmax().date()),
                        "peak_cases7_model": str(stc.idxmax().date()) if len(stc.dropna()) else None,
                        "peak_adm_model_week": str(madm.idxmax().date()) if len(madm) else None})
        # D4 observed admissions per case (7 days earlier) and fitted ihr / rho
        for ws, v in aw.items():
            c = float(cases_d.loc[ws - pd.Timedelta(days=7):ws - pd.Timedelta(days=1)].sum())
            rows_d4.append({"county": county, "week_start": ws, "kind": "observed adm / cases 7 d earlier", "value": v * 7 / c if c > 0 else np.nan})
        po = P[(P["county"] == county) & (P["origin"] >= org[0]) & (P["origin"] <= org[1])]
        for _, r in po.iterrows():
            rows_d4.append({"county": county, "week_start": r["origin"] - pd.Timedelta(days=L2.WINDOW), "kind": "fit ihr / rho (window start)",
                            "value": r["ihr"] / r["rho"]})
        # D5 smoothness penalty
        for _, r in po.iterrows():
            lk = np.log(np.array(json.loads(r["beta_knots"]), float)); rw = np.diff(lk, 2) / L2.BETA_RW_SD
            rows_d5.append({"county": county, "origin": r["origin"], "penalty_ssq": float((rw ** 2).sum()), "total_ssq": 2 * float(r["cost"]),
                            "share": float((rw ** 2).sum() / (2 * r["cost"])), "max_abs_d2": float(np.abs(np.diff(lk, 2)).max()),
                            "beta_first": float(np.exp(lk[0])), "beta_last": float(np.exp(lk[-1]))})
    D1 = pd.concat(rows_d1); D2 = pd.DataFrame(rows_d2); D3 = pd.DataFrame(rows_d3); D4 = pd.DataFrame(rows_d4); D5 = pd.DataFrame(rows_d5)
    D1.to_csv(out / "d1_revisions.csv", index=False); D2.to_csv(out / "d2_residuals.csv", index=False)
    D3.to_csv(out / "d3_lags_peaks.csv", index=False); D4.to_csv(out / "d4_adm_per_case.csv", index=False); D5.to_csv(out / "d5_penalty.csv", index=False)
    log("D1 revision: median log(estimate at window week k / first week estimate), new infections and admissions")
    log(D1.pivot_table(index=["county", "pos_week"], columns="phase", values=["lr_inf", "lr_adm"]).round(3).to_string())
    if len(D2):
        D2["pos_week"] = D2["pos"] // 7; D2["month"] = D2["date"].dt.strftime("%Y-%m")
        log("\nD2 residuals (model minus observed, log), median by window week"); log(D2.pivot_table(index=["county", "pos_week"], columns="series", values="resid", aggfunc="median").round(3).to_string())
        log("\nD2 residuals, median by calendar month"); log(D2.pivot_table(index=["county", "month"], columns="series", values="resid", aggfunc="median").round(3).to_string())
        near = D2[(D2["series"] == "cases7")].copy()
        near["near_holiday"] = near["date"].apply(lambda d: any(0 <= (d - pd.Timestamp(h)).days <= 13 for h in HOLIDAYS))
        log("\nD2 case residuals within 2 weeks after a holiday or not (median)"); log(near.groupby(["county", "near_holiday"])["resid"].median().round(3).to_string())
    log("\nD3 lags (days) and peaks"); log(D3.to_string(index=False))
    log("\nD4 admissions per case: observed by month (median) and fitted ihr / rho by month of window start (median)")
    D4["month"] = pd.to_datetime(D4["week_start"]).dt.strftime("%Y-%m")
    log(D4.pivot_table(index=["county", "month"], columns="kind", values="value", aggfunc="median").round(4).to_string())
    D5["month"] = (D5["origin"] - pd.Timedelta(days=L2.WINDOW)).dt.strftime("%Y-%m")
    log("\nD5 beta smoothness penalty share of the objective and largest second difference of log beta (median by month of window start)")
    log(D5.groupby(["county", "month"])[["share", "max_abs_d2", "beta_first", "beta_last"]].median().round(3).to_string())
    me = Path(__file__).resolve()
    man = {"script": me.name, "script_sha256": sha(me), "layer2_sha256": sha(L2.__file__), "command": " ".join([sys.executable] + sys.argv),
           "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": platform.python_version(),
           "numpy": np.__version__, "pandas": pd.__version__, "inputs_sha256": inputs,
           "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.suffix == ".csv"}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    log(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
