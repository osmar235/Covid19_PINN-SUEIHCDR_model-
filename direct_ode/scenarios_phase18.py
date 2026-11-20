#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 18: scenarios from the validated physical states, 2020 to 2021 (PHASE18_PROTOCOL.md with Addendum A).
No new fitting. Reads the official Phase 17B fits (phase17b_layer3b/params_states.csv and trajectories.csv),
rebuilds each fit's inputs on the first week of its window exactly as layer3b_sero.py built them, and runs
one continuous simulation of the L3B equations per county, chain, period and scenario.

Reconstruction: for the origin t0 whose window starts on w0 = t0 - 56, week w0 .. w0 + 6 uses that fit's
beta knots, a_b, omega_b, ihr, VE, omega_v, escape values, variant shares, behaviour signal s(t) and doses.
S0 starts from the fitted state on the first day of the period and is simulated without the weekly E and I
corrections. Scenarios change one input:
  V0 doses = 0;  V1 per capita doses of San Francisco;  V2 doses shifted 28 days earlier
  B0 a_b = 0 and U_b moved into S at the start;  B1 omega_b x 2
Usage (project root, .venv Python):  python scenarios_phase18.py --out phase18_scenarios
Phase 19 (PHASE19_PROTOCOL.md section 3) adds --fits and --name to run the same reconstruction on other fits:
    python scenarios_phase18.py --fits phase19_layer3t --name L3T --out phase19_scenarios
With the defaults the computation is identical to the official Phase 18 version (SHA-256 338095ff...).
"""
import argparse, hashlib, json, sys, platform, datetime as dt, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import layer2_admissions as L2
import layer3b_sero as B

PERIODS = {"vaccination": ("2020-12-12", "2021-11-27"), "behaviour": ("2020-10-03", "2021-03-27")}  # first and last week starts
SCEN = {"vaccination": ["S0", "V0", "V1", "V2"], "behaviour": ["S0", "B0", "B1"]}
CHAINS = [("L3B", 1), ("L3B", 0), ("L3B_NOBEHAV", 1)]
PRIMARY = ("L3B", 1)
FASTEST = "San Francisco"
SHIFT_DAYS = 28; OMB_FACTOR = 2.0
R1_TOTAL = 0.20; R1_PEAK_WEEKS = 2
VALIDATION_SOURCES = B.VALIDATION_SOURCES
STATE = ["S", "U_v", "U_b", "E", "I", "R_pre", "R_O1", "R_O2", "CUM"]


def sha(p):
    return B.sha(p)


def load_fits(root, inputs, fits="phase17b_layer3b"):
    pp = Path(root) / fits / "params_states.csv"; tp = Path(root) / fits / "trajectories.csv"
    inputs.add(str(pp)); inputs.add(str(tp))
    P = pd.read_csv(pp); P["origin"] = pd.to_datetime(P["origin"])
    T = pd.read_csv(tp, usecols=["county", "model", "delay", "origin", "date"] + STATE + ["adm_model"])
    T["origin"] = pd.to_datetime(T["origin"]); T["date"] = pd.to_datetime(T["date"])
    return P, T


def build_weeks(D, county, name, delay, P, T, first_w0, last_w0, om, cdc, vax_cum, rounds, ca):
    """One dict per week w0 .. w0 + 6 for w0 from first_w0 to last_w0, from the fit at origin w0 + 56."""
    behav = not name.endswith("_NOBEHAV")
    weeks = []
    for w0 in pd.date_range(first_w0, last_w0, freq="7D"):
        t0 = w0 + pd.Timedelta(days=L2.WINDOW)
        row = P[(P["county"] == county) & (P["model"] == name) & (P["delay"] == delay) & (P["origin"] == t0)]
        if len(row) != 1:
            raise RuntimeError(f"no unique fit for {county} {name} {delay} {t0.date()}")
        r = row.iloc[0]
        K = L2.known_inputs(D, t0, delay)
        Pr = B.Problem3B(K, D, t0, np.zeros(9), behav, 0, om, cdc, B.booster_weight(vax_cum, t0),
                         rho_sero=B.rho_sero_at(t0, rounds, ca)[0])
        logk = np.log(np.array(json.loads(r["beta_knots"]), float))
        tr = T[(T["county"] == county) & (T["model"] == name) & (T["delay"] == delay) & (T["origin"] == t0)].set_index("date")
        weeks.append({"w0": w0, "t0": t0, "days": pd.date_range(w0, periods=7, freq="D"),
                      "beta": L2.beta_series(logk, 7), "s": Pr.s_d[:7].copy(), "dose": Pr.dose_d[:7].copy(),
                      "sh": tuple(a[:7].copy() for a in Pr.sh), "f": dict(Pr.f), "ve": Pr.ve, "omv": Pr.omv,
                      "a_b": float(r["a_b"]), "omega_b": float(r["omega_b"]), "ihr": float(r["ihr"]),
                      "x_fit": tr.loc[pd.date_range(w0, periods=8, freq="D"), STATE].to_numpy(float),
                      "adm_fit": tr.loc[pd.date_range(w0, periods=7, freq="D"), "adm_model"].to_numpy(float),
                      "check": {"VE": (Pr.ve, float(r["VE"])), "omega_v": (Pr.omv, float(r["omega_v"])),
                                "f_V": (Pr.f["V"], float(r["f_V"])), "f_pre_O1": (Pr.f["pre_O1"], float(r["f_pre_O1"])),
                                "rho_prior": (Pr.rho_prior, float(r["rho_prior"]))}})
    return weeks


def run_path(weeks, x_start, N, dose_fn=None, ab_fn=None, omb_fn=None):
    """Continuous simulation over the weeks (no E and I corrections). Returns a daily DataFrame."""
    x = np.asarray(x_start, float).copy(); out = []
    for w in weeks:
        dose = w["dose"] if dose_fn is None else dose_fn(w)
        ab = w["a_b"] if ab_fn is None else ab_fn(w)
        omb = w["omega_b"] if omb_fn is None else omb_fn(w)
        X, fE, fI = B.simulate3(x, w["beta"], dose, w["s"], w["sh"], ab, omb, w["omv"], w["ve"], w["f"])
        df = pd.DataFrame(X[:-1], columns=STATE); df.insert(0, "date", w["days"])
        df["adm"] = fI * N * w["ihr"]; df["new_infectious"] = fE * N
        out.append(df); x = X[-1].copy()
    d = pd.concat(out, ignore_index=True)
    d["ever"] = d[["E", "I", "R_pre", "R_O1", "R_O2"]].sum(axis=1)
    d["pop_sum"] = d[STATE[:8]].sum(axis=1)
    return d


def scenario_paths(weeks, N, period, doses_fast=None, doses_own=None):
    x0 = weeks[0]["x_fit"][0].copy()
    paths = {"S0": run_path(weeks, x0, N)}
    if period == "vaccination":
        paths["V0"] = run_path(weeks, x0, N, dose_fn=lambda w: np.zeros(7))
        if doses_fast is not None:
            paths["V1"] = run_path(weeks, x0, N, dose_fn=lambda w: doses_fast.reindex(w["days"]).fillna(0.0).to_numpy(float))
        paths["V2"] = run_path(weeks, x0, N, dose_fn=lambda w: doses_own.reindex(w["days"] + pd.Timedelta(days=SHIFT_DAYS))
                               .fillna(0.0).to_numpy(float))
    else:
        xb = x0.copy(); xb[0] += xb[2]; xb[2] = 0.0
        paths["B0"] = run_path(weeks, xb, N, ab_fn=lambda w: 0.0)
        paths["B1"] = run_path(weeks, x0, N, omb_fn=lambda w: OMB_FACTOR * w["omega_b"])
    return paths


def weekly_obs_model(path, wk):
    """Observed and model admissions per day on the observed weeks (Sunday starts) fully inside the path."""
    a, b = path["date"].min(), path["date"].max()
    s = path.set_index("date")["adm"]; rows = []
    for ws, v in wk.items():
        if ws >= a and ws + pd.Timedelta(days=6) <= b:
            rows.append({"week_start": ws, "obs": float(v), "model": float(s.loc[ws:ws + pd.Timedelta(days=6)].mean())})
    return pd.DataFrame(rows)


def outcomes(path, N):
    blk = (path["date"] - path["date"].min()).dt.days // 7
    w = path.groupby(blk)["adm"].mean(); wd = path.groupby(blk)["date"].min()
    return {"admissions": float(path["adm"].sum()), "admissions_per_100k": float(path["adm"].sum() / N * 1e5),
            "peak_weekly_adm_per_day": float(w.max()), "peak_week_start": str(wd[w.idxmax()].date()),
            "infections": float(path["new_infectious"].sum()), "ever_infected_end": float(path["ever"].iloc[-1])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--adm-dir", default="phase15_admissions")
    ap.add_argument("--vax", default="data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv")
    ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
    ap.add_argument("--periods", default="vaccination,behaviour", help="smoke tests only")
    ap.add_argument("--chains", default="all", help="smoke tests only")
    ap.add_argument("--fits", default="phase17b_layer3b")
    ap.add_argument("--name", default="L3B")
    ap.add_argument("--max-weeks", type=int, default=0, help="smoke tests only (truncates every period)")
    ap.add_argument("--out", default="phase18_scenarios")
    ap.add_argument("--overwrite", action="store_true")
    A = ap.parse_args(argv)
    out = Path(A.out)
    if out.exists() and any(out.iterdir()) and not A.overwrite:
        sys.exit(f"{out} exists and is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run_log.txt", "w", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

    inputs = set(); root = A.root
    counties = [c.strip() for c in A.counties.split(",")]
    periods = [p.strip() for p in A.periods.split(",")]
    CH = [(A.name, 1), (A.name, 0), (A.name + "_NOBEHAV", 1)]; PRIM = (A.name, 1)
    chains = CH if A.chains == "all" else [c for c in CH if f"{c[0]}:{c[1]}" in A.chains.split(",")]
    vax = str(Path(root) / A.vax)
    Pf, Tf = load_fits(root, inputs, A.fits)
    om, cdc = B.load_variants(root, inputs)
    rounds = B.load_commercial_rounds(root, inputs); ca = B.load_ca_cases(root, inputs)
    sero = B.load_serology(root, inputs); sero = sero[sero["source"].isin(VALIDATION_SOURCES)]
    Dfast = L2.load_county(root, FASTEST, A.adm_dir, vax, inputs)
    log(f"Phase 18 | counties {counties} | periods {periods} | chains {chains}")
    daily, summ, recon, checks = [], [], [], []
    for county in counties:
        D = L2.load_county(root, county, A.adm_dir, vax, inputs); vc = B.load_vax_cum(vax, county, inputs); N = D["N"]
        wk = D["wk"].dropna()
        for period in periods:
            a, b = (pd.Timestamp(x) for x in PERIODS[period])
            if A.max_weeks:
                b = min(b, a + pd.Timedelta(days=7 * (A.max_weeks - 1)))
            for name, delay in chains:
                if period == "behaviour" and name.endswith("_NOBEHAV"):
                    continue
                t_ = time.time()
                weeks = build_weeks(D, county, name, delay, Pf, Tf, a, b, om, cdc, vc, rounds, ca)
                for w in weeks:      # stitched reconstruction with the corrections: one week from the fitted state
                    X, fE, fI = B.simulate3(w["x_fit"][0], w["beta"], w["dose"], w["s"], w["sh"], w["a_b"], w["omega_b"],
                                            w["omv"], w["ve"], w["f"])
                    rel = np.abs(X - w["x_fit"]).max() / max(np.abs(w["x_fit"]).max(), 1e-12)
                    rel_adm = np.abs(fI * N * w["ihr"] - w["adm_fit"]).max() / max(np.abs(w["adm_fit"]).max(), 1e-12)
                    checks.append({"county": county, "period": period, "model": name, "delay": delay, "w0": str(w["w0"].date()),
                                   "max_rel_state": float(rel), "max_rel_adm": float(rel_adm),
                                   "inputs_match": bool(all(abs(u - v) < 1e-12 for u, v in w["check"].values()))})
                own = D["doses"]; fast = Dfast["doses"] if county != FASTEST else None
                paths = scenario_paths(weeks, N, period, doses_fast=fast, doses_own=own)
                base = outcomes(paths["S0"], N)
                for sc, pth in paths.items():
                    pth = pth.copy(); pth.insert(0, "scenario", sc); pth.insert(0, "delay", delay); pth.insert(0, "model", name)
                    pth.insert(0, "period", period); pth.insert(0, "county", county); daily.append(pth)
                    o = outcomes(paths[sc], N)
                    row = {"county": county, "period": period, "model": name, "delay": delay, "scenario": sc, **o}
                    for k in ("admissions", "infections", "peak_weekly_adm_per_day", "ever_infected_end"):
                        row[f"diff_{k}"] = o[k] - base[k]
                        row[f"pct_{k}"] = 100.0 * (o[k] - base[k]) / base[k] if base[k] else np.nan
                    row["max_abs_pop_error"] = float((paths[sc]["pop_sum"] - 1.0).abs().max())
                    summ.append(row)
                wom = weekly_obs_model(paths["S0"], wk)
                tot_o, tot_m = float(wom["obs"].sum() * 7), float(wom["model"].sum() * 7)
                pk_o, pk_m = wom.loc[wom["obs"].idxmax(), "week_start"], wom.loc[wom["model"].idxmax(), "week_start"]
                s0 = paths["S0"].set_index("date")
                sv = sero[(sero["county"] == county) & (sero["date"] >= paths["S0"]["date"].min()) & (sero["date"] <= paths["S0"]["date"].max())]
                diffs = [float(s0.loc[d_, "ever"]) - v_ for d_, v_ in zip(sv["date"], sv["sero"])]
                within = [B.SERO_LO <= x_ <= B.SERO_HI for x_ in diffs]
                r1 = bool(abs(tot_m / tot_o - 1.0) <= R1_TOTAL and abs((pk_m - pk_o).days) <= 7 * R1_PEAK_WEEKS)
                r2 = bool(len(within) and sum(within) >= (2.0 / 3.0) * len(within))
                recon.append({"county": county, "period": period, "model": name, "delay": delay, "n_weeks": int(len(wom)),
                              "obs_total": tot_o, "S0_total": tot_m, "ratio": tot_m / tot_o, "obs_peak_week": str(pk_o.date()),
                              "S0_peak_week": str(pk_m.date()), "peak_shift_days": int((pk_m - pk_o).days),
                              "n_sero": len(within), "sero_within": int(sum(within)),
                              "sero_diffs": json.dumps([round(x_, 3) for x_ in diffs]), "R1": r1, "R2": r2, "primary_ok": r1 and r2})
                log(f"  {county} {period} {name} d{delay}: {len(weeks)} weeks, S0/obs {tot_m / tot_o:.2f}, peak shift "
                    f"{(pk_m - pk_o).days} d, serology {sum(within)}/{len(within)} ({time.time() - t_:.0f} s)")
    pd.concat(daily).to_csv(out / "scenario_daily.csv", index=False)
    S = pd.DataFrame(summ); S.to_csv(out / "scenario_summary.csv", index=False)
    R = pd.DataFrame(recon); R.to_csv(out / "reconstruction.csv", index=False)
    C = pd.DataFrame(checks); C.to_csv(out / "stitch_check.csv", index=False)
    crit = {"rules": {"R1": f"S0 total admissions within {int(R1_TOTAL * 100)} % of observed and peak week within {R1_PEAK_WEEKS} weeks",
                      "R2": "S0 ever infected within [-0.10, +0.20] of the blood donor survey at >= 2/3 of dates in the period",
                      "primary": f"chain {A.name} delay 1; a county and period meeting R1 and R2 is reported as primary"},
            "primary": {f"{r['county']}|{r['period']}": {"R1": r["R1"], "R2": r["R2"], "primary_ok": r["primary_ok"]}
                        for _, r in R[(R["model"] == PRIM[0]) & (R["delay"] == PRIM[1])].iterrows()},
            "stitch_check": {"max_rel_state": float(C["max_rel_state"].max()), "max_rel_adm": float(C["max_rel_adm"].max()),
                             "inputs_match_all": bool(C["inputs_match"].all())},
            "max_abs_pop_error": float(S["max_abs_pop_error"].max())}
    (out / "criteria.json").write_text(json.dumps(crit, indent=2))
    log("\nreconstruction (primary chain):"); log(R[(R["model"] == PRIM[0]) & (R["delay"] == PRIM[1])][
        ["county", "period", "ratio", "peak_shift_days", "n_sero", "sero_within", "R1", "R2"]].to_string(index=False))
    log("stitch check:", json.dumps(crit["stitch_check"]), "| population error", crit["max_abs_pop_error"])
    me = Path(__file__).resolve()
    import scipy, statsmodels
    man = {"script": me.name, "script_sha256": sha(me), "layer3b_sha256": sha(Path(B.__file__)),
           "layer2_sha256": sha(Path(L2.__file__)), "command": " ".join([sys.executable] + sys.argv), "args": vars(A),
           "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "python": platform.python_version(),
           "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
           "statsmodels": statsmodels.__version__,
           "inputs_sha256": {p: sha(p) for p in sorted(inputs)},
           "outputs_sha256": {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name not in ("run_log.txt", "run_manifest.json")}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    log(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
