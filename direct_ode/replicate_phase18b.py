#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 18B independent replication (REVIEW_LOG 2026-10-19), to check the Phase 18B run made on 2026-10-15
(projections_phase18b.py, SHA-256 6c17d706...). Written from the approved draft before that code was seen;
after seeing it, two changes only: the last beta at full precision (beta_last in params_states.csv) and the
Phase 17B week grid added for comparison.
No new fitting. For each Saturday origin t0 from 2020-12-12 to 2021-10-02 the projection starts from the
official Phase 17B fit (L3B delay 1) at t0: state on t0 from trajectories.csv, last beta knot, ihr, a_b and
omega_b from params_states.csv, and the inputs that fit used (behaviour signal held, doses held at the last
known 7 day pace, variant shares and escape values as known on t0), rebuilt with the same problem object.
Weeks are reported on two grids of observed HHS weeks (Sunday to Saturday):
  grid "after_t0": 8 weeks starting t0 + 1 (all days projected);
  grid "p17b": the Phase 17B forecast grid used by the 2026-10-15 run, week 1 = t0 - 6 .. t0 (days before t0
  taken from the fitted trajectory), 8 weeks.
  P0 status quo; P1 beta x 0.8 and P2 beta x 1.2 from t0 + 7; P3 doses x 1.5 and P4 doses x 0.5 from t0
  (only when the held pace is above 0); H0 (hindsight, reported) the doses actually given.
Usage (project root, .venv Python):  python replicate_phase18b.py --out phase18b_replication
"""
import argparse, json, sys, platform, datetime as dt, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import layer2_admissions as L2
import layer3b_sero as B

ORIGINS = ("2020-12-12", "2021-10-02")
WEEKS = 8; HDAYS = 7 * WEEKS
BETA_DOWN, BETA_UP, BETA_DELAY = 0.8, 1.2, 7
VAX_UP, VAX_DOWN = 1.5, 0.5
NAME, DELAY = "L3B", 1
STATE = ["S", "U_v", "U_b", "E", "I", "R_pre", "R_O1", "R_O2", "CUM"]
E1_MIN, E1_WEEKS = 0.5, (1, 2, 3, 4)


def load_fits(root, inputs, fits="phase17b_layer3b"):
    pp, tp, fp = (Path(root) / fits / f for f in ("params_states.csv", "trajectories.csv", "forecasts_weekly.csv"))
    for p in (pp, tp, fp):
        inputs.add(str(p))
    P = pd.read_csv(pp, parse_dates=["origin"]); P = P[(P["model"] == NAME) & (P["delay"] == DELAY)]
    T = pd.read_csv(tp, usecols=["county", "model", "delay", "origin", "date", "adm_model"] + STATE, parse_dates=["origin", "date"])
    T = T[(T["model"] == NAME) & (T["delay"] == DELAY)]
    F = pd.read_csv(fp, parse_dates=["origin", "week_start"]); F = F[(F["model"] == NAME) & (F["delay"] == DELAY)]
    return P, T, F


def origin_inputs(D, county, t0, P, T, om, cdc, vax_cum, rounds, ca):
    """Inputs of the official fit at t0 for days t0 .. t0 + 55, and its state on t0."""
    r = P[(P["county"] == county) & (P["origin"] == t0)]
    if len(r) != 1:
        raise RuntimeError(f"no unique fit {county} {t0.date()}")
    r = r.iloc[0]
    K = L2.known_inputs(D, t0, DELAY)
    Pr = B.Problem3B(K, D, t0, np.zeros(9), True, HDAYS, om, cdc, B.booster_weight(vax_cum, t0),
                     rho_sero=B.rho_sero_at(t0, rounds, ca)[0])
    i0 = L2.WINDOW
    tr = T[(T["county"] == county) & (T["origin"] == t0) & (T["date"] == t0)]
    if len(tr) != 1:
        raise RuntimeError(f"no state on t0 {county} {t0.date()}")
    beta_last = float(r["beta_last"])                    # full precision (the knots are stored to 5 decimals)
    trw = T[(T["county"] == county) & (T["origin"] == t0) & (T["date"] < t0)].set_index("date")["adm_model"]
    return {"adm_before_t0": trw, "t0": t0, "days": pd.date_range(t0, periods=HDAYS, freq="D"), "x0": tr[STATE].to_numpy(float)[0],
            "beta": np.full(HDAYS, beta_last), "s": Pr.s_d[i0:i0 + HDAYS].copy(), "dose": Pr.dose_d[i0:i0 + HDAYS].copy(),
            "sh": tuple(a[i0:i0 + HDAYS].copy() for a in Pr.sh), "f": dict(Pr.f), "ve": Pr.ve, "omv": Pr.omv,
            "a_b": float(r["a_b"]), "omega_b": float(r["omega_b"]), "ihr": float(r["ihr"]), "N": D["N"],
            "last_ws": K["wk"].dropna().index.max(), "last_wk_value": float(K["wk"].dropna().iloc[-1]),
            "check": {"VE": (Pr.ve, float(r["VE"])), "omega_v": (Pr.omv, float(r["omega_v"])), "f_V": (Pr.f["V"], float(r["f_V"])),
                      "rho_prior": (Pr.rho_prior, float(r["rho_prior"]))}}


def project(o, beta=None, dose=None):
    X, fE, fI = B.simulate3(o["x0"], o["beta"] if beta is None else beta, o["dose"] if dose is None else dose, o["s"], o["sh"],
                            o["a_b"], o["omega_b"], o["omv"], o["ve"], o["f"])
    d = pd.DataFrame(X[:-1], columns=STATE); d.insert(0, "date", o["days"])
    d["adm"] = fI * o["N"] * o["ihr"]; d["new_infectious"] = fE * o["N"]; d["pop_sum"] = d[STATE[:8]].sum(axis=1)
    return d


def scenarios(o, actual_doses=None):
    b1 = o["beta"].copy(); b1[BETA_DELAY:] *= BETA_DOWN
    b2 = o["beta"].copy(); b2[BETA_DELAY:] *= BETA_UP
    out = {"P0": project(o), "P1": project(o, beta=b1), "P2": project(o, beta=b2)}
    if o["dose"].max() > 0:
        out["P3"] = project(o, dose=o["dose"] * VAX_UP); out["P4"] = project(o, dose=o["dose"] * VAX_DOWN)
    if actual_doses is not None:
        out["H0"] = project(o, dose=actual_doses.reindex(o["days"]).fillna(0.0).to_numpy(float))
    return out


def weekly(path, t0, grid="after_t0", before=None):
    a = path.set_index("date")["adm"]
    if grid == "p17b":
        a = pd.concat([before[before.index >= t0 - pd.Timedelta(days=6)], a])
        first = t0 - pd.Timedelta(days=6)
    else:
        first = t0 + pd.Timedelta(days=1)
    rows = []
    for k in range(1, WEEKS + 1):
        ws = first + pd.Timedelta(days=7 * (k - 1))
        seg = a.loc[ws:ws + pd.Timedelta(days=6)]
        rows.append({"k": k, "week_start": ws, "pred": float(seg.mean()) if len(seg) == 7 else np.nan})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--adm-dir", default="phase15_admissions")
    ap.add_argument("--vax", default="data_epi/statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv")
    ap.add_argument("--counties", default="San Diego,Los Angeles,San Francisco")
    ap.add_argument("--max-origins", type=int, default=0, help="smoke tests only")
    ap.add_argument("--out", default="phase18b_replication")
    A = ap.parse_args(argv)
    out = Path(A.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run_log.txt", "w", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush()

    root = A.root; inputs = set(); vax = str(Path(root) / A.vax)
    Pf, Tf, Ff = load_fits(root, inputs)
    om, cdc = B.load_variants(root, inputs)
    rounds = B.load_commercial_rounds(root, inputs); ca = B.load_ca_cases(root, inputs)
    origins = list(pd.date_range(*ORIGINS, freq="7D"))
    if A.max_origins:
        origins = origins[:A.max_origins]
    counties = [c.strip() for c in A.counties.split(",")]
    log(f"Phase 18B replication | counties {counties} | origins {len(origins)} ({origins[0].date()} to {origins[-1].date()}) | {WEEKS} weeks")
    wrows, srows, checks = [], [], []
    for county in counties:
        D = L2.load_county(root, county, A.adm_dir, vax, inputs); vc = B.load_vax_cum(vax, county, inputs)
        wk = D["wk"]; t_ = time.time()
        for t0 in origins:
            o = origin_inputs(D, county, t0, Pf, Tf, om, cdc, vc, rounds, ca)
            paths = scenarios(o, actual_doses=D["doses"])
            f17 = Ff[(Ff["county"] == county) & (Ff["origin"] == t0)].set_index("week_start")["pred"]
            w0 = weekly(paths["P0"], t0, "p17b", o["adm_before_t0"]).set_index("week_start")["pred"]
            j = pd.concat([w0.rename("p0"), f17.rename("f17")], axis=1, sort=True).dropna()
            checks.append({"county": county, "origin": str(t0.date()), "n_weeks_vs_17b": int(len(j)),
                           "max_rel_vs_17b": float((j["p0"] / j["f17"] - 1).abs().max()) if len(j) else np.nan,
                           "inputs_match": bool(all(abs(u - v) < 1e-12 for u, v in o["check"].values())),
                           "max_abs_pop_error": float(max((p["pop_sum"] - 1).abs().max() for p in paths.values()))})
            for sc, p in paths.items():
              for grid in ("after_t0", "p17b"):
                w = weekly(p, t0, grid, o["adm_before_t0"])
                for _, r in w.iterrows():
                    wrows.append({"grid": grid, "county": county, "origin": str(t0.date()), "scenario": sc, "k": int(r["k"]),
                                  "week_start": str(r["week_start"].date()), "pred": r["pred"],
                                  "obs": float(wk.get(r["week_start"], np.nan)), "naive": o["last_wk_value"]})
              srows.append({"county": county, "origin": str(t0.date()), "scenario": sc, "admissions_8w": float(p["adm"].sum()),
                            "infections_8w": float(p["new_infectious"].sum())})
        log(f"  {county}: {len(origins)} origins ({time.time() - t_:.0f} s)")
    W = pd.DataFrame(wrows); W.to_csv(out / "projections_weekly.csv", index=False)
    S = pd.DataFrame(srows)
    base = S[S["scenario"] == "P0"].set_index(["county", "origin"])["admissions_8w"]
    S["pct_vs_P0"] = [100 * (a / base[(c, o)] - 1) for c, o, a in zip(S["county"], S["origin"], S["admissions_8w"])]
    S.to_csv(out / "projections_summary.csv", index=False)
    C = pd.DataFrame(checks); C.to_csv(out / "checks.csv", index=False)
    # E1 envelope coverage over P0 to P4
    env = W[W["scenario"].isin(["P0", "P1", "P2", "P3", "P4"])].groupby(["grid", "county", "origin", "k"]).agg(lo=("pred", "min"), hi=("pred", "max"), obs=("obs", "first")).reset_index()
    env = env.dropna(subset=["obs"]); env["inside"] = (env["obs"] >= env["lo"]) & (env["obs"] <= env["hi"])
    env.to_csv(out / "e1_envelope.csv", index=False)
    crit = {"E1": {"after_t0": {}, "p17b": {}}, "rules": {"E1": f"observed weekly admissions between the lowest and highest of P0 to P4 in at least {E1_MIN:.0%} of (origin, week) pairs for weeks 1 to 4, in at least 2 of 3 counties"}}
    for grid in ("after_t0", "p17b"):
      for c in counties:
        e = env[(env["county"] == c) & (env["grid"] == grid)]
        a = e[e["k"].isin(E1_WEEKS)]; b = e[~e["k"].isin(E1_WEEKS)]
        crit["E1"][grid][c] = {"coverage_w1_4": float(a["inside"].mean()) if len(a) else None, "n_w1_4": int(len(a)),
                         "coverage_w5_8": float(b["inside"].mean()) if len(b) else None, "n_w5_8": int(len(b)),
                         "met": bool(len(a) and a["inside"].mean() >= E1_MIN)}
      crit[f"E1_met_{grid}"] = bool(sum(v["met"] for v in crit["E1"][grid].values()) >= 2)
    # E2 drift of P0 and the naive projection by week ahead, growth and decline weeks
    p0 = W[W["scenario"] == "P0"].dropna(subset=["obs", "pred"]).copy()
    p0 = p0.sort_values(["grid", "county", "origin", "k"])
    prev = p0.groupby(["grid", "county", "origin"])["obs"].shift(1)
    p0["prev_obs"] = prev.fillna(p0["naive"])
    p0["phase"] = np.where(p0["obs"] > p0["prev_obs"], "growth", "decline")
    p0["lr_P0"] = np.log1p(p0["pred"]) - np.log1p(p0["obs"]); p0["lr_naive"] = np.log1p(p0["naive"]) - np.log1p(p0["obs"])
    E2 = p0.groupby(["grid", "county", "phase", "k"]).agg(median_log_ratio_P0=("lr_P0", "median"), median_abs_P0=("lr_P0", lambda x: float(np.median(np.abs(x)))),
                                                   median_log_ratio_naive=("lr_naive", "median"), median_abs_naive=("lr_naive", lambda x: float(np.median(np.abs(x)))),
                                                   n=("lr_P0", "size")).reset_index()
    E2.to_csv(out / "e2_drift.csv", index=False)
    # E3 hindsight doses
    h = W[W["scenario"].isin(["P0", "H0"])].pivot_table(index=["grid", "county", "origin", "k"], columns="scenario", values=["pred", "obs"], aggfunc="first")
    h.columns = [f"{a}_{b}" for a, b in h.columns]; h = h.reset_index().dropna()
    h["lr_P0"] = np.log1p(h["pred_P0"]) - np.log1p(h["obs_P0"]); h["lr_H0"] = np.log1p(h["pred_H0"]) - np.log1p(h["obs_P0"])
    E3 = h.groupby(["grid", "county", "k"])[["lr_P0", "lr_H0"]].median().reset_index(); E3.to_csv(out / "e3_hindsight.csv", index=False)
    crit["checks"] = {"max_rel_vs_17b": float(C["max_rel_vs_17b"].max()), "inputs_match_all": bool(C["inputs_match"].all()),
                      "max_abs_pop_error": float(C["max_abs_pop_error"].max())}
    (out / "criteria.json").write_text(json.dumps(crit, indent=2))
    for grid in ("after_t0", "p17b"):
        log(f"E1 {grid}:", json.dumps({k: {kk: v[kk] for kk in ("coverage_w1_4", "coverage_w5_8", "met")} for k, v in crit["E1"][grid].items()}), "| met", crit[f"E1_met_{grid}"])
    log("checks:", json.dumps(crit["checks"]))
    me = Path(__file__).resolve()
    import scipy
    man = {"script": me.name, "script_sha256": B.sha(me), "layer3b_sha256": B.sha(Path(B.__file__)), "layer2_sha256": B.sha(Path(L2.__file__)),
           "command": " ".join([sys.executable] + sys.argv), "args": vars(A), "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": scipy.__version__, "inputs_sha256": {p: B.sha(p) for p in sorted(inputs)},
           "outputs_sha256": {f.name: B.sha(f) for f in sorted(out.iterdir()) if f.suffix in (".csv", ".json")}}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=2))
    log(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
