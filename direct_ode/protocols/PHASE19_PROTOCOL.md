# Phase 19 protocol (APPROVED by Osmar 2026-10-14; frozen): ascertainment that follows testing within a window

Approved text = PHASE19_PROTOCOL_DRAFT.md, SHA-256 f676a52d0be34185182b6056f0af7625bb8f78c36f328bf42fd957bdac04e015, all four points of section 4
confirmed by Osmar ("agree, lets do it"). Addendum A records the implementation before any official run. The
draft file is kept unchanged.

---


Written 2026-10-14, after the Phase 18 diagnosis (`phase18_diagnosis`). Osmar chose "Phase 19: changing
ascertainment" on 2026-10-14. Once approved this file is frozen; later changes go to REVIEW_LOG.md.

## 0. Why (from the diagnosis, no new fits)
* The winter 2020 to 2021 drift of Phase 18 comes from an inconsistency between consecutive windows, not
  from a poor fit inside a window (median residuals within about 0.07 on the log scale), from the beta
  smoothness penalty (0 to 7 % of the objective) or from holiday reporting.
* Observed admissions per reported case roughly double from January to February 2021 (Los Angeles 0.058 to
  0.117, San Diego 0.046 to 0.094), and admissions peaked about 6 days before reported cases in San Diego
  and Los Angeles. Each fit has one ascertainment rho for 8 weeks, so when the window moves, the carried E
  and I no longer match the newer data and are corrected down.
* Ascertainment depends on how much testing is done. A relation published during the epidemic links them.

## 1. Change (L3T = Phase 17B L3B plus ascertainment that follows testing)
* Reported cases on day t = rho_w x g(t) x sigma E(t) x N, with g(t) = (T(t) / T_w)^0.5.
  - T(t): tests per day over the 14 days ending t, by specimen collection date, for the county (CDPH
    `statewide-covid-19-cases-deaths-tests.csv`, already in data_epi).
  - T_w: geometric mean of T over the window days, so rho_w stays the window's average ascertainment and
    keeps the Phase 17B prior centred on rho_sero (log sd 0.3).
  - The exponent 0.5 comes from Chiu and Ndeffo-Mbah, medRxiv 10.1101/2020.10.07.20208504, posted
    2020-10-12 (PLoS Comput Biol 2021;17(9):e1009374). They found infections proportional to the square root
    of positivity times reported cases, fitted on US states (n about 0.5, close to 0.5 in nearly all states).
    That relation is equivalent to ascertainment proportional to the square root of tests per person.
  - Used from its public date: origins before 2020-10-12 (2020-10-03 and 2020-10-10) keep g = 1, as Phase
    17B.
* Real time rule for tests: known for collection dates up to t0 minus 7 (judgment; CDPH states only that
  recent dates are incomplete and later updated). Later days hold the last known T. The file is the final
  vintage, so backfill inside the known period cannot be undone (stated limitation).
* Admissions, escape, vaccination, behaviour, priors, serology calibration, chains, comparators and the real
  time rule are otherwise exactly as Phase 17B. Chains L3T delay 1 (primary), L3T delay 0, L3T_NOBEHAV
  delay 1.
* Stated limitations: one exponent for all counties and periods; from 2022 home tests are not counted, so T
  falls and the model lowers ascertainment. That direction is plausible, but it is not checked here.

## 2. Criteria (declared before any fit)
* **Window consistency (primary for this phase):** the Phase 18 reconstruction run unchanged on the L3T
  fits; R1 (total admissions within 20 % of observed, peak week within 2 weeks) for the behaviour period
  (2020-10-03 to 2021-04-02) met in at least 2 of 3 counties, and R1 and R2 for the vaccination period still
  met in all three.
* **Kept from Phase 17B:** plausibility (beta at its upper bound and rho at a bound in at most 10 % of
  origins, every county); serology on the blood donor surveys in all three counties; adequacy (not
  significantly worse than ARIMA_ADM in at least 2 of 3 counties).
* Reported, not criteria: forecast error against L3B (Phase 17B) by period; the weekly E and I corrections in
  December 2020; the 2022 serology dates; Phase 18 scenario effects recomputed on L3T.
* **Decision:** all met: L3T replaces L3B as the base model, Phase 18 scenarios are reported on L3T, and
  Phase 18B is drafted on L3T. Window consistency not met: L3B stays the base model and the result is reported
  as a tested and rejected explanation. Serology or plausibility lost: L3T is rejected.

## 3. Reproducibility
New script `layer3t_tests.py` (a copy of `layer3b_sero.py` with only the changes above), tests extended to
the testing data (tests after t0 minus 7 replaced by random values must not change any forecast; g = 1 before
2020-10-12; g has geometric mean 1 over the window), reduced smoke tests by me, official runs by Osmar:
`layer3t_tests.py --out phase19_layer3t`, then `scenarios_phase18.py` pointed at the new fits (one new
argument for the input folder; outputs to `phase19_scenarios`).

## 4. Points for Osmar to confirm
1. Ascertainment within a window proportional to the square root of tests per person (Chiu and Ndeffo-Mbah,
   public 2020-10-12), window average still centred on the serology value.
2. County tests by collection date, 14 day sum, known through t0 minus 7 and held afterwards.
3. The criteria (window consistency through the Phase 18 reconstruction, plus the Phase 17B criteria) and
   the decision rule.
4. Everything else unchanged from Phase 17B, all 114 origins, same chains and comparators.

## Addendum A (2026-10-14, implementation, before any official run)

A1. `layer3t_tests.py` (SHA-256 89156808901a8d48...) is a copy of `layer3b_sero.py` (a3d6e950...) with only:
`load_tests` and `tests_factor` (T and g as in section 1), class `Problem3T` (cases multiplied by g; states and
admissions unchanged, tested), an extra `tests14` argument to `l3_origin`, model names L3T and L3T_NOBEHAV,
g at the window start and at t0 minus 1 written to params_states.csv, the Phase 17B L3B ever infected written
next to L3T in serology_check.csv, and output folder `phase19_layer3t`. It imports `layer2_admissions.py`
(4b70ab2b..., unchanged).
A2. T(t) = total tests over the 14 days ending t divided by 14, requiring 14 days with data. T_w is the
geometric mean over the 56 window days, the last 7 of which hold the value known at t0 minus 7. If any
window day has no known T, g = 1 for that origin (counted from params_states.csv in the results).
A3. On forecast days g is held at its last known value; it multiplies reported cases only, so admissions
forecasts change only through the fitted states and rates.
A4. `scenarios_phase18.py` gains `--fits` (input folder, default `phase17b_layer3b`) and `--name` (model
name, default L3B); new SHA-256 93aaf8c9bff2a209.... With the defaults it reproduces the Phase 18 reduced
smoke outputs byte for byte (tested); the official Phase 18 version is archived as
`code_archive/scenarios_phase18_v338095ff.py`.
A5. Tests (`test_layer3t.py`, SHA-256 1077d33c4218ed7e...): g = 1 before 2020-10-12; g has geometric mean 1
over the window and is held after t0 minus 7 (three dates, three counties); cases equal the Phase 17B cases
times g with identical states and admissions; the Phase 17B serology tests; causality at 2021-01-09 and
2022-01-08 in three counties with tests after t0 minus 7 also replaced by random values. 24 of 24 passed in the
cloud (the test prints g ranges, which describe inputs, not results). Reduced smoke runs (Los Angeles, first
origin 2020-11-28, 3 origins, 1 start, 30 evaluations; then the reconstruction on those fits, behaviour
period, 3 weeks): `smoke_tests/smoke_p19_reduced` and `smoke_tests/smoke_p19_scenarios_reduced`; code paths
only.
