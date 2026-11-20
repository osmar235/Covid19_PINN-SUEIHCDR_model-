# Phase 16 protocol (APPROVED by Osmar 2026-10-12; frozen): Layer 2, forecasting hospital admissions in real time

Approved text = PHASE16_PROTOCOL_DRAFT.md (version 2), SHA-256
ed51900a995eb4040800343f9ddd91f3c83c21168d5a55a34ce3e9ebb7aa47bf, with all four points of section 8
confirmed by Osmar on 2026-10-12. Addendum A records Osmar's framing given with the approval. Addendum B
(fixed values and priors, point 3) and the restated decision in A2 need Osmar's confirmation before any
code. The draft file is kept unchanged.

---


Version 2 (2026-10-12): U holds vaccination and protective behaviour, as Osmar clarified.

Written 2026-10-12, after the official Phase 15 run (`phase15_recalibration`: C1 met in San Diego and Los
Angeles but not San Francisco; C2 not met) and before any Phase 16 code. Osmar chose "Layer 2 forecasts
admissions" on 2026-10-12. Once approved this file is frozen; later changes go to REVIEW_LOG.md.

## 0. Why
* Phase 15 showed where the forecasting value is lost. Given the true admissions, the recalibrated hospital
  model beats ICU ARIMA at about 62 % of origins at 14 days and 67 to 75 % at 28 days; under the real time
  rule, with admissions guessed by persistence and published one week late, it loses to ARIMA. The missing
  piece is a forecast of admissions.
* By Phase 15 section 6, C1 was not met in every county, so Layer 2 targets admissions directly.
* Layer 2 is the transmission and immunity part of the paper's SUEIHCDR model, fitted directly (no neural
  network). Whether a PINN helps is Layer 3.

## 1. Real time information rule (as Phase 15 section 1, extended)
* Admissions (corrected HHS series, `phase15_admissions`): last known week ends t0 minus 7 (primary);
  timely reporting sensitivity: last known week ends t0.
* Cases: USAFacts cumulative cases by report date, known through t0 minus 1. Cases by episode date are not
  used (backfilled).
* Vaccination: CDPH doses by county and administration date, known through t0 minus 7 (assumed one week
  lag; administration date counts are backfilled for a few days).
* Deaths: USAFacts deaths by report date (known through t0 minus 1, as reported, not backfilled) are used
  only for the paper's behaviour signal (section 3). Deaths by date of death are not used (backfilled).
* Variant dates: not scheduled (not known in real time); their effect must be absorbed by the time
  varying transmission and the waning terms.
* Fixed parameters and priors: only values published before the first origin, or before the origin at
  which they are first used (documented with dates, as Phase 15).
* Final data vintage limitation as in Phase 15.

## 2. Question
Under the real time rule, does a transmission and immunity model forecast weekly hospital admissions better
than ARIMA on admissions, and better than a simple statistical forecast that uses cases as a leading signal?

## 3. Model (L2, deterministic, daily, population fractions)
From the SUEIHCDR structure: S susceptible, U protected (as in the paper, U holds both vaccinated people
and people protected by their own behaviour: masks, hygiene, avoiding contacts), E exposed, I infectious,
R recovered (protected by infection). Hospital compartments are not modelled; the admission flux is the
output. The paper's single alpha(t) is kept as the total flow into U but split into its two sources, each
with its own way out, because vaccine protection lasts months while protective behaviour can relax within
weeks:

    alpha(t) S = v(t) S + a_b s(t) S          (vaccination + protective behaviour)
    dS/dt  = -beta(t) S I - v(t) S - a_b s(t) S + omega_v U_v + omega_b (1 - s(t)) U_b + eta R
    dU_v/dt =  v(t) S - omega_v U_v
    dU_b/dt =  a_b s(t) S - omega_b (1 - s(t)) U_b
    dE/dt  =  beta(t) S I - sigma E
    dI/dt =  sigma E - delta I
    dR/dt =  delta I - eta R
    reported cases(t)  = rho * sigma E(t) * N      (as the paper: Cpd = rho sigma E)
    admissions(t)      = ihr * delta I(t) * N      (as the paper's flow (1 - m) delta I into H, ihr = 1 - m)

* v(t): effective vaccination rate from the observed doses (first doses and boosters times a vaccine
  effectiveness against infection, fixed from literature with publication dates).
* s(t) in [0, 1]: the paper's behaviour signal, computed exactly as in the paper code
  (`bayesian_alpha_series` on USAFacts report date deaths, kappa 0.08 and lambda 0.995 as in the paper
  call). Because the paper's signal standardizes deaths by medians of the series it is given, it is
  recomputed at every origin from deaths through t0 minus 1 only; computing it once on the whole series
  would leak future deaths into the past. a_b: the largest
  behavioural protection rate, estimated in the paper's bounds (0 to 0.003 per day). omega_b (1 - s(t)):
  relaxation of behaviour, slower while worry is high, as the paper's om_t = omega (1 - s).
* omega_v and eta: waning of vaccine and infection protection, fixed from literature (the paper's dual
  waning); omega_b with a soft prior.
* Identifiability: protective behaviour enters both through U_b and through beta(t), and an 8 week window
  can hardly separate them. A sensitivity arm without U_b (a_b = 0) shows whether the behaviour
  compartment changes the forecasts; the fitted a_b is reported, not interpreted as a measured quantity.
* beta(t): piecewise linear with weekly knots inside the window and a random walk curvature penalty; held
  at its last value for the forecast.
* rho (case ascertainment) and ihr (admissions per infection) constant within the window, estimated.

## 4. Estimation: moving horizon (causal, sequential)
* At each Saturday origin (2020-10-03 to 2022-12-03, as Phase 15) the model is fitted on the last 56 days:
  daily reported cases (7 day mean, through t0 minus 1) and the known weekly admissions, log scale least
  squares, plus the beta random walk penalty and soft priors on rho and ihr.
* The state at the window start (S, U_v, U_b, E, I, R) is taken from the fit at the previous origin, which used
  only data known then. The first window (starting 2020-08-08) is initialized from cumulative reported
  cases scaled by a real time ascertainment prior (2020 seroprevalence studies) and from zero vaccination.
* Starts: 3, as Phase 15. Fixed values and priors: section 3 and a table to be researched with publication
  dates before any code (point 3 below).

## 5. Forecasts and comparators (weekly admissions, weeks h = 1 to 4 after the last known week)
* **L2:** the fitted model run forward from t0 with beta held at its last value, vaccination held at its
  last known weekly rate and s(t) held at its last known value.
* **L2_NOBEHAV (sensitivity):** the same with a_b = 0 (no behaviour compartment).
* **ARIMA_ADM:** ARIMA on daily admissions (weekly values spread over the days), Phase 10 definition,
  last 150 days of known admissions.
* **CASELAG:** admissions(t) = k * cases7(t - L), k and L (0 to 14 days) estimated on the window, cases
  held at their last known 7 day mean after t0 minus 1.
* **NAIVE_ADM:** last known week held flat.
* All under the primary one week admission delay and under the timely reporting sensitivity.
* **ICU (secondary, reported only):** L2 admissions fed into the Phase 15 recalibrated hospital model
  (same code and settings) give an ICU forecast, compared with ICU ARIMA at 14 days as in Phase 15 C2.

## 6. Metrics and criteria (declared before any fit)
* Error: absolute error of weekly admissions per day, for h = 1, 2, 3, 4; primary metric is the mean over
  h = 1 to 3 (the admissions needed for a 14 day ICU forecast); relative error = error / mean observed.
* **D1 (primary):** in at least 2 of 3 counties, the mean primary error of L2 over origins is lower than
  that of ARIMA_ADM and the block bootstrap 95 % interval of the mean paired difference is below 0 (block
  4 origins, 2000 replicates, fixed seed, as Phase 15).
* **D2 (does mechanism add?):** the same test of L2 against CASELAG; reported, and used in the decision.
* **Decision:**
  * D1 and D2 met: the transmission and immunity layer is accepted for forecasting; Layer 3 compares PINN
    estimation with the direct fit.
  * D1 met, D2 not met: admissions are forecastable from cases, but the mechanism adds nothing over a lead
    indicator; reported as such, and the paper's forecasting claim rests on the simpler method.
  * D1 not met: ARIMA is reported as the better short term admissions forecaster under real time data; the
    model is then evaluated only for scenario use (decided with Osmar).

## 7. Reproducibility
New script `layer2_admissions.py` (numpy, pandas, scipy, statsmodels), manifests with hashes, causality
tests as in Phase 15 (future data replaced by random values must not change any forecast), reduced smoke
tests by me, official runs by Osmar, no overwriting.

## 8. Points for Osmar to confirm
1. Model of section 3: the transmission part of SUEIHCDR, with the paper's alpha(t) split into observed
   vaccination (U_v) and the paper's deaths based behaviour signal (U_b), each with its own waning; the
   L2_NOBEHAV sensitivity; no scheduled variant dates.
2. Moving horizon estimation with the state carried from the previous origin (section 4).
3. Before any code I research the fixed values and priors (incubation and infectious periods, infection
   to admission delay, vaccine effectiveness, waning of vaccine and infection protection, 2020
   ascertainment) with publication dates, and add them to this draft for your approval.
4. Comparators ARIMA_ADM, CASELAG and NAIVE_ADM; primary error over weeks 1 to 3; criteria D1 and D2 and
   the decision rule.

---

## Addendum A (2026-10-12): framing given by Osmar with the approval

A1. Osmar: "the comparison now is just to confirm model works, the story is the physical understanding
that possibilitates scenarios and other things". The paper's contribution is a mechanistic model whose
states and rates mean something physically (immunity, behaviour, transmission, hospital pathway) and can
therefore be used for scenarios. The comparisons with ARIMA, CASELAG and NAIVE are validation, not the
headline claim.
A2. Restated decision (to be confirmed by Osmar). The model is called adequate for scenario use if, in at
least 2 of 3 counties, L2 is not significantly worse than ARIMA_ADM on the primary error, that is, the
block bootstrap 95 % interval of the mean paired difference (L2 minus ARIMA_ADM) includes 0 or lies below
0. D1 (L2 significantly better than ARIMA_ADM) and D2 (against CASELAG) are still reported as stated. If
the model is adequate, Phase 17 designs the scenario analyses; if not, the failure is reported and the next
step is decided with Osmar.

## Addendum B (2026-10-12): fixed values and priors (protocol point 3; researched before any code)

Rule: a value is used only at origins on or after the date it was public (online publication or preprint
posting). Before that date the previous value applies. All rates are per day.

| Quantity | Value | Public from | Source |
|---|---|---|---|
| sigma (E to I) | 1 / 5.1 | 2020-03-10 | Lauer et al., Ann Intern Med 2020;172(9):577 to 582 (median incubation 5.1 days) |
| delta (I out) | 1 / 7 | 2020-02-07 | Wang D et al., JAMA 2020;323(11):1061 to 1069 (median 7.0 days from first symptom to hospital admission); also the paper's prior duration |
| VE against infection, primary series and boosters | 0.95 | 2020-12-10 | Polack et al., N Engl J Med 2020;383:2603 to 2615 (95 % against symptomatic disease 7 days after dose 2) |
| same | 0.90 | 2021-03-29 | Thompson et al., MMWR 2021;70(13):495 to 500 (90 % against infection, full vaccination, US workers) |
| omega_v (vaccine protection waning) | 0 | start | no published US waning estimate yet |
| same | 1 / 191 | 2021-10-04 | Tartof et al., Lancet 2021;398(10309):1407 to 1416, Kaiser Permanente Southern California: VE against infection 88 % in month 1, 47 % at 5 months; exponential decay over 4 months gives a mean of 191 days |
| eta (infection protection waning) | 0 | start | Hansen et al., Lancet 2021;397(10280):1204 to 1212 (80.5 % protection, no waning within 6 months, online 2021-03-17); no earlier estimate of waning existed. Sensitivity: eta = 1 / 365 (the paper's lower bound) |
| rho prior (case ascertainment) | 1 / 12, log sd 0.7 | 2020-07-21 | Havers et al., JAMA Intern Med 2020;180(12):1576 to 1586: infections 6 to 24 times reported cases (geometric middle 12) |
| same | 1 / 4.6, log sd 0.7 | 2020-11-25 | Reese et al., Clin Infect Dis 2021;72(12):e1010 to e1017 (1 in 4.6 infections reported, US, through September 2020) |
| ihr prior (admissions per infection) | 0.026, log sd 0.7 | 2020-05-13 | Salje et al., Science 2020;369(6500):208 to 211 (2.6 % of infections hospitalized, France) |
| a_b (behaviour inflow, maximum) | estimated in [0, 0.003], no prior | | the paper's ALPH_BOUNDS |
| omega_b (behaviour relaxation, at s = 0) | estimated in [1/180, 1/30], soft prior at the geometric middle 1/73.5, log sd 0.5 | | the paper's OMEGA_BOUNDS; no literature value exists, judgment |
| beta(t) smoothness | second difference of log beta between weekly knots, penalty sd 0.1 | | judgment, fixed before any fit |

Other implementation choices fixed now:
* Vaccination flow: v(t) S = VE x (people completing a primary series + booster and bivalent booster
  recipients that day, CDPH `fully_vaccinated` + `booster_recip_count` + `bivalent_booster_recip_count`) / N
  x S / (S + E + I + R), i.e. doses are spread over people not currently in U_v or U_b; doses known through
  t0 minus 7, held at their last weekly mean afterwards.
* First window (origin 2020-10-03, window from 2020-08-08): R = cumulative reported cases at the window
  start x (1 / rho prior) / N; E and I from the 7 day mean of cases at the window start divided by rho
  prior, times 5.1 and 7 days; U_v = U_b = 0; S = 1 minus the rest.
* Later windows: S, U_v, U_b and R carried exactly from the previous origin's fitted trajectory; E and I
  carried with a multiplicative correction estimated in the window (soft prior centred at 1, log sd 0.5), so
  the slow immunity memory is kept and the fast infection states can be corrected by new data.
* Residuals: log(1 + 7 day mean of daily cases) on each window day through t0 minus 1, and log(1 +
  admissions per day) for each known admission week weighted as 7 observations.
* Not represented (stated as limitations): the shorter incubation of Omicron (Jansen et al., MMWR,
  2021-12-31) and the loss of protection against Omicron (immune escape); their effects must be absorbed by
  beta(t), as decided in point 1 (no scheduled variant dates).

## Addendum C (2026-10-12, implementation details fixed before any official run)

Osmar confirmed Addendum A2 and Addendum B on 2026-10-12 ("confirm").
C1. beta(t): log beta knots at window days 0, 7, 14, 21, 28, 35, 42, 49 and 55, linear in between, beta in
[0.01, 3] per day; after day 55 (forecast) beta is held at the last knot.
C2. Integration: classical Runge Kutta 4 with 2 substeps per day; daily fluxes sigma E and delta I integrated
with the Runge Kutta weights. Population conservation is tested.
C3. Parameter bounds: rho in [0.005, 1], ihr in [0.0001, 0.5], a_b and omega_b inside the paper bounds of
Addendum B (logistic transforms), E and I multipliers in [exp(-3), exp(3)].
C4. Starts (3): start 0 is the previous origin's solution with beta flattened to its last knot and the E and
I multipliers reset to 1 (warm start of the moving horizon; a default start at the first origin); starts 1
and 2 are the default start (beta = delta / S, priors at their centres) plus seeded normal noise (sd 0.5).
max_nfev 300.
C5. Case residuals use window days 7 to 56 (the model's 7 day mean needs 7 model days); admission
residuals use the published weeks that lie entirely inside the simulated window.
C6. Chains, each carrying its own state from origin to origin: L2 with delay 1 (primary), L2 with delay 0
(timely reporting), L2_NOBEHAV (a_b = 0, delay 1), L2_ETA365 (eta = 1/365, delay 1). Comparators are
computed for delay 1 and delay 0; the ICU secondary for delay 1.
C7. Behaviour signal: computed at each origin from all USAFacts report date deaths known (from 2020-01-22
through t0 minus 1). The paper computed it on its training window only; with the moving horizon the whole
known history is used. Tested to equal the paper function on the same input.
C8. CASELAG is fitted on the last 8 known admission weeks; ARIMA_ADM on known admissions spread over days,
last 150 days.
C9. ICU secondary: Phase 15 H4 (window 56, 3 starts, 300 evaluations) with the known admission weeks and the
L2 daily admissions after them; compared with the Phase 15 ICU ARIMA at 14 days on the same origin.
C10. Tests (`test_layer2.py`): population conservation and flux balance; behaviour signal equal to the
paper code; causality (cases and deaths after t0 minus 1, doses after t0 minus 7 and unpublished admission
weeks replaced by random values: L2 forecast, carried state and comparators unchanged, delay 1 and 0,
three counties); schedule switch dates. All passed in the cloud. Reduced smoke run (San Diego, 3
origins, 1 start, 30 evaluations): `smoke_tests/smoke_p16_reduced`; timing check (4 origins, official
settings, metrics not inspected) about 3 s per origin for the primary chain with comparators and ICU.
