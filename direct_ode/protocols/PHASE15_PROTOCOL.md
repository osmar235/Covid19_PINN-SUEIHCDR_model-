# Phase 15 protocol (APPROVED by Osmar 2026-10-11; frozen): real time recalibration of the hospital pathway

Approved text = PHASE15_PROTOCOL_DRAFT.md (version 2), SHA-256
fc0d66435a69281b956c5c843d1022cbc1502315591cd40e91920d142d344da5, approved with all five points of
section 9 accepted as proposed ("agree, we can continue now"). The draft file is kept unchanged.
Addenda A and B were written before any official Phase 15 run.

---


Written 2026-10-11, after the official Phase 14 run (`phase14_layer1b`, H4 failed the pre declared rule)
and before any Phase 15 code. Osmar chose "causal recalibration" on 2026-10-11. Version 2 (same day) follows
Osmar's instruction that the design must reproduce what a county health department could actually know
while the epidemic was happening, so that the method is usable in a similar situation in the future.
Once approved this file is frozen; later changes go to REVIEW_LOG.md with date and reason.

## 0. Why
* Phase 14 failure stands and is reported as such: rates held constant over whole periods cannot follow
  the data (census per admission changes about 2.5 fold inside the Delta period), and the in hospital death
  fraction h is not identified from these data.
* The same H4 with priors gave useful conditional projections from 4 origins (ICU 28 day relMAE <= 0.20 at
  8 of 12). A forecaster does not need one set of rates per period; it needs rates that describe the
  recent weeks before the origin. Phase 15 asks that question directly, at many origins.
* This is a new question with new criteria, declared before any fit. It is not a rerun of Phase 14 with
  relaxed thresholds.

## 1. Real time information rule (governs every choice below)
A forecast issued on day t0 may use only information a county health department had on t0.
* **Census (CHHS ward and ICU, confirmed):** known through t0 minus 1 day (published with a one day lag;
  assumed, conservative).
* **Admissions (HHS, weeks Sunday to Saturday):** primary assumption, the week ending on t0 is not yet
  published, so the last known week ends on t0 minus 7 days (one week publication delay; assumed,
  conservative, because the release schedule of the facility file could not be confirmed). Admissions after
  the last known week are unknown and are filled by the admission projection rule (section 5), also inside
  the fitting window.
* **Deaths:** not used (deaths by date of death are backfilled for weeks, and Phase 14 showed h is not
  identified).
* **Literature priors:** only values published before the first origin (2020-10-03) are used, one fixed set
  for all origins (section 4). The Phase 14 table cannot be used: its values come from papers published in
  2022 and 2023 that describe periods the forecaster has not yet seen.
* **Variant periods:** the Phase 12 period boundaries were set in hindsight; they are used only to report
  results by period, never inside the forecaster.
* **Data vintage:** all series are the final vintage; earlier vintages are not available. Revisions of the
  census are expected to be small; this is stated as a limitation. ARIMA and NAIVE face the same rule.
* **Timely reporting scenario (sensitivity):** the same forecasts with no admission delay (the week ending
  on t0 known). The difference shows what faster hospital reporting would add, which is the relevant
  question for a future epidemic.

## 2. Questions
* **Q1 (pathway):** with H4 recalibrated on a trailing window and the admissions that actually occurred
  after the origin, does H4 reproduce ward and ICU census over 28 days? This tests the hospital pathway on
  which Layer 2 would be built. It uses future admissions, so it is never presented as a forecast.
* **Q2 (forecast):** under the real time rule of section 1, is H4 a better ICU census forecaster than
  ARIMA on the ICU census?

## 3. Origins and windows
* Counties: San Diego, Los Angeles, San Francisco.
* Origins t0: every Saturday from 2020-10-03 to 2022-12-03 (114 per county). The first origin is the first
  with 8 complete known admission weeks (admissions start 2020-08-02); the last is the last with 28 days of
  truth before 2022-12-31.
* Horizon k = 1 to 28 days after the last known census day, so day k is t0 minus 1 plus k.
* Trailing window: the 56 days of census ending on t0 minus 1 (primary). Sensitivity: 28 and 84 days; the
  84 day sensitivity starts at 2020-10-31 and is compared with the primary on the same origins.

## 4. Model, priors and estimation
* Structure H4 exactly as Phase 14 (ward W, ICU C, post ICU ward P, direct ICU fraction q, no cycling), one
  set of 7 rates per origin (L_w, p_wc, p_wd, L_c, p_cd, L_p, q). Exact daily solution (matrix exponential).
* **Real time priors** (one set, all origins, sd 0.5 on the log scale because they are early, partly
  foreign, and applied across variants):

| Quantity per admission | Prior | Source (available before 2020-10-03) |
|---|---|---|
| Mean hospital stay (days) | 7.0 | Rees et al., medRxiv 17 May 2020 (BMC Medicine Sep 2020): outside China median 5, IQR 3 to 9 |
| Probability of ICU admission | 0.32 | Kim et al., COVID-NET, medRxiv 22 May 2020 (Clin Infect Dis, online Jul 2020): 2491 US adults, Mar to May 2020 |
| In hospital death | 0.17 | Kim et al., same cohort |
| Mean ICU stay (days) | 9.3 | Rees et al.: outside China median 7, IQR 4 to 11 |
| Death among ICU patients | 0.416 | Armstrong, Kane, Cook, Anaesthesia, online 15 Jul 2020 (meta analysis, completed ICU admissions) |

  Means are obtained from the reported median and IQR assuming a lognormal stay distribution:
  sigma = (ln Q3 minus ln Q1) / 1.349, mean = median x exp(sigma^2 / 2). These values describe the first
  wave and are expected to be too severe later; the trailing window recalibration is what adapts them, and
  the prior z of the fitted values is reported at every origin.
* Fit: least squares on log(1 + census) for daily ward and ICU census over the trailing window, plus the
  priors as penalties ((log value minus log prior) / 0.5)^2. Bounds as Phase 14 (stays 1 to 60 days, q at
  most 0.9). Initial state at the window start: observed census, W:P split from the steady state of the
  candidate rates (as Phase 14). Starts: 3 (the Phase 14 fixed start and 2 seeded random starts); best
  objective kept.
* State at the last known census day: W:P split from the fitted trajectory, scaled to the observed ward
  census; C set to the observed ICU census (as Phase 14).

## 5. Admission projection and forecasts (28 days, daily)
* **H4_PERSIST (primary forecast):** admissions after the last known week held at that week's mean.
* **H4_TREND (secondary forecast):** log linear growth from the last two known weeks,
  r = ln(A_last / A_prev) / 7, with |r| capped at ln(2) / 7 (doubling or halving per week).
* **H4_ORACLE (Q1 only):** observed admissions throughout the window and the projection.
* Each of H4_PERSIST and H4_TREND is run under the primary one week delay and under the timely reporting
  sensitivity.
* **Comparators:** ARIMA on the ICU census and on the ward census, Phase 10 definition (orders p, d, q in
  {0, 1} chosen by AIC, statsmodels SARIMAX), fitted on the 150 days of census ending on t0 minus 1 (the
  median of the Phase 10 lookbacks, fixed for all origins); NAIVE (last known census held flat).

## 6. Metrics and criteria (declared before any fit)
* Error: MAE over days k = 1 to h, h = 7, 14 and 28; relative MAE = MAE / mean observed over the same days.
* **C1 (Q1, pathway):** H4_ORACLE median over origins of the 28 day relative MAE <= 0.20 for ICU and for
  ward, in every county.
* **C2 (Q2, forecast), primary horizon 14 days, primary delay assumption:** in at least 2 of 3 counties,
  the mean ICU MAE of H4_PERSIST over origins is lower than that of ARIMA, and the 95 % interval of the
  mean paired difference (H4_PERSIST minus ARIMA) excludes 0. Interval: moving block bootstrap over
  origins, block length 4 origins (overlapping horizons make neighbouring origins dependent), 2000
  replicates, fixed seed.
* Reported for every county and model: errors at 7, 14 and 28 days, win fraction against ARIMA and NAIVE,
  results by Phase 12 period (reporting only), the delay and window sensitivities, fitted rates over time
  and their prior z values.
* **Decision:**
  * C1 met: the recalibrated H4 is accepted as the hospital pathway, and Layer 2 (transmission and
    immunity) is drafted with admissions as its link to the hospital layer, under the same real time rule.
  * C1 not met: the hospital pathway is not modelled mechanistically with these data; Layer 2 targets
    admissions or census directly, decided with Osmar.
  * C2 is reported as the forecasting result whatever its outcome; if it fails, ARIMA is reported as the
    better short term ICU forecaster. The timely reporting sensitivity is reported next to it.

## 7. Reproducibility
New script `layer1c_recalibration.py` (numpy, pandas, scipy, statsmodels), manifests with code and data
hashes, smoke tests by me on reduced settings (one county, a few origins, one start), so results are first
seen in Osmar's official run. No overwriting of output folders.

## 8. References for the real time priors
* Rees EM, Nightingale ES, Jafari Y, et al. COVID-19 length of hospital stay: a systematic review and data
  synthesis. medRxiv 2020, doi:10.1101/2020.04.30.20084780 (v3 posted 17 May 2020); BMC Medicine 2020
  (September).
* Kim L, Garg S, O'Halloran A, et al. Risk factors for intensive care unit admission and in-hospital
  mortality among hospitalized adults identified through the US Coronavirus Disease 2019
  (COVID-19)-Associated Hospitalization Surveillance Network (COVID-NET). Clin Infect Dis
  2021;72(9):e206 to e214 (online July 2020; preprint medRxiv doi:10.1101/2020.05.18.20103390, 22 May 2020).
* Armstrong RA, Kane AD, Cook TM. Outcomes from intensive care in patients with COVID-19: a systematic
  review and meta-analysis of observational studies. Anaesthesia 2020;75(10):1340 to 1349,
  doi:10.1111/anae.15201, online 15 July 2020 (volume and pages to be checked once more before the
  official run).

## 9. Points for Osmar to confirm
1. The real time information rule of section 1, including the one day census lag and the one week
   admission delay as the primary assumptions, and the timely reporting scenario as a sensitivity.
2. The real time prior set of section 4 (2020 literature only, sd 0.5) replacing the Phase 14 table.
3. Origins every Saturday, trailing window 56 days (28 and 84 as sensitivity).
4. H4_PERSIST as the primary forecast and H4_TREND as secondary.
5. Criteria C1 and C2, including the 14 day primary horizon and the block bootstrap.

---

## Addendum A (2026-10-11, implementation details fixed before any official run)

A1. Each origin and arm has 7 raw parameters: log L_w, logit p_wc, logit of the share of non ICU ward exits
that die (p_wd = (1 minus p_wc) times that share), log L_c, logit p_cd, log L_p, logit q / 0.9; stays in
[1, 60] days; logits bounded at plus or minus 8 (Phase 14 Addendum A1 without h). max_nfev 300.
A2. Starts: start 0 is the Phase 14 fixed start; starts 1 and 2 are uniform inside the bounds, seeded by
(20261011, start, county index, origin ordinal day, window, arm).
A3. Prior values are those of the section 4 table (7.0, 0.32, 0.17, 9.3, 0.416). The lognormal formula
gives 6.97 and 9.27 for the two stays; the rounded table values are used.
A4. Day convention: census(d + 1) = Phi census(d) + Gam A(d), so the projection for day L + k uses
admissions up to day L + k minus 1. (Phase 14 projections used the admissions of the projected day, a one
day shift that matters only at week boundaries; Phase 14 decisions were based on in sample criteria.)
A5. If the census on t0 minus 1 is missing, the last day with both ward and ICU census is used as the
projection start and for NAIVE and ARIMA; the horizon days stay t0 to t0 plus 27. The window starts at its
first day with both values; missing days inside the window are skipped in the residuals.
A6. With delay 0 the window needs no admission fill, so one fit serves H4_PERSIST and H4_TREND (if the
last known week is missing, the PERSIST fill is used inside the window and TREND only after it).
A7. ARIMA: ICU and ward separately, census of the last 150 calendar days through t0 minus 1 with missing
days dropped, Phase 10 orders and fitting options.
A8. Outputs: forecasts_daily.csv, metrics_origin.csv, params.csv, summary.csv, summary_by_period.csv,
summary_common_origins_from_2020-10-31.csv (window sensitivity on common origins), criteria.json,
run_manifest.json, run_log.txt.
A9. Tests (`test_layer1c.py`): one simulated admission reproduces the analytic mean stay, death probability
and ICU days; causality, by replacing all census after t0 minus 1 and all unpublished admission weeks with
random values and checking that every forecast and comparator is unchanged (delay 1 and delay 0, three
counties); the PERSIST and TREND formulas. All passed. Smoke run in the cloud with reduced settings
(San Diego, 3 origins, 1 start, 20 evaluations) and a timing check on 2 origins with official settings
(about 2 s per origin and county; metrics not inspected).

## Addendum B (2026-10-11, admissions series corrected before any official run)

B1. Bug found while testing Phase 15. In `epi_data_audit.py` (Phase 12, PHASE12 Addendum A3) a facility
counted as reporting only if its adult admission value was a number. Suppressed values (1 to 3
admissions) are reports, not missing reports. In low incidence weeks many hospitals are suppressed, so
those weeks failed the 90 % completeness rule and were set to missing. Evidence, San Diego, weeks of
2021-05-23 to 2021-07-18: all 19 hospitals reported every week (8 to 9 suppressed in June), observed
admissions 2 to 15 per day, yet the weeks were missing and Phases 13 and 14 filled them by interpolation
(about 20 per day in June 2021).
B2. Fix: `hhs_admissions_v2.py` counts a hospital as reporting if its adult value is a number or the
suppression flag; a week is usable if at least 90 % of the county's usual number of reporting hospitals
report (median over 2020-08-02 to 2022-12-31: 19, 84 and 10, a number a county knows). Admission values are
unchanged (suppressed = 0 primary, = 2 in adm_per_day_supp2). Device smoke run
(`smoke_tests/smoke_p15_admissions`): values identical to Phase 12 where both exist (largest difference
1.4e-14); missing weeks 2020-08 to 2022-12 fall from 24, 17, 36 to 3, 0, 0 (San Diego, Los Angeles, San
Francisco; the three San Diego weeks are 2021-05-09, with 15 of 19 hospitals in the file, and the last two
weeks of 2022).
B3. Phase 15 uses the corrected series from Osmar's official run of `hhs_admissions_v2.py`
(`--audit phase15_admissions`, the script default).
B4. Consequences for earlier phases: the Phase 12 ratios that use admissions simply lost those weeks;
Phases 13 and 14 used interpolated admissions in them. The Phase 14 overprediction of census in June 2021
is consistent with this. Phase 14 is rerun unchanged except for the admissions folder
(`--audit phase15_admissions`, output `phase14_layer1b_admfix`) as a documented correction; both runs are
reported and the Phase 14 decision rule is applied to the corrected run.
