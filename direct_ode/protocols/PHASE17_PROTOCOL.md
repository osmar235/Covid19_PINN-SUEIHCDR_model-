# Phase 17 protocol (APPROVED by Osmar 2026-10-12; frozen): immune escape in real time and an external check of immunity

Approved text = PHASE17_PROTOCOL_DRAFT.md, SHA-256 43ee95d8f47ea86e005373bb62e499cb384bbb8544e0939a4717671b97a61c35, all four points of section 7
confirmed by Osmar ("1. ok 2. confirm 3. confirm 4. done"). Addendum A records what the downloaded files
contain and the resulting adjustments; it needs Osmar's confirmation before any code. The draft file is
kept unchanged.

---


Written 2026-10-12, after the official Phase 16 run (`phase16_layer2`: adequate under A2; the susceptible
pool is exhausted by Omicron, so 2022 waves need transmission at its upper bound). Osmar chose "escape plus
serology check" on 2026-10-12. Once approved this file is frozen; later changes go to REVIEW_LOG.md.

## 0. Why
* The paper's story is physical understanding that supports scenarios (Phase 16 Addendum A1). In Phase 16 the
  forecasts were adequate but the 2022 states were not physical: S at about 0.02 in San Diego and Los
  Angeles in January 2022, beta at its upper bound in most Los Angeles origins of 2022, ascertainment at 1.
* The missing mechanism is immune escape: Omicron (BA.1 and BA.2) and later BA.4 and BA.5 reinfected people
  protected by earlier infection or vaccination.
* Physical states can be checked against data the model never sees: serosurveys of infection induced
  antibodies. This check is what makes the states credible enough for scenarios.

## 1. Real time rule (as Phase 16, plus)
* Variant shares: CDPH statewide sequencing (`covid19_variants.csv`), known for specimens collected up to
  t0 minus 14 days (the reporting lag stated by CDPH). Later shares are extrapolated by logistic growth
  fitted to the last 3 known weeks of each variant group, capped to sum to 1. The same statewide shares are
  used for the three counties (stated as a limitation).
* Escape magnitudes enter at the date they became public (table in section 3); before that date the
  forecaster does not know them and uses no escape for that variant group.
* Serosurvey data are never used in fitting or forecasting; they are only compared with the model
  afterwards.

## 2. Model change (L3 = L2 of Phase 16 plus leaky protection by variant group)
Variant groups: pre (all lineages before Omicron), O1 (Omicron BA.1, BA.2 and their sublineages before BA.4
and BA.5), O2 (BA.4, BA.5 and later). Shares p_pre(t), p_O1(t), p_O2(t).
* The recovered pool is split by the variant group of the infection: R_pre, R_O1, R_O2 (flow delta I split
  by the current shares).
* Infection pressure on a protected pool k: beta I x sum over groups g of p_g(t) x f(k, g) x pool_k. Those
  infected go to E. f(k, g) is the loss of protection of pool k against group g relative to full protection
  (0 = fully protected, as in Phase 16; f is applied only from the date its source became public).
* U_b (protective behaviour) is not affected by escape. Everything else as Phase 16 (vaccination, behaviour
  signal, waning, eta = 0, moving horizon with the state carried, 56 day window, priors and schedules of
  Phase 16 Addendum B, same comparators).

## 3. Escape values (relative: f = 1 minus protection against the new group / protection against earlier
variants, with earlier protection 0.90)

| Pool, group | f | Public from | Source |
|---|---|---|---|
| R_pre against O1 | 1 - 0.56/0.90 = 0.38 | 2022-01-06 | Altarawneh et al., medRxiv 10.1101/2022.01.05.22268782 (N Engl J Med 2022, 2022-02-09): prior infection protection 90 % Alpha, 92 % Delta, 56 % Omicron |
| R_pre against O2 | 1 - 0.283/0.90 = 0.69 | 2022-07-12 | Altarawneh et al., medRxiv 10.1101/2022.07.11.22277448: pre Omicron infection 28.3 % against BA.4 or BA.5 reinfection |
| R_O1 against O2 | 1 - 0.797/0.90 = 0.11 | 2022-07-12 | same: BA.1 or BA.2 infection 79.7 % against BA.4 or BA.5 |
| R_O1 and R_O2 against O1; R_O2 against O2 | 0 | | same group or later group |
| U_v against O1 and O2 | w_b x 0.18 + (1 - w_b) x 0.45 | 2021-12-14 | Andrews et al., medRxiv 10.1101/2021.12.14.21267615 (N Engl J Med 2022;386:1532 to 1546): booster 75.5 % against Omicron vs 92.6 % against Delta (f = 0.18); two doses after 25 weeks about 35 % vs 63.5 % (f = 0.45). w_b = booster recipients / fully vaccinated, as known on t0 from CDPH |

No published estimate of additional vaccine escape for O2 existed during the study period in the form
needed; U_v uses the O1 value for O2 (judgment, stated).

## 4. Criteria (declared before any fit)
* **Physical plausibility (new, primary):** for L3 with delay 1, in every county, beta at its upper bound
  in at most 10 % of origins and rho at a bound in at most 10 % of origins (Phase 16 L2, Los Angeles 2022:
  43 of 53 and 23 of 53).
* **External check of immunity (new, primary):** the model's share of the population ever infected, 1 minus
  exp(minus cumulative infections per person) at each survey date, compared with infection induced
  (anti nucleocapsid) seroprevalence for California from the CDC Nationwide Commercial Laboratory
  Seroprevalence Survey (and the blood donor survey where a county or metropolitan estimate exists).
  Serology is a lower bound (antibodies wane, anti nucleocapsid is less often detected after vaccination).
  Criterion: at least 2 of 3 counties have the model within 0.10 (absolute) of the survey estimate or above
  it but within 0.20, at two thirds or more of the survey dates. Results for the L2 states of Phase 16 are
  reported next to L3 on the same dates.
* **Adequacy (as Phase 16 A2):** L3 not significantly worse than ARIMA_ADM in at least 2 of 3 counties.
* D1 and D2 of Phase 16 and the ICU secondary are reported again.
* **Decision:** if plausibility, the external check and adequacy are met, Phase 18 designs the scenarios. If
  not, the failures are reported and the next step is decided with Osmar.

## 5. Data Osmar downloads into data_epi/ (before any code)
1. CDPH variant data: https://data.chhs.ca.gov/dataset/52e4aa7a-2ea3-4bfd-8cd6-7d653db1ee74/resource/d7f9acfa-b113-4cbc-9abc-91e707efc08a/download/covid19_variants.csv
2. CDC commercial laboratory seroprevalence: https://data.cdc.gov/api/views/d2tw-32xv/rows.csv?accessType=DOWNLOAD
3. CDC blood donor seroprevalence 2020 to 2021, infection induced: https://data.cdc.gov/api/views/mtc3-kq6r/rows.csv?accessType=DOWNLOAD

## 6. Reproducibility
New script `layer3_escape.py` reusing the Phase 16 code (imports `layer2_admissions.py`); causality tests
extended to variant shares (shares after t0 minus 14 replaced by random values must not change any
forecast); reduced smoke tests by me; official run by Osmar; no overwriting.

## 7. Points for Osmar to confirm
1. Escape as leaky protection by variant group, with R split by the group that caused the infection, and
   statewide variant shares with the 14 day lag.
2. The escape values of section 3 and their dates, including the judgments (relative to 0.90; U_v for O2
   as for O1; no escape for behaviour).
3. The two new primary criteria (plausibility and the serology check) and their thresholds.
4. The three downloads of section 5.

---

## Addendum A (2026-10-12, after inspecting the downloads, before any code; to be confirmed by Osmar)

A1. Variant groups. `covid19_variants.csv` (CDPH, California statewide, 2021-01-01 to 2023-11-30) has a
single "Omicron" category, so BA.1 and BA.2 cannot be separated from BA.4 and BA.5 with it. Adjustment: the
Omicron share comes from CDPH as approved (14 day lag); the split of Omicron into O1 (BA.1, BA.2 and their
sublineages) and O2 (BA.4, BA.5 and later) comes from the CDC SARS-CoV-2 Variant Proportions dataset
(data.cdc.gov jr58-6ysp), HHS Region 9 (California about three quarters of the population), weighted
empirical estimates, using at each origin only the rows with published_date on or before t0 (the estimates
as they were published at the time). Before the first published BA.4 or BA.5 estimate the O2 share is 0.
A2. Serology. The blood donor survey (2020 to 2021, infection induced) has regional estimates: "Northern
California, Bay Region" (CA-1), used for San Francisco, and "Southern California Region" (CA-3), used for
Los Angeles and San Diego; monthly from 2020-07 to 2021-12. The commercial laboratory survey gives
California statewide only; its all ages anti nucleocapsid estimate exists for rounds 1 to 30 (to early
2022; later rounds report children only); it is compared with all three counties. For 2022 the blood donor
2022 to 2023 survey (data.cdc.gov ar8q-3jhn) is requested; if it has California infection induced
estimates they are added to the check. The criterion of section 4 is applied to every available survey
date per county, pooling the sources; results are also reported per source. Both surveys are lower bounds
(antibody waning; donors are adults; the regions are larger than the counties).
A3. Additional downloads requested from Osmar (data_epi/):
* https://data.cdc.gov/resource/jr58-6ysp.csv?usa_or_hhsregion=9&$limit=2000000 (Region 9 rows only)
* https://data.cdc.gov/api/views/ar8q-3jhn/rows.csv?accessType=DOWNLOAD

Osmar confirmed Addendum A and downloaded the two extra files on 2026-10-12 ("confirm both"):
data_epi/jr58-6ysp.csv (SHA-256 c39cd056..., Region 9 rows; publication date column is `creation_date`;
weighted estimates published from 2021-05-04, first BA.4 and BA.5 estimates published 2022-06-07) and
data_epi/2022_2023_Nationwide_Blood_Donor_Seroprevalence_Survey_Combined_Infection-_and_Vaccination-
Induced_Seroprevalence_Estimates.csv (SHA-256 36c67b8a...; California quarterly "Past infection with or
without vaccination" for 2022 and 2023).

## Addendum B (2026-10-12, implementation, before any official run)

B1 (needs Osmar's confirmation). Serology measure. With eta = 0 nobody returns from R to S, U_v or U_b, so
the model's share ever infected is exactly R_pre + R_O1 + R_O2 + E + I (= 1 - S - U_v - U_b, tested). The
protocol formula 1 - exp(-cumulative infections) assumes reinfection at random and is biased low: with 0.5
cumulative infections and no reinfection it gives 0.39 instead of 0.50. Proposed: the exact compartment
measure is primary for the criterion; the protocol formula is computed and reported next to it.
B2. Variant groups in the CDC data: O1 = B.1.1.529, BA.1, BA.1.1, BA.2, BA.2.12.1, BA.3; pre = names
starting with A., AY., B.1 (except B.1.1.529), P., R.; "Other" is excluded; every other lineage is O2 (BA.4,
BA.5 and later, including BA.2.75, BN.1, BQ.1 and XBB).
B3. Shares: CDPH 7 day average Omicron percentage (daily percentage where the average is missing), known up
to t0 minus 14; CDC: for each week the latest estimate created on or before t0; O2 fraction of Omicron =
O2 / (O1 + O2). Extrapolation beyond the last known value: logistic growth through the last 3 weekly points
when all lie in (0.001, 0.999), otherwise the last value held.
B4. The recovered flow delta I is split by the current shares. Early isolated CDPH Omicron values (for
example 1.6 % on 2021-01-01, one specimen) move a small share of 2021 recoveries into R_O1; this is left as
it was known at the time (no cleaning with hindsight) and has no effect before escape values are public.
B5. w_b = cumulative booster recipients / cumulative fully vaccinated (CDPH), known at t0 minus 7.
B6. Model value at a survey date: the state on that date from the fit at the first origin on or after it.
Survey dates: midpoint of the collection range (commercial laboratory), median donation date (blood donor
2020 to 2021), mid quarter (blood donor 2022); only dates inside the modelled period. The Phase 16 L2
comparison uses `phase16_layer2/trajectories.csv` with the exact measure R + E + I.
B7. Chains: L3 delay 1 (primary), L3 delay 0, L3_NOBEHAV delay 1; eta = 0. Comparators and ICU secondary as
Phase 16. Plausibility counts origins with the forecast beta (last knot) at 3.0 and rho at 0.005 or 1.
B8. Tests (`test_layer3.py`): population conservation with all leaks; cumulative infections; ever infected
identity; shares sum to 1; causality including variant data (CDPH after t0 minus 14 and CDC rows created
after t0 replaced by random values) at 2022-01-08 and 2022-07-02 in three counties; escape switch dates. All
passed. Before any escape value is public, L3 reproduces L2 exactly (identical fit costs in the timing
check). Reduced smoke run `smoke_tests/smoke_p17_reduced` (San Diego, 3 origins, 1 start, 30 evaluations).
