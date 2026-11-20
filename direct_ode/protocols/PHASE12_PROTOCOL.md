# Phase 12 protocol (APPROVED by Osmar 2026-10-11): epidemiology first, starting with a data audit

Written 2026-10-09. Once approved this file is frozen; later changes go to REVIEW_LOG.md with date and
reason.

## 0. Decision and scope
* Osmar, 2026-10-09: pause estimation work; first get the epidemiology right, then ask whether a PINN
  helps estimate it. Everything in the model is open to revision, including the U compartment and the
  dual waning structure.
* Phase 10 stands as reported (PINN failed Stage 1 and Stage 2 in San Diego, Los Angeles, San Francisco).
  The Phase 11 draft (equation constrained fit of the unchanged model) is on hold.

## 1. Plan of the epidemiology track (each layer gets its own protocol after the previous one)
* **Layer 1, clinical pathway.** From an input of new hospital admissions (or lagged cases if admissions
  are not usable), can a compartmental hospital model (ward, ICU, deaths in ward and ICU, discharge)
  reproduce observed ward census, ICU census and deaths in all three counties and all periods, with plain
  equation fitting (no neural network)?
* **Layer 2, transmission and immunity.** Susceptibility from vaccination and prior infection, waning,
  variants, a reporting fraction that may change over time; fitted with plain equation fitting.
* **Layer 3, joint model and estimation method.** Connect the layers; only then compare estimation methods
  (direct fit versus PINN) and decide what the PINN contributes.

## 2. Phase 12: data audit (no fitting)
Script `epi_data_audit.py`; inputs, outputs and hashes recorded in `audit_manifest.json`.
* Counties: San Diego, Los Angeles, San Francisco. Period 2020-03-29 (start of the CHHS hospital data) to
  2022-12-31.
* Core data (already in the project): USAFacts cases and deaths (report date), CHHS hospital census.
* Additional data requested (downloaded by Osmar, see `PHASE12_WINDOWS_COMMANDS.txt`):
  1. CDPH "COVID-19 Time-Series Metrics by County and State" (archived): cases by episode date and deaths
     by date of death, i.e. epidemiological timing instead of report dates.
  2. CDPH "Statewide COVID-19 Vaccines Administered By County" (and its archived booster version):
     cumulative primary series and booster counts by county of residence.
  3. HHS "COVID-19 Reported Patient Impact and Hospital Capacity by Facility" (healthdata.gov anag-cw7u,
     weekly, from 2020-07-31, no longer updated after 2024-05-03): confirmed COVID admissions per facility,
     summed here to county by FIPS code. Values 1 to 3 are suppressed (-999999); the number of suppressed
     facility weeks is reported.
  Columns of the additional files are detected by name; the script stops with the list of available columns
  if a needed one is missing, and saves its choices in `column_choices.json`.

## 3. Pre declared periods
P1 FirstWave 2020-03-29 to 2020-09-30; P2 Winter20 2020-10-01 to 2021-03-31; P3 Delta 2021-04-01 to
2021-11-30; P4 Omicron 2021-12-01 to 2022-04-30; P5 BA2 and BA5 2022-05-01 to 2022-12-31.

## 4. Questions and how each answer is used (decided before seeing the official audit)
Weekly ratios are ratios of weekly means. "Material change" means the period median differs by a factor
of at least 1.5 between any two periods in at least 2 of the 3 counties.

| Question | Measure | Consequence for the model |
|---|---|---|
| Q1 Do deaths occur outside the ICU? | deaths per day per occupied ICU bed versus the range the current model allows (0.017 to 0.13) | If outside the range in at least 10 % of weeks in at least 2 periods in at least 2 counties: add deaths from the ward (and test deaths without hospital admission) |
| Q2 Did ICU severity change? | ICU share of hospitalized patients per period | Material change: ICU fraction depends on time, variant or immunity, not a constant |
| Q3 Did hospitalization per case change? | hospital census or admissions per lagged case | Material change: infection severity and/or reporting fraction must change over time; with Q6 decide which |
| Q4 Did length of stay change? | census divided by admissions per day (only with HHS admissions) | Material change: discharge rates depend on period |
| Q5 Timing | lag maximizing the correlation cases to hospital census and hospital census to deaths | Sets the delay structure (incubation, infectious period, hospital stay) the model must reproduce |
| Q6 Immunity | vaccination coverage and cumulative reported cases as fractions of the population | Susceptibility must include vaccination and unreported infection when coverage is material |
| Q7 Data quality | suspected share of hospital patients; report date versus date of death | Decides which death and case series the fits use |

## 5. Not decided by this audit
Parameter values, loss weights and the estimation method. The audit describes data only; Layer 1 gets its
own protocol, with fit criteria declared before any fit.

## 6. Points for Osmar to confirm
1. The layered plan (clinical pathway first, then transmission and immunity, then estimation method).
2. The three additional data sources.
3. The periods in section 3 and the decision rules in section 4.
4. Counties: San Diego, Los Angeles, San Francisco.

---

## Addendum A (2026-10-10, implementation details fixed before the official audit run; approved with the protocol)

A1. Files: CDPH `statewide-covid-19-cases-deaths-tests.csv` (current version of the time series; columns
date, area, cases, deaths; deaths by date of death per its data dictionary; rows dated "None" are dropped);
CDPH `statewide-covid-19-vaccines-administered-by-county-booster-metric-archived.csv`; HHS
`COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility.csv` (columns collection_week, fips_code,
previous_day_admission_adult_covid_confirmed_7_day_sum, previous_day_admission_pediatric_covid_confirmed_7_day_sum).
A2. With the CDPH file, all ratios use cases by episode date and deaths by date of death; USAFacts series are
kept in the weekly tables and in the lag table for comparison.
A3. HHS admissions: weekly sums over the county's facilities; a week in which no facility reported is
missing (not zero); weeks are used only if the number of reporting facilities is at least 90 % of the
county's median over the study period; suppressed values (1 to 3) count as 0 in the primary series and as 2
in a sensitivity series (`adm_per_day_supp2`).
A4. Admissions per lagged case use cases one week earlier; hospital census per lagged case uses the lag that
maximizes the correlation of cases (by episode date) with hospital census.
