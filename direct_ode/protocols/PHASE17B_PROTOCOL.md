# Phase 17B protocol (APPROVED by Osmar 2026-10-13; frozen): ascertainment calibrated by serology in real time

Approved text = PHASE17B_PROTOCOL_DRAFT.md, SHA-256 9613d277846bb5553c0dda9867cd007c8a3584765173726211e371e924eddba3, all four points of section 4
confirmed by Osmar ("confirm"), including point 4 (the exact compartment measure, Phase 17 Addendum B1,
now confirmed). Addendum A records the implementation; it changes nothing in the approved design. The draft
file is kept unchanged.

---


Written 2026-10-13, after the official Phase 17 run (`phase17_layer3`) and before any Phase 17B code. Osmar
chose "fix Los Angeles first" on 2026-10-13. Once approved this file is frozen; later changes go to
REVIEW_LOG.md.

## 0. Why (diagnosis from the Phase 17 outputs, no new fits)
* Los Angeles fails the serology check at every date, by +0.24 to +0.48. The error is present at the very
  start: on 2020-08-12 the model has 0.285 of Los Angeles ever infected against 0.044 in the Southern
  California blood donor survey. The first window starts from cumulative reported cases times 12 (Havers
  2020, which described spring 2020 testing), and Los Angeles had twice the cumulative cases per person of
  San Diego in August 2020 (0.022 against 0.010), so its error is largest. San Diego and San Francisco start
  about +0.09 and +0.11 too high.
* The gap then grows during the 2020 to 2021 winter (Los Angeles +0.30 to +0.43). Cases and admissions fix
  only the ratio rho / ihr; the absolute number of infections is set mainly by the priors. Serology is the
  data that fixes it.
* Serosurveys were published while the epidemic was happening, so a county could have used them.

## 1. Change (L3B = Phase 17 L3 plus serology calibrated ascertainment)
* Input: the CDC commercial laboratory survey for California (all ages, anti nucleocapsid, rounds 1 to 30).
  A round is known 6 weeks after the end of its collection period (judgment; the exact publication dates
  are not in the file).
* At each origin, with the latest known round k: rho_sero = (California cumulative reported cases per
  person at the midpoint of round k) / (seroprevalence of round k). California cases are the sum of the
  USAFacts county series.
* rho_sero replaces the Havers and Reese schedule as the centre of the rho prior once a round is known
  (log sd 0.3, judgment: a measured California value, tighter than the 0.7 used for literature values from
  other places). After round 30, the last all ages round, it is held.
* The first window (origin 2020-10-03) is initialized with rho_sero from the rounds known then: R = county
  cumulative cases / rho_sero / N, and E and I from the case rate / rho_sero, instead of the factor 12.
* Everything else as Phase 17 L3 (escape, variant shares, real time rule, chains, comparators, ICU
  secondary).
* Stated limitation: anti nucleocapsid antibodies wane and are detected less often after vaccination, so
  rho_sero tends to be too high (infections too low). The validation surveys share this bias.

## 2. Validation (the commercial survey is now an input, so it is no longer used to check)
* Serology check on the blood donor surveys only: Southern California for Los Angeles and San Diego, Bay
  Region for San Francisco (monthly, 2020-07 to 2021-12), and California for 2022 (quarterly). Same measure
  and thresholds as Phase 17 (exact ever infected; within -0.10 to +0.20 of the survey at two thirds of the
  dates).
* **Criteria:** (1) serology met in all three counties (the point of this phase is Los Angeles);
  (2) plausibility as Phase 17 (forecast beta at its upper bound and rho at a bound in at most 10 % of
  origins); (3) adequacy as Phase 17 (not significantly worse than ARIMA_ADM in at least 2 of 3 counties).
* Reported: forecasts against Phase 16 L2 and Phase 17 L3, states and rates, both serology measures.
* **Decision:** all three met: Phase 18 designs scenarios for the three counties. Serology met in San Diego
  and San Francisco but not Los Angeles: scenarios for those two only, and Los Angeles is reported as a
  failure. Otherwise the result is reported and the next step is decided with Osmar.

## 3. Reproducibility
New script `layer3b_sero.py` reusing `layer3_escape.py` and `layer2_admissions.py`; causality test extended
to serology (rounds not yet known at t0 replaced by random values must not change any forecast); reduced
smoke tests by me; official run by Osmar.

## 4. Points for Osmar to confirm
1. Serology calibrated ascertainment from the commercial survey (California), known 6 weeks after collection.
2. rho prior centred on rho_sero with log sd 0.3, and the first window initialized with it.
3. Validation on the blood donor surveys only, and the three criteria with serology required in all
   three counties.
4. The exact compartment measure of ever infected for the serology check (Phase 17 Addendum B1, not yet
   explicitly confirmed; in Phase 17 both measures gave the same pass or fail in every county).

## Addendum A (2026-10-13, implementation, before any official run)

A1. Script `layer3b_sero.py` (SHA-256 a3d6e950c3644bd6...) is a copy of `layer3_escape.py` (Phase 17,
40955512891558c3...) with only the changes listed in its header: rho_sero, the rho prior, the first window,
the validation rule. It imports `layer2_admissions.py` (4b70ab2b08421e3e..., unchanged). Chains: L3B delay 1
(primary), L3B delay 0, L3B_NOBEHAV delay 1; eta = 0; comparators and ICU secondary as Phase 17.
A2. California cases: sum over all USAFacts rows with State CA, including the statewide unallocated row,
divided by the summed California population of the USAFacts population file. Cases are known through t0
minus 1; the round midpoints are always earlier than the day the round becomes known, so no later case data
are used.
A3. Rounds: the column "Anti-N, All Ages Cumulative Prevalence, Rounds 1-30 only" for Site CA (30 rounds);
midpoint = middle of the collection range; known from end of collection + 42 days. rho_sero is clipped to
[0.005, 1] (the bounds of rho). The first round is known on 2020-09-16, before the first origin
(2020-10-03, round 2 known, rho_sero 0.397), so the Phase 16 schedule fallback in the code is never used in
the official run. Values by round when it becomes known (input data, not a result): 0.27 to 0.51, mostly
0.34 to 0.48; round 30 (known 2022-04-04) gives 0.374, held to the last origin.
A4. Prior term: the rho term among the prior residuals is replaced by (log rho - log rho_sero) / 0.3; all
other prior terms unchanged.
A5. Output: `serology_check.csv` labels every survey row as "validation" (blood donor: blood_donor_CA-3 for
Los Angeles and San Diego, blood_donor_CA-1 for San Francisco, blood_donor_2022_CA) or "input (not a
check)" (commercial survey), and reports the Phase 17 L3 and Phase 16 L2 exact measures on the same dates.
`criteria.json` adds the rho_sero and round used at every origin. Serology criterion: exact measure,
validation rows only, met in all three counties.
A6. Stated limitation added: the last all ages round ends 2022-02-21; for the rest of 2022, when home
testing grew, rho_sero is held, which probably overstates ascertainment (understates infections) in late
2022. No change with hindsight is made.
A7. Tests (`test_layer3b.py`, SHA-256 45257b25f6cbe311...): 30 rounds read; latest known round chosen;
rho_sero formula; no round known before 2020-09-16; rho prior term with sd 0.3; causality at 2021-01-09 and
2022-01-08 in three counties (cases and deaths after t0 minus 1, doses and boosters after t0 minus 7,
unpublished admission weeks, CDPH after t0 minus 14, CDC rows created after t0, serosurvey rounds not yet
known at t0, California cases after t0 minus 1, all replaced by random values; forecasts, carried state and
rho_sero unchanged). 11 of 11 passed in the cloud. Reduced smoke run `smoke_tests/smoke_p17b_reduced` (Los
Angeles, 3 origins, 1 start, 30 evaluations): runs end to end; checks the code path only, not a result.
