# Phase 18 protocol (APPROVED by Osmar 2026-10-14; frozen): scenarios from the validated physical states, 2020 to 2021

Approved text = PHASE18_PROTOCOL_DRAFT.md, SHA-256 82d5290a8fce9f2d92881daa5afc5b918c9aaeed4b22d2c209623485347824a0, all four points of section 5
confirmed by Osmar ("confirmed"). On point 4 Osmar agreed to run Phase 18 first and to draft real time
scenario projections afterwards as Phase 18B. Addendum A will record the implementation before any official
run. The draft file is kept unchanged.

---


Written 2026-10-14, after the official Phase 17B run (`phase17b_layer3b`: plausibility, serology and
adequacy met in all three counties). Osmar chose "scenarios in 2020 to 2021" on 2026-10-14: scenarios use
the period in which the states are validated by serology; 2022 is reported as a stated limitation. Once
approved this file is frozen; later changes go to REVIEW_LOG.md.

## 0. Why
* The paper's story is physical understanding that supports scenarios (Phase 16 Addendum A1). Phase 17B
  gives states that agree with the blood donor serosurvey at all 17 dates of 2020 to 2021 in every county,
  with transmission and ascertainment away from their bounds.
* A scenario changes one physical input (vaccination or protective behaviour) and keeps everything else as
  estimated, so the difference is attributable to that input. This is what a forecasting model like ARIMA
  cannot do.
* 2022 is excluded: the model understates infection in late 2022 (Phase 17B, Addendum A6), and the escape
  switch makes January 2022 unstable.

## 1. Reconstruction used by every scenario (no new fitting)
* Source: the official Phase 17B fits, chain L3B delay 1 (primary). Each Saturday origin t0 has a fit on
  the window t0 minus 56 to t0 minus 1; its first week (t0 minus 56 to t0 minus 50) has 7 weeks of later
  data behind it and is the part carried to the next origin. The reconstruction takes from each origin the
  values on its first week: the daily beta(t) from its knots, a_b, omega_b, ihr, rho, VE, omega_v, the
  variant shares and escape values (the Omicron shares are near zero in the scenario periods), and the
  deaths signal s(t) and vaccination doses as that fit used them. The behaviour period needs origins
  2020-11-28 to 2021-05-22 and the vaccination period origins 2021-02-06 to 2022-01-22.
* Baseline S0: one continuous simulation of the L3B equations from the reconstructed state on the first
  day of the scenario period, with these inputs and without the weekly E and I corrections of the moving
  horizon fits (corrections encode observed data and do not belong in a counterfactual). Every scenario is
  the same simulation with one input changed, and is reported as a difference from S0.
* Held fixed in every scenario (stated): beta(t) and s(t) as estimated. There is no feedback from the
  epidemic to contacts or to the behaviour signal, so a scenario that changes infections does not change
  how people behaved. Reported admissions = ihr(t) x delta I x N, as in the fits.

## 2. Scenarios
Periods: vaccination scenarios 2020-12-12 to 2021-11-27 (first doses to the last week before the Omicron
period); behaviour scenarios 2020-10-03 to 2021-03-27 (winter 2020 to 2021 wave).

| Code | Change | Question |
|---|---|---|
| V0 | no vaccination (doses set to 0) | admissions and infections averted by vaccination |
| V1 | per capita doses of the fastest county (San Francisco: 0.27 fully vaccinated per person by 2021-03-31, against 0.20 in Los Angeles and 0.21 in San Diego) | what the pace of the fastest county would have meant elsewhere |
| V2 | observed doses shifted 4 weeks earlier | value of earlier vaccine availability, for a future pandemic |
| B0 | no explicit protective behaviour (a_b = 0 and U_b emptied into S at the start) | contribution of the protective behaviour pool to the winter wave |
| B1 | relaxation of protective behaviour twice as fast (omega_b x 2) | effect of returning to normal sooner |

For San Francisco V1 equals S0 and is not reported. B0 is a lower bound for behaviour, because part of the
behaviour response is also inside beta(t) (Phase 16 L2_NOBEHAV fits beta without U_b).

Outcomes per county and scenario: admissions over the period (total and per 100 000), the highest weekly
admissions, infections over the period, and ever infected at the end; differences from S0 (absolute and in
percent).

## 3. Criteria (declared before any scenario is run)
* **R1, reconstruction of admissions:** S0 total admissions over the scenario period within 20 % of the
  observed total (HHS corrected admissions of Phase 15), and the S0 week of highest admissions within 2
  weeks of the observed one, for each period and county.
* **R2, reconstruction of immunity:** S0 ever infected within [-0.10, +0.20] of the blood donor survey at
  two thirds or more of the survey dates inside the period (same rule as Phase 17B).
* A county and period that meet R1 and R2 have their scenario results reported as primary. If not, the
  failure and the size of the drift are reported, and the scenario numbers for that county and period are
  shown only as secondary.
* **Structural range:** every scenario is repeated on the L3B delay 0 chain and, for V0 to V2, on the
  L3B_NOBEHAV chain; the range across chains is reported next to the primary value. There is no posterior
  distribution (stated limitation).

## 4. Reproducibility
New script `scenarios_phase18.py` that reads `phase17b_layer3b` (params_states.csv, trajectories.csv) and the
same input data, imports `layer3b_sero.py` for the equations, and writes `phase18_scenarios`. Tests
(`test_phase18.py`): the stitched reconstruction with the weekly E and I corrections applied reproduces the
Phase 17B first week states exactly; population conserved in every scenario; V0 gives at least as many
infections as S0 and V1 at most as many in Los Angeles and San Diego; B0 with a_b already 0 equals S0.
Reduced smoke tests by me; official run by Osmar; no new downloads.

## 5. Points for Osmar to confirm
1. Scenarios as retrospective counterfactuals on the Phase 17B reconstruction (first week of each fit,
   continuous simulation without the E and I corrections), with beta(t) and the behaviour signal held as
   estimated.
2. The five scenarios and their periods (V0 no vaccination, V1 fastest county pace, V2 four weeks earlier,
   B0 no protective behaviour pool, B1 faster relaxation).
3. The reconstruction criteria R1 and R2, and the structural range across chains in place of a posterior.
4. Out of scope for this phase: 2022, and real time scenario projections made at an origin with only the
   data known then (possible later phase).

## Addendum A (2026-10-14, implementation, before any official run)

A1. Scripts: `scenarios_phase18.py` (SHA-256 338095ff95fc26fe...) and `test_phase18.py` (b50c34ac395ceaa9...).
They import `layer3b_sero.py` (a3d6e950c3644bd6..., the equations `simulate3` and the input builders) and
`layer2_admissions.py` (4b70ab2b08421e3e...), both unchanged. Inputs: `phase17b_layer3b/params_states.csv`
and `trajectories.csv`, plus the same data files as Phase 17B. Output folder `phase18_scenarios`.
A2. Periods run in whole weeks: vaccination 2020-12-12 to 2021-12-03 (51 weeks), behaviour 2020-10-03 to
2021-04-02 (26 weeks). The inputs of each week are rebuilt by constructing the same problem object as the
official fit at that origin (behaviour signal, doses, variant shares, escape values, VE, omega_v); the
fitted values (beta knots, a_b, omega_b, ihr) come from params_states.csv. The beta knots are stored to 5
decimals, so a week simulated from the fitted state reproduces the Phase 17B states to a relative 1e-6 and
admissions to 1e-5 (observed in the cloud check; the test threshold is 1e-4).
A3. The last weeks of the vaccination period come from origins in January 2022, whose fits already carry
the escape values public by then (f_V from 2021-12-14, f_pre_O1 from 2022-01-06), applied to the variant
shares those fits used for late November and early December 2021. Kept as fitted.
A4. Infections = flow from E to I times N (the quantity CUM accumulates). Ever infected = E + I + R_pre +
R_O1 + R_O2 (Phase 17B measure). R1 uses the observed HHS weeks (Sunday starts) fully inside the period, on
the same grid for model and data; the reported peak of each scenario uses 7 day blocks from the period
start.
A5. V1 uses San Francisco's daily doses per person (fully vaccinated plus boosters, as in the fits); V2 uses
the county's own doses from 28 days later (up to 2021-12-31 at the period end). VE and omega_v stay as each
week's fit used them. B1 doubles omega_b without the fit bound of 1/30 per day. B0 and B1 run on L3B delay 1
and delay 0 only.
A6. Tests (`test_phase18.py`): stitched week reproduces the fitted states and admissions and the rebuilt
inputs match params_states.csv (three weeks, three counties); population conserved in every path; V0 at
least as many infections and admissions as S0; V1 at most as many infections as S0 in Los Angeles and San
Diego; B0 equals S0 when a_b is 0 and U_b is empty. 24 of 24 passed in the cloud; the test prints no
scenario outcome. Reduced smoke run `smoke_tests/smoke_p18_reduced` (Los Angeles, 3 weeks per period, all
chains).
A7. Disclosure. Because there is no fitting, a full run costs seconds. The first cloud smoke run, made
before a truncation option existed, ran the full Los Angeles primary chain and printed its reconstruction
line: S0 over observed admissions 1.13 (vaccination period) and 1.38 (behaviour period), peak 7 days early,
serology 12 of 12 and 6 of 6. This was seen before this addendum was written; nothing in the approved design
(scenarios, periods, criteria) was changed in response, and that output was deleted. Later smoke runs use
`--max-weeks 3`.
