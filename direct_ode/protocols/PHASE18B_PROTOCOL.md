# Phase 18B protocol (FROZEN, approved by Osmar): real time scenario projections, 2020 to 2021

Written 2026-10-14, after Phase 19 (testing driven ascertainment rejected as the explanation of the winter
drift; L3B of Phase 17B stays the base model). Osmar chose "accept drift, go to 18B" on 2026-10-14: the drift
is a known limitation, behaviour scenarios of Phase 18 stay secondary, and the real time projections are
designed with the drift in mind. Approved and frozen 2026-10-15.

## 0. Why
* Osmar's framing: the model should be useful to a city in a future epidemic, with only the data it has at
  the time. Phase 18 answered "what if" questions with hindsight; Phase 18B asks what a county could have been
  told on each Saturday, using only that day's fit.
* Known limitation going in: during growth the model's transmission, run forward without the weekly
  corrections, overshoots (Phase 18; Phase 17B forecasts 1 to 4 weeks ahead overshoot by about 10 to 55 %
  during growth). Projections are therefore limited to 8 weeks and the drift is measured, not hidden.

## 1. Design (no new fitting)
* Fits: the official Phase 17B L3B delay 1 fits (`phase17b_layer3b`). For each origin t0 the projection
  starts from that fit's state on t0 and uses only what that fit knew: its last beta knot, rho, ihr, a_b,
  omega_b, the behaviour signal held at its last known value, doses held at the last known 7 day pace,
  variant shares and escape values as known on t0 (all exactly as the Phase 17B forecast, extended).
* Origins: every Saturday from 2020-12-12 to 2021-10-02 (43 origins per county), so every 8 week projection
  ends by 2021-11-27, inside the period validated by serology and before the escape switch.
* Horizon: 8 weeks of daily admissions, reported as weekly means on the observed HHS weeks.
* Scenarios posed at each origin (the kind of questions a health department asked at the time):

| Code | Assumption from t0 on |
|---|---|
| P0 | status quo: transmission and vaccination pace as on t0 |
| P1 | transmission 20 % lower from t0 + 7 days (stronger measures, one week to take effect) |
| P2 | transmission 20 % higher from t0 + 7 days (relaxation) |
| P3 | vaccination pace 1.5 times the pace on t0 (origins with doses only) |
| P4 | vaccination pace half the pace on t0 (origins with doses only) |

* Outputs per origin and scenario: weekly admissions for weeks 1 to 8, total admissions over 8 weeks, peak
  week, infections; differences from P0.

## 2. Evaluation (declared before any projection is run)
* **E1, envelope coverage (criterion):** share of (origin, week) pairs where the observed weekly admissions
  fall between the lowest and highest of P0 to P4. Criterion: at least 50 % for weeks 1 to 4 and reported for
  weeks 5 to 8, in at least 2 of 3 counties.
* **E2, status quo drift (reported):** median log(P0 / observed) by week ahead, separately for growth and
  decline weeks; compared with a naive projection (last observed week held).
* **E3, hindsight vaccination (reported):** P0 rerun with the doses that were actually given (known only
  later), to separate the vaccination input from the transmission assumption.
* **Decision:** E1 met: real time scenario projections are reported in the paper as a demonstration of
  decision support, with E2 as the stated limit of their reach. E1 not met: they are reported as a limitation
  of the approach, and the paper relies on the hindsight scenarios of Phase 18.

## 3. Reproducibility
New script `projections_phase18b.py` reading `phase17b_layer3b` and the same data, rebuilding each origin's
inputs with the Phase 17B problem object (as `scenarios_phase18.py` does) and simulating with `simulate3`;
tests: P0 reproduces the Phase 17B forecast for weeks 1 to 4 exactly; scenario changes start on the stated
day; population conserved; P1 at most and P2 at least the admissions of P0.

---

## Addendum A: Implementation details (2026-10-15)
1. **Forecast horizon & grid:** At each Saturday origin $t_0 \in [\text{2020-12-12}, \text{2021-10-02}]$ (43 origins per county), the 8-week projection covers 56 days $[t_0, t_0 + 55\text{ d}]$, partitioned into 8 whole weeks $h \in \{1, \dots, 8\}$ where week $h$ spans $[t_0 + 7(h-1), t_0 + 7h - 1]$.
2. **Causal state and inputs on $t_0$:** Extracted from `phase17b_layer3b/params_states.csv` and `trajectories.csv` for chain `L3B` (`delay=1` primary; `delay=0` and `L3B_NOBEHAV` `delay=1` included as secondary sensitivity chains). For weeks 1 to 8, status quo `P0` holds $\beta(t) = \beta_{\text{last}}$, behaviour signal $s(t) = s_{\text{last}}$, and daily vaccination flow $v(t) = v_{\text{last}}$ (the causal 7-day pace known at $t_0$ with the 1-week lag). During this period (ending 2021-11-27), escape knowledge is not yet public (Altarawneh/Andrews preprints are December 2021 / January 2022), so pre-Omicron single-pool immunity holds causally at every origin.
3. **Scenario modifiers:**
   * `P1` ($\beta \times 0.8$ after 7 days): $\beta(t) = \beta_{\text{last}}$ for $t \in [t_0, t_0 + 6\text{ d}]$, and $0.8\,\beta_{\text{last}}$ for $t \in [t_0 + 7\text{ d}, t_0 + 55\text{ d}]$.
   * `P2` ($\beta \times 1.2$ after 7 days): $\beta(t) = \beta_{\text{last}}$ for $t \in [t_0, t_0 + 6\text{ d}]$, and $1.2\,\beta_{\text{last}}$ for $t \in [t_0 + 7\text{ d}, t_0 + 55\text{ d}]$.
   * `P3` ($1.5\times$ vaccination pace) and `P4` ($0.5\times$ vaccination pace): daily vaccination flow into $U_v$ scaled by 1.5 or 0.5 over $[t_0, t_0 + 55\text{ d}]$, capped so cumulative protected inflows never exceed $S(t)$. At origins where causal known doses on $t_0$ are zero ($v_{\text{last}} = 0$), `P3` and `P4` equal `P0` and are flagged `has_doses = False` in the weekly table.
   * `P0_HINDSIGHT` (`E3`): identical to `P0` except daily vaccination flow over $[t_0, t_0 + 55\text{ d}]$ uses the realized county doses actually administered on those dates (multiplied by the contemporary schedule $\text{VE}$ active at $t_0$).
4. **Growth vs. decline classification for E2:** An origin-week $(t_0, h)$ is classified two ways in `drift_e2.csv`: (a) *ex-ante (causal)* based on whether the last two causally known observed admission weeks at $t_0$ were rising ($\text{obs}(t_0 - 1\text{w}) > \text{obs}(t_0 - 2\text{w})$), and (b) *realized* based on whether observed admissions at week $h$ exceeded observed admissions at $t_0$ ($\text{obs}_h > \text{obs}_{t_0}$). Naive comparator holds the last causally known admission week ($\text{obs}_{t_0-\text{delay}}$) constant across $h = 1 \dots 8$.