# Correction C1 (2026-10-24, before submission): ascertainment prior of L2 and L3

Declared before any corrected run. Osmar asked for the error to be fixed rather than disclosed.

## 1. The error
Phase 16 Addendum B (2026-10-12) fixed the case ascertainment prior centre for L2 (and so for L3, Phase 17) as
1/12 from 2020-07-21 (Havers et al., JAMA Intern Med 2020;180(12):1576-1586; infections 6 to 24 times reported
cases, geometric middle 12) and 1/4.6 from 2020-11-25, attributed to Reese et al., Clin Infect Dis
2021;72(12):e1010-e1017 (online 2020-11-25). The Reese et al. full text, checked on 2026-10-24, gives "1 in 7.7, or
13%, of total infections were identified and reported" through 30 September 2020. It does not contain 4.6. The value
1/4.6 therefore has no verified source available on 2020-11-25.

## 2. The correction
* The prior centre from 2020-11-25 becomes 1/7.7, the value of the cited study, on the same date. The log standard
  deviation (0.7), the 1/12 value before that date, and every other setting of Phases 16 and 17 are unchanged.
* `layer2_admissions.py` keeps the old schedule under `--rho-schedule protocol_v1`, so the 2026-10-12 outputs remain
  reproducible. `layer3_escape.py` takes the same option. The original code is archived by its SHA-256
  (`code_archive/layer2_admissions_v4b70ab2b.py`, `code_archive/layer3_escape_v40955512.py`).
* New outputs go to new folders (`phase16_layer2_c1`, `phase17_layer3_c1`); the original folders stay unchanged.

## 3. What is and is not affected
* Affected: the L2 and L3 fits and everything computed from them: Table S4 rows L2 and L3, Figure S1, the
  Phase 16 and 17 criteria, the L3 survey comparison (Results, abstract values 0.168, 0.376 and 0.140, Figure 2,
  Table S10, the graphical abstract).
* Not affected: L3B and L3T. Their prior is centred on the commercial serosurvey at every origin (checked in the
  saved Phase 17B parameters: rho_prior equals rho_sero at all 1026 rows), and their initial state at the first
  origin also uses the serosurvey value. Phase 17B reads the L2 and L3 trajectories only to add comparison columns to
  `serology_check.csv`. Phases 18, 18B and 19 use L3B or L3T fits only. None of them is rerun.
* The L2 and L3 comparison columns of the Phase 17B serology check are rebuilt from the corrected trajectories by
  `serology_compare_c1.py`, using the same rule as `layer3b_sero.py`. Rebuilding from the original trajectories
  reproduces the original file byte for byte (checked in the cloud, line endings aside); the command list repeats
  this check on Osmar's computer.

## 4. Reporting
Whatever the corrected results are, they replace the L2 and L3 values in the manuscript. The historical decisions of
Phases 16 and 17 are reported as made; if a corrected criterion would have changed, that is stated in REVIEW_LOG and
in the revision notes. No other setting is tuned. The optional environment check (San Diego primary L2 chain rerun
with protocol_v1) shows whether this computer reproduces the original fits exactly, which separates the effect of the
correction from optimizer level noise.
