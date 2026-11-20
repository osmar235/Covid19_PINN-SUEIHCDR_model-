Version 1.1.0, Correction C1 (prepared 25 October 2026)

The case ascertainment prior of the comparison versions L2 and L3 used 1/4.6 from 25 November 2020, a value
attributed to Reese et al. (2021) that the cited study does not report; the study reports that 1 in 7.7
infections were identified. L2 and L3 were refitted with 1/7.7 (phase16_layer2_c1, phase17_layer3_c1), and the
L2 and L3 comparison columns of the Phase 17B serology check were rebuilt (phase17b_serology_c1). The primary
L3B model, L3T and every reconstruction and projection are unchanged. The superseded folders are kept under
provenance/superseded_before_c1; layer2_admissions.py --rho-schedule protocol_v1 reproduces them. See
protocols/PHASE16_17_CORRECTION_C1.md and verification/c1_compare.txt. In the optional refitting commands
below, the L2 and L3 outputs are named phase16_layer2_c1 and phase17_layer3_c1 by default.

COVID19 direct ODE reproducibility release
Version 1.1.0 prepared 25 October 2026 (1.0.0 prepared 18 October 2026)

Companion study
From serological calibration to scenario credibility: retrospective evaluation
of a multiwave COVID-19 model
Osmar Pinto Neto and Devan R Romero

This release contains the direct ODE workflow reported in the manuscript.
It is distinct from the earlier PINN workflow. The release is locally staged;
no public DOI or repository release URL has yet been assigned. Do not cite the
old PINN repository as the reproduction source for this study.

What is included

- Exact fitting and simulation source used by the current workflow.
- The original Phase 18 scenario source, recovered by its recorded SHA-256,
  plus the later backward-compatible source used for Phase 19 and this replay.
- Frozen aggregate public surveillance snapshots, including the full HHS
  facility input, and corrected county admissions series.
- Saved fits and result tables for L2, L3, L3B and L3T, reconstruction and
  origin-specific scenarios; fixed scientific assumptions and dated protocols.
- Portable paper-table extraction, calendar re-scoring, tolerance summaries
  and plotting scripts; all 32 reference tables and ten reference figures.
- Original run manifests, explicit provenance notes and a fresh replay report.

Quick reproduction

Use Python 3.14.3 and a new virtual environment for matching the recorded
scientific runtime. From this release directory run:

  python -m venv .venv
  .venv\Scripts\activate
  python -m pip install -r requirements-model.txt
  python -B reproduce.py --out replay/my_clean_replay

On Linux/macOS activate the environment with source .venv/bin/activate instead.
The scripts use pathlib and create their outputs inside the release directory.
Each replay must use a fresh empty output directory. Frozen input and result
folders are never overwritten. No network access is needed after dependencies
are installed, because the exact surveillance snapshots are included.

The replay checks raw input hashes, regenerates corrected admissions from the
HHS snapshot, reruns all saved-fit P0-P4 and realized-vaccination projections,
reproduces L3B and L3T reconstructions, executes the original 18 projection
checks, and regenerates all 32 paper CSV tables. It compares CSV/JSON content
at relative and absolute tolerance 1e-10, also recording byte identity.
Runtime-dependent manifests and logs are not expected to be byte-identical.

The 18 October clean replay produced byte-identical content for all 21 compared
admissions/scenario result files. All 18 projection checks passed; all 32 paper
tables agreed numerically. See verification/replay_verification.json.
This reruns simulations from saved fitted states. It does not repeat estimation,
constitute an untouched validation, or resolve biological uncertainty.

Paper figures

Install requirements-figures.txt in a separate environment if desired:

  python -m pip install -r requirements-figures.txt
  python -B analysis/build_evidence.py
  python -B analysis/rescore_saved_results.py
  python -B analysis/extra_figures.py

The last command generates all ten figures as SVG, embedded-font PDF and PNG.
Do not run make_figures.py as the final step: extra_figures.py replaces Figure 5
with the calendar-based comparison and adds Figures S3 and S4.
Arial was used for the reference figures. On systems without Arial, the script
looks for Liberation Sans or DejaVu Sans. Set FIGURE_FONT_REGULAR and
FIGURE_FONT_BOLD to installed TrueType files to control this explicitly.
Changing fonts can change layout and pixels while retaining numerical content.
All ten PNG figures were reproduced byte for byte with Arial on the recorded
Windows host. The data visualizations are deterministic plots of archived data;
they were not synthesized with an image-generation model. AI code assistance
is disclosed in the manuscript. The model/evaluation schematic is explanatory.

Optional complete refitting

Full estimation was not rerun for release verification. These commands expose
the original fitting workflow without overwriting the archived fitted results.
Expect substantially longer runs than the saved-fit replay.

  python -B layer2_admissions.py --out replay/refits/layer2
  python -B layer3_escape.py --phase16 replay/refits/layer2/trajectories.csv --out replay/refits/layer3
  python -B layer3b_sero.py --phase16 replay/refits/layer2/trajectories.csv --phase17 replay/refits/layer3/trajectories.csv --out replay/refits/layer3b
  python -B layer3t_tests.py --phase16 replay/refits/layer2/trajectories.csv --phase17 replay/refits/layer3/trajectories.csv --phase17b replay/refits/layer3b/trajectories.csv --out replay/refits/layer3t

The historical window dates, fitting starts, constraints and fixed assumptions
are retained. Cross-platform solvers may yield small numerical differences;
the release does not promise that all re-estimated parameters will match bitwise.

Source and historical provenance

The reported phase18b_projections_v2 run has traceable source, thirteen recorded
inputs and seven recorded outputs. Its code/dependency/input/output hashes
match the archived files. Earlier phase18b_projections outputs are retained
under provenance/earlier_phase18b for historical transparency and are not inputs
to the manuscript results. Their matching projection source and full change
rationale remain unrecovered. The v2 overwrite option alone does not establish
the prior contents of that directory. A new successful replay verifies the
reported version; it does not recover the missing earlier development history.

The original Phase 18 scenario source was recovered separately by matching its
recorded hash. The root scenarios_phase18.py includes the later Phase 19 options;
its defaults reproduce the Phase 18 results byte for byte in the clean replay.
See provenance/recovered_sources.json and PROVENANCE.txt.

Data and licenses

All input datasets are public aggregated surveillance or survey data; no
individual patient records are included. Snapshot names, sizes and SHA-256
values are recorded in provenance/input_inventory.json. Historical retrieval
dates not present in the source records are not invented. Final-vintage data
and imposed availability lags do not recreate a real-time revision archive.

The code retains the repository's existing MIT license in LICENSE. Original
provider terms govern the included third-party data; the MIT license does not
relicense those data or the manuscript. See DATA_SOURCES_AND_TERMS.txt.

SHA256SUMS.txt and RELEASE_MANIFEST.json cover every distributed file other than
those two self-referential manifest files. verify_archive.py verifies the hashes.
CITATION.cff and zenodo_metadata_draft.json are supplied for the public release;
add the actual persistent identifier after deposition.

Scientific limits

The same counties contributed to adaptive development. Diagnostic tolerances
are pragmatic. Scenario ranges are deterministic, have no nominal probability,
and represent selected input perturbations. Biological parameter/path uncertainty,
external validation and revised clinical delay/severity mechanisms remain open.

Correspondence: osmar@csusm.edu

Graphical abstract
The optional separate graphical abstract is prepared from archived numeric data
with ReportLab. After extracting the data, run python -B analysis/graphical_abstract.py.
Reference outputs and the caption are in reference_graphical_abstract/.
