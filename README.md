# COVID-19 SUEIHCDR model and reproducibility materials

This repository now includes the direct ordinary differential equation (ODE)
workflow supporting **From serological calibration to scenario credibility:
retrospective evaluation of a multiwave COVID-19 model**, by Osmar Pinto Neto
and Devan R Romero. The paper is being prepared for submission.

The current paper uses direct ODE integration and penalized least squares.
The earlier PINN package remains available for historical work; it is not the
estimation workflow used for this retrospective case study.

## Current study release

- [Version ode-v1.2.0](https://github.com/osmar235/Covid19_PINN-SUEIHCDR_model-/releases/tag/ode-v1.2.0)
- [Complete frozen input/code/result archive](https://github.com/osmar235/Covid19_PINN-SUEIHCDR_model-/releases/download/ode-v1.2.0/COVID19_ODE_reproducibility_v1.2.0.zip)
- [Exact workflow and commands](direct_ode/README.txt)
- [Scientific provenance](direct_ode/PROVENANCE.txt)
- [Correction C1 protocol](direct_ode/protocols/PHASE16_17_CORRECTION_C1.md)

The release includes the corrected L2/L3 ascertainment prior, saved fits,
county admissions, origin-specific scenarios, all 32 paper tables, ten figures,
their source scripts, and a separate projection implementation with its tests.
The primary L3B model, testing sensitivity L3T, and reconstruction/projection
results were unaffected by Correction C1. Manuscript drafts are not distributed
in this software release.

## Reproduce the reported calculations

The simplest route is to download and extract the full archive linked above.
It includes the exact frozen aggregate inputs, including the large HHS file.
After installing dependencies, the calculations require no live data downloads.

Alternatively, after cloning this repository:

```bash
cd direct_ode
python fetch_frozen_inputs.py
python verify_archive.py
python -m venv .venv
```

Use Python 3.14.3 to match the scientific environment. Activate `.venv` with
`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS:

```bash
python -m pip install -r requirements-model.txt
python -B reproduce.py --out replay/my_clean_replay
```

Use a fresh output directory. This regenerates admissions, projections and
reconstructions from saved fits and checks all 32 paper tables. Full optional
refitting and separate plotting-environment commands are in `direct_ode/README.txt`.
Installing plotting requirements into the model environment is not required.

The 481 MB raw HHS CSV exceeds GitHub's individual Git-file limit. Only that
file is omitted from the Git tree; `fetch_frozen_inputs.py` retrieves it from
the versioned release ZIP and verifies its frozen SHA-256. Every other release
file is present under `direct_ode/`, with line-ending conversion disabled.

## Verification and interpretation

A fresh saved-fit replay on 20 November 2026 reproduced **21 result files byte
for byte**, verified **32 paper tables**, and passed **18 projection checks**.
The separate projection implementation passed **48 checks**; matching calendar
dates gave identical predictions on **6,186 primary-chain scenario rows**.
The single historical grid-shift exception is recorded without imputation.

The separate implementation shares the fitted states, input modules and ODE
integrator. These checks establish computational consistency conditional on
the archived fits and assumptions. The study is retrospective, uses counties
involved in model development, and does not establish prospective validation,
forecast superiority, or independently validated causal vaccination effects.
The five prescribed scenario paths are deterministic, without an assigned
probability level. The unrecovered earlier projection-source limitation is
retained in the provenance statement.

## Earlier PINN work

The existing `covid_pinn/` package, notebook, entrypoints and April 2026 release
are retained. Their original instructions are in
[the historical PINN README](docs/legacy_PINN_README.md).

## Citation and licenses

Use the versioned release URL and `CITATION.cff` when referring to these
research materials. No Zenodo DOI has been assigned. Software retains the
existing MIT license. Third-party data remain subject to provider terms;
see `direct_ode/DATA_SOURCES_AND_TERMS.txt`. The software license does not
relicense third-party data or the manuscript.

Correspondence: osmar@csusm.edu
