# Covid19_PINN-SUEIHCDR_model-

Physics-Informed Neural Networks (PINNs) implementation of an extended SUEIHCDR compartmental model for multi-wave COVID-19 forecasting with dual waning immunity.

This repository accompanies the manuscript:

**Generalizable Multi-Wave COVID-19 Forecasting via Physics-Informed Neural Networks with Dual Waning Immunity**

## Overview

This codebase supports:

- PINN-based mechanistic model fitting
- Multi-city forecasting workflows
- Uncertainty analysis across repeated runs
- Ablation experiments
- Publication tables and figures

The repository is organized as a lightweight Python package with top-level entrypoints for the main workflows.

## Repository structure

```text
Covid19_PINN-SUEIHCDR_model-/
├── requirements.txt
├── requirements-optional.txt
├── README.md
├── .gitignore
├── covid_pinn_workflow.ipynb
├── run_publication.py
├── run_ablation.py
├── run_uncertainty.py
└── covid_pinn/
    ├── __init__.py
    ├── core.py
    ├── uncertainty.py
    ├── runner.py
    ├── ablation_runs.py
    ├── publication_analysis.py
    ├── publication_figures.py
    ├── master_loader.py
    ├── revision_helpers.py
    └── stats_summary.py
```

## Installation

Create a fresh environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional baselines (Prophet is heavy and pulls in `cmdstanpy`; only needed if you want the Prophet column in the comparison tables):

```bash
pip install -r requirements-optional.txt
```

## Data paths

The package modules read input data and write generated outputs relative to a single root directory. This directory is expected to contain (or to have written to it) the per-city output folders (`outputs_SUEIHCDR_PUBLICATION_v3_{CITY}/`), per-city uncertainty folders (`parameter_uncertainty_{CITY}/`), and the regime metrics directory (`Resultados_Cidades_02152026/`).

### Input CSV files

The core data loader expects the following CSVs to be resolvable at runtime:

- `covid_county_population_usafacts.csv`
- `covid_confirmed_usafacts.csv`
- `covid_deaths_usafacts.csv`
- `2020_US_Region_Mobility_Report.csv`, `2021_...`, `2022_...`
- `index.csv`, `epidemiology.csv`, `mobility.csv`

These files are not bundled with the repository because of size; download them from the original sources (USAFacts, Google COVID-19 Open Data / Community Mobility Reports) and place them either in the repository root or in a folder named `data/` at the repository root.

### Automatic path resolution

`covid_pinn.core` searches several sensible locations for each CSV, in order:

1. The current working directory
2. The repository root
3. A `data/` subfolder at the repository root
4. Sibling folders commonly used during development

You only need to set `PINN_DATA_PATH` explicitly if your data is stored somewhere outside those locations:

```bash
export PINN_DATA_PATH=/path/to/data            # macOS / Linux
set PINN_DATA_PATH=C:\path\to\data             # Windows cmd
$env:PINN_DATA_PATH="C:\path\to\data"          # Windows PowerShell
```

If `PINN_DATA_PATH` is unset, the modules fall back to the search order above.

## Main entrypoints

1. Uncertainty analysis

```bash
python run_uncertainty.py
```

2. Ablation analysis

```bash
python run_ablation.py
```

3. Publication analysis

```bash
python run_publication.py
```

### Running individual stages as modules

Each package module is also runnable directly via `python -m`, useful for re-running a single stage without invoking the full top-level wrapper:

```bash
python -m covid_pinn.revision_helpers      # build per-city master tables
python -m covid_pinn.master_loader         # per-horizon stats + boxplots/bar figures
python -m covid_pinn.stats_summary         # augmented metrics + Holm-corrected p-values
python -m covid_pinn.publication_figures   # publication-quality figure regeneration
```

## Notebook

The notebook below is kept as a workflow / reference notebook that walks through the same steps interactively:

```text
covid_pinn_workflow.ipynb
```

## Package modules

- `covid_pinn/core.py` — core PINN model and training logic
- `covid_pinn/uncertainty.py` — uncertainty analysis workflow (entry: `main()`)
- `covid_pinn/runner.py` — run logic for city / model workflows
- `covid_pinn/ablation_runs.py` — ablation experiment routines (entry: `run_recommended_ablation()`)
- `covid_pinn/publication_analysis.py` — manuscript summary analyses (entry: `main()`)
- `covid_pinn/publication_figures.py` — figure generation utilities (entry: `generate_all_figures()`)
- `covid_pinn/master_loader.py` — loading and aggregation helpers (entry: `run_master_analysis()`)
- `covid_pinn/revision_helpers.py` — manuscript / revision support utilities (entry: `build_master_tables()`)
- `covid_pinn/stats_summary.py` — statistical summary helpers (entry: `run_full_summary()`)

All modules are import-side-effect free: importing any of them performs no file I/O and prints nothing.

## Data and outputs

This repository does not bundle large raw datasets or generated results folders by default. Some workflows expect local input files and/or previously generated output directories. Before running the full analyses, set `PINN_DATA_PATH` (see above) and verify your input data is in the expected layout.

## Reproducibility notes

For the manuscript analyses, reproducibility depends on:

- the Python package versions in `requirements.txt`
- access to the expected input data (set via `PINN_DATA_PATH`)
- consistent directory structure for generated outputs

## Code availability

Repository: https://github.com/osmar235/Covid19_PINN-SUEIHCDR_model-

## Citation

If you use this code, please cite the associated manuscript:

> Pinto Neto, O. et al. *Generalizable Multi-Wave COVID-19 Forecasting via Physics-Informed Neural Networks with Dual Waning Immunity.* (manuscript under review).

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

Osmar Pinto Neto — arena235research@gmail.com
