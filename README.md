# Covid19_PINN-SUEIHCDR_model

Physics-Informed Neural Networks (PINNs) implementation of the **SUEIHCDR** compartmental model for COVID-19 dynamics. Includes Jupyter notebooks and Python code for model fitting, identifiability analysis, forecasting, rolling-origin evaluation, and figure generation using public epidemiological and mobility data.

> **Paper context:** This repository supports the experiments and figures reported in the associated manuscript (Nature Communications submission).

---

## Repository contents

Recommended structure:

```text
.
├── notebooks/
│   └── covid_PINN_versaoPaper_02282026.ipynb
├── src/                      # optional (exported scripts)
│   ├── run_cities.py
│   ├── parameter_uncertainty.py
│   ├── master_loader_all_cities.py
│   └── publication_analysis.py
├── data/                     # optional (raw + intermediate datasets)
├── results/                  # generated outputs (figures, tables, fitted params)
└── README.md
```

If you currently only have notebooks, you can run everything from Jupyter.  
If you prefer command-line execution, export the relevant notebook cells into `.py` scripts under `src/`.

---

## Requirements

- Python **3.9+** recommended  
- PyTorch (CPU or GPU)
- Core scientific stack: `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `scikit-learn`
- Baseline forecasting libraries (if used): `prophet`, `tbats`

Example install:

```bash
pip install -U numpy pandas scipy matplotlib scikit-learn statsmodels
pip install -U torch
pip install -U prophet tbats
```

> Note: `prophet` may require additional system dependencies on some platforms. If installation fails, consider using `conda` (conda-forge) or follow Prophet’s official installation instructions.

---

## Data

The workflow expects public epidemiological time series and (optionally) mobility/distancing covariates consistent with the manuscript setup.

The notebook includes local path configuration blocks. Before running, **edit the path variables** to point to your local folders, e.g.:

- `ROOT`
- `BASE_DIR`
- `DATA_DIR`
- `RESULTS_DIR`

Search inside the notebook/scripts for lines like:

```python
ROOT = Path(r"...")
BASE_DIR = Path(r"...")
RESULTS_DIR = ROOT / "Resultados_Cidades_02152026"
```

and update them for your system (Mac/Linux/Windows).

---

## Quick start

### 1) Multi-city PINN runner (fit-check vs full)

The runner supports:
- `fit_check`: trains the model and exports outputs (fast)
- `full`: trains + runs multi-window evaluation (slow)

Example commands (if exported to `src/`):

```bash
python src/run_cities.py --mode fit_check
python src/run_cities.py --mode fit_check --cities "Seattle,London,Rome"

python src/run_cities.py --mode full --cities "Seattle,London"
python src/run_cities.py --skip-us        # world cities only
python src/run_cities.py --skip-world     # US cities only
```

Outputs are written into the configured results directory (see the path variables).

---

## Parameter uncertainty (repeated fits)

Runs multiple fits per city (no multi-window evaluation) and saves outputs such as:
- `parameter_uncertainty_results.csv`
- `parameter_summary_for_table1.csv`
- `variant_multiplier_results.csv`
- `variant_multiplier_summary.csv`

Key configuration typically includes:
- `UNC_N_RUNS` (e.g., 10–20)
- `UNC_BASESEED`

---

## Master loader (combine cities into master tables)

Aggregates outputs across cities and generates master tables, typically under:
- `.../MASTER_TABLES/`

Includes helper utilities for daily-from-cumulative conversions and consolidated summaries.

---

## Publication analysis (figures + stats)

Generates paper figures and statistical comparisons (e.g., paired tests and summary metrics).  
Typical output folder:
- `publication_figures/`

Make sure this is set in your analysis script/notebook:

```python
DATA_DIR = Path("...")  # update for your machine
```

---

## Reproducibility notes

- Training and uncertainty runs are seed-controlled.
- Results may vary slightly depending on hardware (CPU vs GPU), PyTorch version, and floating-point nondeterminism.
- For stronger reproducibility, pin versions in a `requirements.txt` or `environment.yml`.

---

## How to cite

If you use this code in academic work, please cite the associated manuscript (update once published):

> Neto, O.P. et al. *[Title]*. *Nature Communications* (in review / year).

---

## License

Add your preferred license (e.g., MIT, BSD-3, Apache-2.0) in a `LICENSE` file.

---

## Contact

Osmar Pinto Neto  
GitHub: https://github.com/osmar235
