#Publication Figure Generation Code for SUEIHCDR-PINN COVID-19 Forecasting Paper
#================================================================================

#This script generates publication-quality figures using your actual model outputs.
#Copy and paste this code into your Jupyter notebook.

#Data path: <PINN_DATA_PATH>/outputs_SUEIHCDR_PUBLICATION



import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set your base path
BASE_PATH = Path(os.environ.get("PINN_DATA_PATH", "."))
OUTPUT_PATH = BASE_PATH / "outputs_SUEIHCDR_PUBLICATION_v2_SanDiego"
METRICS_PATH = BASE_PATH / "pinn_sueihcdr_multiwindow_v2_SanDiego"

# Figure output directory (created lazily by _configure_environment())
FIG_OUTPUT = OUTPUT_PATH / "publication_figures"

# Publication-quality defaults (applied lazily by _configure_environment())
_PLOT_RCPARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
}


def _configure_environment():
    """Create the figure output directory and apply publication rcParams.

    Called lazily by generate_all_figures() so that merely importing this
    module does not create directories or mutate global matplotlib state.
    """
    FIG_OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_PLOT_RCPARAMS)

# Color palette
COLORS = {
    'PINN': '#2E86AB',
    'Hybrid': '#A23B72',
    'ARIMA': '#F18F01',
    'LogLinear': '#C73E1D',
    'Bayesian': '#6B4226',
    'observed': '#1a1a1a',
    'observed_light': '#a0a0a0',
    'S': '#1f77b4',
    'U': '#ff7f0e',
    'E': '#2ca02c',
    'I': '#d62728',
    'H': '#9467bd',
    'C': '#8c564b',
    'D': '#e377c2',
    'R': '#7f7f7f',
}

# Parameters from your model (from parameters_final.json)
PARAMS = {
    'beta0': 0.2293527275,
    'sigma': 0.2912383676,
    'delta': 0.1604585648,
    'zeta': 0.1000003666,
    'epsilon': 0.0714286864,
    'm': 0.9126121998,
    'c': 0.4633679986,
    'omega': 0.0078014438,
    'eta': 0.0020505516,
}

PHASE_ORDER = ['FirstWave', 'Delta', 'Omicron', 'BA5', 'Winter22']
PHASE_LABELS = {
    'FirstWave': 'First Wave',
    'Delta': 'Delta', 
    'Omicron': 'Omicron',
    'BA5': 'BA.5',
    'Winter22': 'Winter 22-23'
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_compartments():
    """Load compartment time series data."""
    filepath = OUTPUT_PATH / "compartments_counts.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, parse_dates=['date'])
        df.set_index('date', inplace=True)
        return df
    else:
        print(f"Warning: {filepath} not found")
        return None

def load_parameters_ts():
    """Load time-varying parameters."""
    filepath = OUTPUT_PATH / "parameters_time_series.csv"
    if filepath.exists():
        df = pd.read_csv(filepath, parse_dates=['date'])
        df.set_index('date', inplace=True)
        return df
    else:
        print(f"Warning: {filepath} not found")
        return None

def load_metrics():
    """Load forecasting metrics."""
    filepath = METRICS_PATH / "multi_window_metrics_modeloAVD.csv"
    if filepath.exists():
        return pd.read_csv(filepath)
    else:
        print(f"Warning: {filepath} not found")
        return None

def load_training_history():
    """Load training history."""
    filepath = OUTPUT_PATH / "training_history.csv"
    if filepath.exists():
        return pd.read_csv(filepath)
    else:
        print(f"Warning: {filepath} not found")
        return None

# =============================================================================
# FIGURE 1: Parameter Interpretation Table (as figure)
# =============================================================================


# =============================================================================
# FIGURE: Enhanced Compartment Dynamics
# =============================================================================

def plot_compartments_enhanced():
    """
    Create enhanced compartment visualization with better visibility of small compartments.
    """
    df = load_compartments()
    if df is None:
        print("Cannot load compartments data")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Main compartments (S, R) - fraction
    ax = axes[0, 0]
    pop = 3.3e6  # San Diego population
    
    if 'S' in df.columns:
        S_frac = df['S'] / pop if df['S'].max() > 1 else df['S']
        R_frac = df['R'] / pop if df['R'].max() > 1 else df['R']
        
        ax.fill_between(df.index, 0, S_frac, alpha=0.6, color=COLORS['S'], label='S (Susceptible)')
        ax.fill_between(df.index, S_frac, S_frac + R_frac, alpha=0.6, color=COLORS['R'], label='R (Recovered)')
        
        ax.set_ylabel('Fraction of population')
        ax.set_title('A. Susceptible and Recovered Compartments')
        ax.legend(loc='right')
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # Panel B: Active infection compartments (E, I)
    ax = axes[0, 1]
    if 'E' in df.columns and 'I' in df.columns:
        ax.plot(df.index, df['E'], '-', color=COLORS['E'], linewidth=2, label='E (Exposed)')
        ax.plot(df.index, df['I'], '-', color=COLORS['I'], linewidth=2, label='I (Infectious)')
        
        ax.set_ylabel('Individuals')
        ax.set_title('B. Active Infection Compartments')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Panel C: Protected compartment (U)
    ax = axes[1, 0]
    if 'U' in df.columns:
        ax.fill_between(df.index, 0, df['U'], alpha=0.6, color=COLORS['U'])
        ax.plot(df.index, df['U'], '-', color=COLORS['U'], linewidth=1.5, label='U (Protected)')
        
        ax.set_ylabel('Individuals')
        ax.set_title('C. Protected (Behavioral) Compartment')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Panel D: Clinical compartments (H, C, D)
    ax = axes[1, 1]
    if 'H' in df.columns:
        ax.plot(df.index, df['H'], '-', color=COLORS['H'], linewidth=2, label='H (Hospitalized)')
        ax.plot(df.index, df['C'], '-', color=COLORS['C'], linewidth=2, label='C (Critical/ICU)')
        ax.plot(df.index, df['D'], '-', color=COLORS['D'], linewidth=2, label='D (Deceased)')
        
        ax.set_ylabel('Individuals')
        ax.set_title('D. Clinical Compartments')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.suptitle('Inferred Compartment Dynamics from SUEIHCDR-PINN', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(FIG_OUTPUT / 'Figure_compartments_enhanced.png', dpi=300)
    plt.savefig(FIG_OUTPUT / 'Figure_compartments_enhanced.pdf')
    plt.show()
    
    return fig

# =============================================================================
# FIGURE: Waning Mechanisms Visualization
# =============================================================================

def plot_waning_mechanisms():
    """
    Visualize the two waning mechanisms that enable multi-wave dynamics.
    """
    df = load_compartments()
    params_ts = load_parameters_ts()
    
    if df is None:
        print("Cannot load data")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pop = 3.3e6
    
    # Panel A: Waning rates over time
    ax = axes[0, 0]
    
    # Behavioral waning rate (time-varying based on fear signal)
    omega_base = PARAMS['omega']
    eta = PARAMS['eta']
    
    # Create time array
    t = np.arange(len(df))
    
    # Approximate omega(t) = omega_base * (1 - s(t)) where s(t) is fear signal
    # For now, use constant rates
    ax.axhline(omega_base * 365, color=COLORS['U'], linestyle='-', linewidth=2,
               label=f'ω (behavioral): {omega_base*365:.2f}/year')
    ax.axhline(eta * 365, color=COLORS['R'], linestyle='--', linewidth=2,
               label=f'η (biological): {eta*365:.3f}/year')
    
    ax.set_ylabel('Waning rate (year⁻¹)')
    ax.set_title('A. Waning Rate Parameters')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 4)
    
    # Add interpretation text
    ax.text(0.5, 0.7, f'Behavioral protection duration: ~{1/omega_base:.0f} days\n'
                      f'Biological immunity duration: ~{1/eta:.0f} days',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel B: Waning flows over time
    ax = axes[0, 1]
    
    if 'U' in df.columns and 'R' in df.columns:
        U = df['U'].values
        R = df['R'].values
        
        # Compute flows
        flow_US = omega_base * U  # U → S
        flow_RS = eta * R         # R → S
        
        ax.fill_between(df.index, 0, flow_US, alpha=0.6, color=COLORS['U'], 
                        label='Behavioral waning (U→S)')
        ax.fill_between(df.index, flow_US, flow_US + flow_RS, alpha=0.6, 
                        color=COLORS['R'], label='Biological waning (R→S)')
        
        ax.set_ylabel('Daily flow to S (individuals/day)')
        ax.set_title('B. Daily Susceptible Replenishment')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Panel C: Cumulative waning contributions
    ax = axes[1, 0]
    
    if 'U' in df.columns and 'R' in df.columns:
        cum_US = np.cumsum(flow_US)
        cum_RS = np.cumsum(flow_RS)
        
        ax.plot(df.index, cum_US / pop, '-', color=COLORS['U'], linewidth=2, 
                label='Cumulative U→S')
        ax.plot(df.index, cum_RS / pop, '-', color=COLORS['R'], linewidth=2,
                label='Cumulative R→S')
        ax.plot(df.index, (cum_US + cum_RS) / pop, 'k--', linewidth=2,
                label='Total replenishment')
        
        ax.set_ylabel('Cumulative flow (fraction of population)')
        ax.set_title('C. Cumulative Susceptible Replenishment')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # Panel D: Multi-wave enabling mechanism
    ax = axes[1, 1]
    
    if 'S' in df.columns:
        S_frac = df['S'] / pop if df['S'].max() > 1 else df['S']
        
        # Compute what S would be without waning (monotonic depletion)
        # This is approximate - actual would require re-running model
        ax.plot(df.index, S_frac, '-', color=COLORS['S'], linewidth=2, 
                label='S(t) with waning')
        
        ax.axhline(0.1, color='gray', linestyle=':', alpha=0.7)
        ax.text(df.index[50], 0.12, 'Herd immunity threshold (~10%)', fontsize=9, color='gray')
        
        ax.set_ylabel('Susceptible fraction')
        ax.set_title('D. Susceptible Pool Dynamics')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.set_ylim(0, 0.25)
    
    plt.suptitle('Waning Mechanisms Enabling Multi-Wave Epidemic Dynamics', fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(FIG_OUTPUT / 'Figure_waning_mechanisms.png', dpi=300)
    plt.savefig(FIG_OUTPUT / 'Figure_waning_mechanisms.pdf')
    plt.show()
    
    return fig

# =============================================================================
# FIGURE: Forecasting MAE Summary (Main Results Figure)
# =============================================================================

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_all_figures():
    """Generate all publication figures."""

    _configure_environment()

    print("=" * 70)
    print("SUEIHCDR-PINN Publication Figure Generation")
    print("=" * 70)
    print(f"\nOutput directory: {FIG_OUTPUT}")
    print(f"Data directory: {OUTPUT_PATH}")
    print()
    
    figures = {}
    
    
    
    print("\n2. Creating enhanced compartment figure...")
    try:
        figures['compartments'] = plot_compartments_enhanced()
        print("   ✓ Complete")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n3. Creating waning mechanisms figure...")
    try:
        figures['waning'] = plot_waning_mechanisms()
        print("   ✓ Complete")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    
    
    print("\n" + "=" * 70)
    print("Figure generation complete!")
    print(f"Figures saved to: {FIG_OUTPUT}")
    print("=" * 70)
    
    return figures

# Run if executed directly
if __name__ == "__main__":
    figures = generate_all_figures()


#%% =============================================================================
# UTILITY: Load and process your actual metrics file
# =============================================================================

def analyze_metrics_file():
    """
    Load and analyze your actual metrics CSV file.
    Run this to get the exact values for the Results tables.
    """
    
    metrics = load_metrics()
    
    if metrics is None:
        print("Could not load metrics file. Check the path:")
        print(f"  {METRICS_PATH / 'multi_window_metrics_modeloAVD.csv'}")
        return None
    
    print("Metrics file loaded successfully!")
    print(f"Shape: {metrics.shape}")
    print(f"Columns: {list(metrics.columns)}")
    print()
    
    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS FOR RESULTS SECTION")
    print("=" * 60)
    
    # Adjust column names based on your actual file
    # Common patterns: 'mae_pinn', 'mae_arima', 'horizon', 'window', etc.
    
    print("\nFirst few rows:")
    print(metrics.head())
    
    print("\nColumn statistics:")
    print(metrics.describe())
    
    return metrics


# Note: analyze_metrics_file() is intentionally NOT run at import time.
# Call it explicitly from a script or from the __main__ block above.

