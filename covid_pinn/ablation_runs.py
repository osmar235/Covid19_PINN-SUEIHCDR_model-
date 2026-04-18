from .revision_helpers import *

def run_recommended_ablation():
    """Run the compact ablation set used for the paper revision."""
    results = []
    for city in ["SanDiego", "NewYork", "London", "Tokyo"]:
        for ablation_name in ["baseline", "no_waning", "no_physics"]:
            res = run_targeted_ablation(
                city,
                ablation_name=ablation_name,
                regimes=("Winter20", "BA5_Waning"),
                horizons=(7, 14),
                n_models=3,
            )
            results.append((city, ablation_name, res))
    return results

if __name__ == "__main__":
    run_recommended_ablation()
