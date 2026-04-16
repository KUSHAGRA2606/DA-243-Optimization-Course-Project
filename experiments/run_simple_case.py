"""
Run only the Simple scenario with Standard PSO, GTCPSO, and GTCPSO+HDPLO.
This is the fastest way to validate basic functionality.
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.uav import UAV
from src.cost_function import CostFunction
from src.pso_base import StandardPSO
from src.gtcpso import GTCPSO
from src.gtcpso_hdplo import GTCPSO_HDPLO
from src.visualization import create_full_dashboard
from datasets.generate_datasets import create_scenario_simple


def load_scenario(scenario_data):
    env_data = scenario_data["environment"]
    env = Environment(env_data["bounds"], env_data.get("terrain_type", "flat"))

    for threat in env_data.get("threats", []):
        env.add_threat(
            threat["center"],
            threat["radius"],
            threat["strength"],
            threat.get("alpha", 2.0),
            threat.get("height_range"),
        )

    for nfz in env_data.get("no_fly_zones", []):
        env.add_no_fly_zone(nfz["center"], nfz["size"], nfz.get("shape", "cylinder"))

    uavs = [UAV.from_dict(uav_data) for uav_data in scenario_data["uavs"]]
    return env, uavs


def run_once(env, uavs, algorithm_cls, seed=42, **kwargs):
    cost_fn = CostFunction(env, uavs)
    optimizer = algorithm_cls(env, uavs, cost_fn, seed=seed, **kwargs)

    start = time.time()
    optimizer.optimize(verbose=False)
    runtime = time.time() - start

    results = optimizer.get_results()
    results["runtime"] = runtime
    return results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_root = os.path.join(base_dir, "results", "simple_only")
    os.makedirs(output_root, exist_ok=True)

    scenario_data = create_scenario_simple()
    env, uavs = load_scenario(scenario_data)

    params = {"num_particles": 20, "max_iterations": 60, "num_waypoints": 10}

    algorithms = {
        "Standard PSO": (StandardPSO, {}),
        "GTCPSO": (GTCPSO, {"gt_probability": 0.3}),
        "GTCPSO+HDPLO": (
            GTCPSO_HDPLO,
            {
                "gt_probability": 0.3,
                "cma_population": 4,
                "cma_iterations": 3,
                "direction_weight": 0.5,
            },
        ),
    }

    print("=" * 56)
    print("SIMPLE SCENARIO ONLY")
    print("=" * 56)

    all_results = {}
    viz_results = []

    for name, (algo_cls, algo_kwargs) in algorithms.items():
        print(f"Running {name}...", end=" ", flush=True)
        result = run_once(env, uavs, algo_cls, **params, **algo_kwargs)
        all_results[name] = {
            "cost": result["best_cost"],
            "runtime": result["runtime"],
            "path_lengths": result["path_lengths"],
        }
        viz_results.append(result)
        print(f"cost={result['best_cost']:.2f} ({result['runtime']:.1f}s)")

    create_full_dashboard(env, viz_results, uavs, "Simple_Only", output_root)

    out_json = os.path.join(output_root, "simple_only_summary.json")
    with open(out_json, "w", encoding="utf-8") as file_obj:
        json.dump(all_results, file_obj, indent=2)

    print("\nSaved outputs:")
    print(f"  Dashboard folder: {output_root}")
    print(f"  Summary JSON: {out_json}")


if __name__ == "__main__":
    main()
