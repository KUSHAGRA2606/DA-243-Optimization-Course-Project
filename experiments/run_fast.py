"""
Minimal experiment: 1 run per algorithm, 2 scenarios only, small swarms.
Designed to finish in ~3-5 minutes total.
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.uav import UAV
from src.cost_function import CostFunction
from src.pso_base import StandardPSO
from src.gtcpso import GTCPSO
from src.gtcpso_hdplo import GTCPSO_HDPLO
from src.visualization import create_full_dashboard
from datasets.generate_datasets import (
    create_scenario_simple, create_scenario_moderate,
    create_scenario_complex, create_scenario_extreme
)


def load_scenario(sd):
    ed = sd['environment']
    env = Environment(ed['bounds'], ed.get('terrain_type', 'flat'))
    for t in ed.get('threats', []):
        env.add_threat(t['center'], t['radius'], t['strength'], t.get('alpha', 2.0), t.get('height_range'))
    for nfz in ed.get('no_fly_zones', []):
        env.add_no_fly_zone(nfz['center'], nfz['size'], nfz.get('shape', 'cylinder'))
    return env, [UAV.from_dict(u) for u in sd['uavs']]


def run_one(env, uavs, cls, seed=42, **kw):
    cf = CostFunction(env, uavs)
    opt = cls(env, uavs, cf, seed=seed, **kw)
    t0 = time.time()
    paths, cost = opt.optimize(verbose=False)
    return opt.get_results() | {'runtime': time.time() - t0}


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(base, 'results')
    os.makedirs(out, exist_ok=True)

    scenarios = {
        'Simple':   (create_scenario_simple(),   {'num_particles': 20, 'max_iterations': 60}),
        'Moderate': (create_scenario_moderate(),  {'num_particles': 15, 'max_iterations': 50}),
        'Complex':  (create_scenario_complex(),   {'num_particles': 15, 'max_iterations': 40}),
        'Extreme':  (create_scenario_extreme(),   {'num_particles': 12, 'max_iterations': 30}),
    }

    algs = {
        'Standard PSO': (StandardPSO, {}),
        'GTCPSO':       (GTCPSO, {'gt_probability': 0.3}),
        'GTCPSO+HDPLO': (GTCPSO_HDPLO, {
            'gt_probability': 0.3, 'cma_population': 4, 'cma_iterations': 3,
            'direction_weight': 0.5,
        }),
    }

    print("=" * 60)
    print("COOPERATIVE UAV PATH PLANNING — QUICK RUN")
    print("=" * 60)

    summary = {}

    for sc_name, (sc_data, params) in scenarios.items():
        print(f"\n{'='*50}\n{sc_name}: {sc_data['description']}\n{'='*50}")
        env, uavs = load_scenario(sc_data)
        sc_res = {}

        for aname, (acls, akw) in algs.items():
            print(f"  {aname}...", end=' ', flush=True)
            r = run_one(env, uavs, acls, num_waypoints=10, **params, **akw)
            print(f"cost={r['best_cost']:.2f}  ({r['runtime']:.1f}s)")
            sc_res[aname] = r

        # visualizations
        print("  Generating HTML visualizations...")
        vd = os.path.join(out, sc_name.lower())
        create_full_dashboard(env, list(sc_res.values()), uavs, sc_name, vd)

        # summary stats
        summary[sc_name] = {a: {'cost': r['best_cost'], 'runtime': r['runtime'],
                                'path_lengths': r['path_lengths']}
                            for a, r in sc_res.items()}

    # Final table
    print(f"\n\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<12} {'PSO':>10} {'GTCPSO':>10} {'HDPLO':>10} {'Improv%':>10}")
    print("-" * 55)
    for sc, data in summary.items():
        pso  = data['Standard PSO']['cost']
        gt   = data['GTCPSO']['cost']
        hdp  = data['GTCPSO+HDPLO']['cost']
        imp  = (pso - hdp) / pso * 100
        print(f"{sc:<12} {pso:>10.2f} {gt:>10.2f} {hdp:>10.2f} {imp:>+9.1f}%")

    print(f"\n{'='*60}")
    print("IMPROVEMENT: GTCPSO+HDPLO vs GTCPSO")
    print(f"{'='*60}")
    for sc, data in summary.items():
        gt  = data['GTCPSO']['cost']
        hdp = data['GTCPSO+HDPLO']['cost']
        imp = (gt - hdp) / gt * 100
        print(f"  {sc:<12}: {imp:+.1f}%")

    with open(os.path.join(out, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll results & HTML visualizations saved to: {out}/")


if __name__ == '__main__':
    main()
