"""
Main experiment runner for comparing PSO, GTCPSO, and GTCPSO+HDPLO
across all scenarios with multiple independent runs.
"""

import sys
import os
import json
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.uav import UAV
from src.cost_function import CostFunction
from src.pso_base import StandardPSO
from src.gtcpso import GTCPSO
from src.gtcpso_hdplo import GTCPSO_HDPLO
from src.visualization import create_full_dashboard
from datasets.generate_datasets import save_all_datasets


def load_scenario(scenario_data):
    """Load environment and UAVs from scenario dict."""
    env_data = scenario_data['environment']
    env = Environment(env_data['bounds'], env_data.get('terrain_type', 'flat'))
    
    for t in env_data.get('threats', []):
        env.add_threat(t['center'], t['radius'], t['strength'],
                      t.get('alpha', 2.0), t.get('height_range'))
    
    for nfz in env_data.get('no_fly_zones', []):
        env.add_no_fly_zone(nfz['center'], nfz['size'], nfz.get('shape', 'cylinder'))
    
    uavs = [UAV.from_dict(u) for u in scenario_data['uavs']]
    
    return env, uavs


def run_single_experiment(env, uavs, algorithm_class, num_waypoints=10,
                           num_particles=40, max_iterations=150, seed=None, **kwargs):
    """Run a single optimization experiment."""
    cost_fn = CostFunction(env, uavs)
    
    optimizer = algorithm_class(
        env, uavs, cost_fn,
        num_particles=num_particles,
        num_waypoints=num_waypoints,
        max_iterations=max_iterations,
        seed=seed,
        **kwargs
    )
    
    start_time = time.time()
    best_paths, best_cost = optimizer.optimize(verbose=False)
    elapsed = time.time() - start_time
    
    results = optimizer.get_results()
    results['runtime'] = elapsed
    
    return results


def run_experiments(num_runs=10, num_particles=40, max_iterations=150,
                    num_waypoints=10, output_dir='results'):
    """Run full experiment suite."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate datasets
    print("=" * 70)
    print("COOPERATIVE UAV PATH PLANNING EXPERIMENTS")
    print("PSO vs GTCPSO vs GTCPSO+HDPLO")
    print("=" * 70)
    print(f"\nSettings: {num_runs} runs, {num_particles} particles, "
          f"{max_iterations} iterations, {num_waypoints} waypoints\n")
    
    scenarios = save_all_datasets(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
    )
    
    algorithms = {
        'Standard PSO': (StandardPSO, {}),
        'GTCPSO': (GTCPSO, {'gt_probability': 0.3, 'gt_phase_switch': 0.5}),
        'GTCPSO+HDPLO': (GTCPSO_HDPLO, {
            'gt_probability': 0.3, 'gt_phase_switch': 0.5,
            'crossover_rate_init': 0.5, 'diversity_threshold': 0.1,
            'cma_sigma_init': 3.0, 'cma_population': 8, 'cma_iterations': 10,
            'direction_weight': 0.5, 'direction_history_len': 5,
        }),
    }
    
    all_experiment_results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_data['name']} - {scenario_data['description']}")
        print(f"{'='*60}")
        
        env, uavs = load_scenario(scenario_data)
        
        scenario_results = {}
        
        for alg_name, (alg_class, alg_kwargs) in algorithms.items():
            print(f"\n  Running {alg_name} ({num_runs} runs)...")
            
            run_results = []
            best_overall = None
            best_cost_overall = float('inf')
            
            for run in range(num_runs):
                print(f"    Run {run+1}/{num_runs}...", end=' ')
                
                results = run_single_experiment(
                    env, uavs, alg_class,
                    num_waypoints=num_waypoints,
                    num_particles=num_particles,
                    max_iterations=max_iterations,
                    seed=run * 42 + 7,
                    **alg_kwargs
                )
                
                run_results.append({
                    'best_cost': results['best_cost'],
                    'runtime': results['runtime'],
                    'path_lengths': results['path_lengths'],
                    'convergence': results['convergence'],
                })
                
                print(f"Cost = {results['best_cost']:.4f} ({results['runtime']:.1f}s)")
                
                if results['best_cost'] < best_cost_overall:
                    best_cost_overall = results['best_cost']
                    best_overall = results
            
            # Compute statistics
            costs = [r['best_cost'] for r in run_results]
            runtimes = [r['runtime'] for r in run_results]
            
            stats = {
                'best_cost': min(costs),
                'worst_cost': max(costs),
                'mean_cost': np.mean(costs),
                'std_cost': np.std(costs),
                'median_cost': np.median(costs),
                'mean_runtime': np.mean(runtimes),
                'std_runtime': np.std(runtimes),
            }
            
            # Compute convergence speed (iterations to reach 95% of final)
            avg_convergence = np.mean(
                [r['convergence'] for r in run_results], axis=0
            )
            final_cost = avg_convergence[-1]
            target = avg_convergence[0] - 0.95 * (avg_convergence[0] - final_cost)
            conv_speed = max_iterations
            for i, c in enumerate(avg_convergence):
                if c <= target:
                    conv_speed = i
                    break
            stats['convergence_speed'] = conv_speed
            stats['avg_convergence'] = avg_convergence.tolist()
            
            scenario_results[alg_name] = {
                'stats': stats,
                'best_result': best_overall,
                'all_runs': run_results,
            }
            
            print(f"\n  {alg_name} Summary:")
            print(f"    Best: {stats['best_cost']:.4f} | Mean: {stats['mean_cost']:.4f} ± {stats['std_cost']:.4f}")
            print(f"    Conv. Speed: {stats['convergence_speed']} iters | Avg Runtime: {stats['mean_runtime']:.1f}s")
        
        all_experiment_results[scenario_name] = scenario_results
        
        # Generate visualizations for this scenario
        print(f"\n  Generating visualizations...")
        results_for_viz = []
        for alg_name in algorithms:
            best = scenario_results[alg_name]['best_result']
            results_for_viz.append(best)
        
        viz_dir = os.path.join(output_dir, scenario_data['name'].lower())
        create_full_dashboard(env, results_for_viz, uavs, scenario_data['name'], viz_dir)
    
    # Generate summary report
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    summary_data = {}
    
    for scenario_name, scenario_results in all_experiment_results.items():
        sc_name = scenarios[scenario_name]['name']
        print(f"\n--- {sc_name} ---")
        print(f"{'Algorithm':<20} {'Best':>10} {'Mean±Std':>18} {'Conv.Spd':>10} {'Runtime':>10}")
        print("-" * 70)
        
        summary_data[sc_name] = {}
        
        for alg_name, data in scenario_results.items():
            s = data['stats']
            print(f"{alg_name:<20} {s['best_cost']:>10.3f} "
                  f"{s['mean_cost']:>8.3f}±{s['std_cost']:<8.3f} "
                  f"{s['convergence_speed']:>10d} "
                  f"{s['mean_runtime']:>8.1f}s")
            
            summary_data[sc_name][alg_name] = {
                'best': s['best_cost'],
                'mean': s['mean_cost'],
                'std': s['std_cost'],
                'convergence_speed': s['convergence_speed'],
                'runtime': s['mean_runtime']
            }
    
    # Compute improvement percentages
    print(f"\n\n{'='*70}")
    print("IMPROVEMENT ANALYSIS (GTCPSO+HDPLO vs others)")
    print(f"{'='*70}")
    
    for scenario_name, scenario_results in all_experiment_results.items():
        sc_name = scenarios[scenario_name]['name']
        hdplo_stats = scenario_results['GTCPSO+HDPLO']['stats']
        
        print(f"\n--- {sc_name} ---")
        for alg_name in ['Standard PSO', 'GTCPSO']:
            other_stats = scenario_results[alg_name]['stats']
            
            cost_improvement = ((other_stats['mean_cost'] - hdplo_stats['mean_cost']) / 
                                other_stats['mean_cost'] * 100)
            conv_improvement = ((other_stats['convergence_speed'] - hdplo_stats['convergence_speed']) /
                                max(1, other_stats['convergence_speed']) * 100)
            
            print(f"  vs {alg_name}:")
            print(f"    Cost improvement: {cost_improvement:+.1f}%")
            print(f"    Convergence improvement: {conv_improvement:+.1f}%")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return all_experiment_results, summary_data


if __name__ == '__main__':
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    
    run_experiments(
        num_runs=5,
        num_particles=30,
        max_iterations=100,
        num_waypoints=10,
        output_dir=results_dir
    )

