import sys
import os
import numpy as np

from src.environment import Environment, Threat
from src.uav import UAV
from src.cost_function import CostFunction
from src.gtcpso_hdplo import GTCPSO_HDPLO
from src.rhc_manager import RHCManager
from src.visualization import create_3d_scene

def extract_paths(executed_paths, uavs, arrival_times):
    paths_array = []
    path_times = []
    for u in uavs:
        if u.uav_id in executed_paths:
            p = np.array(executed_paths[u.uav_id])
            if len(p) < 2:
                p = np.array([u.start, u.goal])
            paths_array.append(p)
            
            t_start = arrival_times[u.uav_id]
            pt = [f"{t_start + k * 2.0:.1f}" for k in range(len(p))]
            path_times.append(pt)
    return paths_array, path_times

def run_threat_deviation():
    print("Running Threat Deviation Showcase...")
    env_baseline = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})
    env_dynamic = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})

    params = {'num_particles': 40, 'num_waypoints': 6, 'max_iterations': 50, 'gt_probability': 0.2, 'seed': 42}

    # BASELINE (No Popup)
    u1_base = UAV('UAV1', [10, 50, 20], [90, 50, 20], min_speed=10.0)
    mgr_base = RHCManager(env_baseline, CostFunction, GTCPSO_HDPLO, params, dt=2.0)
    mgr_base.add_event(0.0, 'UAV_ARRIVE', {'uav': u1_base})
    paths_base = mgr_base.run_simulation(max_time=40.0)
    p_b_arr, t_b_arr = extract_paths(paths_base, [u1_base], {'UAV1': 0.0})

    # DYNAMIC (With Popup)
    u1_dyn = UAV('UAV1', [10, 50, 20], [90, 50, 20], min_speed=10.0)
    mgr_dyn = RHCManager(env_dynamic, CostFunction, GTCPSO_HDPLO, params, dt=2.0)
    mgr_dyn.add_event(0.0, 'UAV_ARRIVE', {'uav': u1_dyn})
    
    # Popup exactly in the way mid-flight. At t=4.0, uav is around x=50
    threat = Threat(center=(60, 50, 20), radius=15, strength=10000.0)
    mgr_dyn.add_event(4.0, 'THREAT_ARRIVE', {'threat': threat})
    paths_dyn = mgr_dyn.run_simulation(max_time=40.0)
    p_d_arr, t_d_arr = extract_paths(paths_dyn, [u1_dyn], {'UAV1': 0.0})

    results = [
        {'algorithm': 'Original Path (No Popup)', 'best_paths': p_b_arr, 'path_times': t_b_arr},
        {'algorithm': 'Dynamic Evasion Path', 'best_paths': p_d_arr, 'path_times': t_d_arr}
    ]
    
    # The plot shows everything sequentially. For dynamic, the plot already adds the threat at t=4.0 but the visualization tool extracts threats from env_dynamic.
    fig = create_3d_scene(env_dynamic, results, scenario_name='Clean Threat Evasion Deviation', threat_times=[4.0])
    out_path = os.path.abspath('deviation_threat_popup.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Saved Threat Deviation to {out_path}\n")

def run_uav_popup_deviation():
    print("Running UAV Popup Deviation Showcase...")
    env_baseline = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})
    env_dynamic = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})

    params = {'num_particles': 40, 'num_waypoints': 6, 'max_iterations': 60, 'gt_probability': 0.2, 'seed': 42}

    # BASELINE (No Popup)
    u1_base = UAV('UAV1', [10, 50, 20], [90, 50, 20], min_speed=10.0)
    mgr_base = RHCManager(env_baseline, CostFunction, GTCPSO_HDPLO, params, dt=2.0)
    mgr_base.add_event(0.0, 'UAV_ARRIVE', {'uav': u1_base})
    paths_base = mgr_base.run_simulation(max_time=40.0)
    p_b_arr, t_b_arr = extract_paths(paths_base, [u1_base], {'UAV1': 0.0})

    # DYNAMIC (With UAV Popup)
    u1_dyn = UAV('UAV1', [10, 50, 20], [90, 50, 20], min_speed=10.0)
    # The popup arrives mid-flight, on a crossing intercept course trajectory
    u2_popup = UAV('UAV2_Popup', [50, 10, 20], [50, 90, 20], min_speed=10.0)
    
    mgr_dyn = RHCManager(env_dynamic, CostFunction, GTCPSO_HDPLO, params, dt=2.0)
    mgr_dyn.add_event(0.0, 'UAV_ARRIVE', {'uav': u1_dyn})
    
    # Popup another UAV at t=4.0 directly aiming to intercept its path
    mgr_dyn.add_event(4.0, 'UAV_ARRIVE', {'uav': u2_popup})
    paths_dyn = mgr_dyn.run_simulation(max_time=40.0)
    p_d_arr, t_d_arr = extract_paths(paths_dyn, [u1_dyn, u2_popup], {'UAV1': 0.0, 'UAV2_Popup': 4.0})

    results = [
        {'algorithm': 'Original Path (No Collision)', 'best_paths': p_b_arr, 'path_times': t_b_arr},
        {'algorithm': 'Dynamic Evasion Path', 'best_paths': p_d_arr, 'path_times': t_d_arr}
    ]
    
    fig = create_3d_scene(env_dynamic, results, scenario_name='Clean UAV Collision Avoidance Deviation')
    out_path = os.path.abspath('deviation_uav_popup.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Saved UAV popup deviation to {out_path}\n")

def run_combination_deviation():
    print("Running Combined Popup Deviation Showcase...")
    env_baseline = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})
    env_dynamic = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})

    params = {'num_particles': 40, 'num_waypoints': 6, 'max_iterations': 60, 'gt_probability': 0.2, 'seed': 42}

    # BASELINE (No Popup)
    u1_base = UAV('UAV1', [10, 50, 20], [90, 50, 20], min_speed=10.0)
    mgr_base = RHCManager(env_baseline, CostFunction, GTCPSO_HDPLO, params, dt=2.0)
    mgr_base.add_event(0.0, 'UAV_ARRIVE', {'uav': u1_base})
    paths_base = mgr_base.run_simulation(max_time=40.0)
    p_b_arr, t_b_arr = extract_paths(paths_base, [u1_base], {'UAV1': 0.0})

    # DYNAMIC (With Threat and UAV Popup)
    u1_dyn = UAV('UAV1', [10, 50, 20], [90, 50, 20], min_speed=10.0)
    u2_popup = UAV('UAV2_Popup', [50, 10, 20], [50, 90, 20], min_speed=10.0) # crossing trajectory
    
    mgr_dyn = RHCManager(env_dynamic, CostFunction, GTCPSO_HDPLO, params, dt=2.0)
    mgr_dyn.add_event(0.0, 'UAV_ARRIVE', {'uav': u1_dyn})
    
    # Dual popup event exactly mid-flight
    threat = Threat(center=(60, 50, 20), radius=10, strength=10000.0)
    mgr_dyn.add_event(4.0, 'THREAT_ARRIVE', {'threat': threat})
    mgr_dyn.add_event(4.0, 'UAV_ARRIVE', {'uav': u2_popup})
    
    paths_dyn = mgr_dyn.run_simulation(max_time=40.0)
    p_d_arr, t_d_arr = extract_paths(paths_dyn, [u1_dyn, u2_popup], {'UAV1': 0.0, 'UAV2_Popup': 4.0})

    results = [
        {'algorithm': 'Original Path (No Popup)', 'best_paths': p_b_arr, 'path_times': t_b_arr},
        {'algorithm': 'Dynamic Evasion Path (Threat + UAV)', 'best_paths': p_d_arr, 'path_times': t_d_arr}
    ]
    
    fig = create_3d_scene(env_dynamic, results, scenario_name='Combined Threat & UAV Target Evasion', threat_times=[4.0])
    out_path = os.path.abspath('deviation_combined_popup.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Saved Combined popup deviation to {out_path}\n")

if __name__ == "__main__":
    run_threat_deviation()
    run_uav_popup_deviation()
    run_combination_deviation()
