import sys
import os
import numpy as np

from src.environment import Environment, Threat
from src.uav import UAV
from src.cost_function import CostFunction
from src.gtcpso_hdplo import GTCPSO_HDPLO
from src.rhc_manager import RHCManager
from src.visualization import create_3d_scene

def run_large_scenario(scenario_index, scenario_name, env_setup, uav_events, threat_events, max_time=40.0):
    print(f"--- Running Large Scale Scenario {scenario_index}: {scenario_name} ---")
    
    # 200x200x100 Large Scale Environment
    env = Environment(bounds={'x': (0, 200), 'y': (0, 200), 'z': (0, 100)})
    for t_setup in env_setup:
        env.add_threat(t_setup['center'], radius=t_setup['radius'], strength=10000.0)

    params = {
        'num_particles': 30, 
        'num_waypoints': 6, # Lowering waypoints speeds up execution and prevents tight zig-zags
        'max_iterations': 40, # Good balance of smoothing vs speed
        'gt_probability': 0.2
    }

    manager = RHCManager(env, CostFunction, GTCPSO_HDPLO, params, dt=2.0)

    uavs_dict = {}
    uav_arrival_times = {}
    for time, event_type, data in uav_events:
        manager.add_event(time, event_type, data)
        if event_type == 'UAV_ARRIVE':
            u = data['uav']
            uavs_dict[u.uav_id] = u
            uav_arrival_times[u.uav_id] = time

    threat_times = [0.0] * len(env_setup)
    for time, event_type, data in threat_events:
        manager.add_event(time, event_type, data)
        # Assuming only THREAT_ARRIVE
        threat_times.append(time)

    executed_paths = manager.run_simulation(max_time=max_time)

    paths_array = []
    path_times = []
    # Using sorted uav keys so it stays consistent
    for u_id, u in uavs_dict.items():
        if u_id in executed_paths:
            p = np.array(executed_paths[u_id])
            if len(p) < 2:
                p = np.array([u.start, u.goal])
            paths_array.append(p)
            
            t_start = uav_arrival_times[u_id]
            pt = [f"{t_start + k * 2.0:.1f}" for k in range(len(p))]
            path_times.append(pt)

    fake_results = [{
        'algorithm': 'D-GTCPSO (Large Showcase)',
        'best_paths': paths_array,
        'path_times': path_times
    }]

    fig = create_3d_scene(env, fake_results, scenario_name=scenario_name, threat_times=threat_times)
    out_path = os.path.abspath(f'large_scenario_{scenario_index}_{scenario_name.replace(" ", "_").replace(",", "")}.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    
    # Large Scenario 1: 4 UAVs, one pops up dynamically
    run_large_scenario(
        1, "4 UAV Fleet with Dynamic Spawn",
        env_setup=[
            {'center': [40, 100, 0], 'radius': 15},
            {'center': [80, 100, 0], 'radius': 15},
            {'center': [120, 100, 0], 'radius': 15},
            {'center': [160, 100, 0], 'radius': 15}
        ],
        uav_events=[
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV1', [40, 20, 20], [40, 180, 20])}),
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV2', [80, 20, 20], [80, 180, 20])}),
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV3', [120, 20, 20], [120, 180, 20])}),
            # UAV 4 injects into the simulation dynamically later
            (6.0, 'UAV_ARRIVE', {'uav': UAV('UAV4_Popup', [160, 20, 20], [160, 180, 20])})
        ],
        threat_events=[]
    )

    # Large Scenario 2: 4 UAVs, one pops up dynamically, alongside Dynamic popup threats
    run_large_scenario(
        2, "4 UAV Fleet with Threat and UAV Popups",
        env_setup=[
            # A couple static threats out of the way
            {'center': [20, 40, 0], 'radius': 12},
            {'center': [180, 140, 0], 'radius': 12}
        ],
        uav_events=[
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV1', [40, 10, 20], [40, 190, 20])}),
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV2', [80, 10, 20], [80, 190, 20])}),
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV3', [120, 10, 20], [120, 190, 20])}),
            # UAV 4 injects
            (4.0, 'UAV_ARRIVE', {'uav': UAV('UAV4_Popup', [160, 10, 20], [160, 190, 20])})
        ],
        threat_events=[
            (2.0, 'THREAT_ARRIVE', {'threat': Threat(center=(40, 50, 0), radius=12, strength=10000.0)}),
            (2.0, 'THREAT_ARRIVE', {'threat': Threat(center=(80, 60, 0), radius=12, strength=10000.0)}),
            (2.0, 'THREAT_ARRIVE', {'threat': Threat(center=(120, 50, 0), radius=12, strength=10000.0)}),
            # This threat specifically drops to block the popup UAV
            (6.0, 'THREAT_ARRIVE', {'threat': Threat(center=(160, 60, 0), radius=12, strength=10000.0)})
        ]
    )
