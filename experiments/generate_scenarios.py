import os
import numpy as np

from src.environment import Environment, Threat
from src.uav import UAV
from src.cost_function import CostFunction
from src.gtcpso_hdplo import GTCPSO_HDPLO
from src.rhc_manager import RHCManager
from src.visualization import create_3d_scene

def run_scenario(scenario_index, scenario_name, env_setup, uav_events, threat_events, max_time=40.0):
    print(f"--- Running Scenario {scenario_index}: {scenario_name} ---")
    env = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})
    for t_setup in env_setup:
        env.add_threat(t_setup['center'], radius=t_setup['radius'], strength=10000.0)

    params = {
        'num_particles': 40, 
        'num_waypoints': 6, 
        'max_iterations': 50,
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
        'algorithm': 'D-GTCPSO',
        'best_paths': paths_array,
        'path_times': path_times
    }]

    fig = create_3d_scene(env, fake_results, scenario_name=scenario_name, threat_times=threat_times)
    out_path = os.path.abspath(f'scenario_{scenario_index}_{scenario_name.replace(" ", "_").replace(",", "")}.html')
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Saved: {out_path}\n")


if __name__ == "__main__":
    # Scenario 1: Standard Popup Threat (Testing basic avoidance)
    run_scenario(
        1, "Standard Popup Threat",
        env_setup=[
            {'center': [50, 20, 0], 'radius': 12}
        ],
        uav_events=[
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV1', [0, 10, 10], [90, 90, 10])}),
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV2', [10, 0, 10], [90, 80, 10])})
        ],
        threat_events=[
            (6.0, 'THREAT_ARRIVE', {'threat': Threat(center=(40, 60, 0), radius=15, strength=10000.0)})
        ]
    )

    # Scenario 2: Dynamic UAV Popup (Testing fleet adjustment)
    run_scenario(
        2, "Dynamic UAV Popup",
        env_setup=[
            {'center': [30, 40, 0], 'radius': 10},
            {'center': [70, 70, 0], 'radius': 10}
        ],
        uav_events=[
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV1', [0, 10, 15], [90, 80, 15])}),
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV2', [10, 0, 15], [80, 90, 15])}),
            # Mid-flight UAV injection
            (8.0, 'UAV_ARRIVE', {'uav': UAV('UAV3_Popup', [0, 50, 15], [90, 50, 15])})
        ],
        threat_events=[]
    )

    # Scenario 3: Crossfire Threats (Testing dense environment)
    run_scenario(
        3, "Crossfire Threats",
        env_setup=[
            {'center': [30, 20, 0], 'radius': 12},
            {'center': [70, 20, 0], 'radius': 12},
            {'center': [50, 80, 0], 'radius': 12}
        ],
        uav_events=[
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV1', [10, 10, 10], [50, 95, 10])}),
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV2', [90, 10, 10], [50, 95, 10])})
        ],
        threat_events=[
            (4.0, 'THREAT_ARRIVE', {'threat': Threat(center=(50, 50, 0), radius=10, strength=10000.0)})
        ]
    )

    # Scenario 4: The Gauntlet (UAVs and Threats appearing dynamically)
    run_scenario(
        4, "The Gauntlet",
        env_setup=[
            {'center': [50, 50, 0], 'radius': 15}
        ],
        uav_events=[
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV1', [10, 10, 10], [90, 90, 10])}),
            (6.0, 'UAV_ARRIVE', {'uav': UAV('UAV2_Delay', [0, 90, 12], [90, 10, 12])}) # Crossing path
        ],
        threat_events=[
            (8.0, 'THREAT_ARRIVE', {'threat': Threat(center=(20, 70, 0), radius=12, strength=10000.0)}),
            (8.0, 'THREAT_ARRIVE', {'threat': Threat(center=(80, 30, 0), radius=12, strength=10000.0)})
        ]
    )

    # Scenario 5: High Altitude Dive
    run_scenario(
        5, "High Altitude Dive",
        env_setup=[
            {'center': [50, 50, 0], 'radius': 20, 'height_range': (0, 20)} # Cylinder blocks lower altitude
        ],
        uav_events=[
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV1', [10, 50, 5], [90, 50, 5])}), # Must fly around or above
            (0.0, 'UAV_ARRIVE', {'uav': UAV('UAV2', [50, 10, 30], [50, 90, 5])}) # Dives down
        ],
        threat_events=[
            (10.0, 'THREAT_ARRIVE', {'threat': Threat(center=(50, 50, 0), radius=15, strength=10000.0, height_range=(20, 50))}) # Blocks the top specifically later
        ]
    )
