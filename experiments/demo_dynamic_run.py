import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.environment import Environment, Threat
from src.uav import UAV
from src.cost_function import CostFunction
from src.gtcpso_hdplo import GTCPSO_HDPLO
from src.rhc_manager import RHCManager
from src.visualization import create_3d_scene


print("Starting Dynamic Optimization Demo...")

env = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})
env.add_threat([50, 20, 0], radius=15, strength=10000.0)

u1 = UAV('UAV1', [0, 10, 10], [90, 90, 10])
u2 = UAV('UAV2', [10, 0, 10], [90, 80, 10])
u3 = UAV('UAV3', [0, 80, 10], [80, 10, 10]) # Will be injected mid-flight

params = {
    'num_particles': 30, 
    'num_waypoints': 6, 
    'max_iterations': 30,
    'gt_probability': 0.2
}

# Increase strength massive to force avoidance, shrink radius to 12
popup_threat = Threat(center=(50, 50, 0), radius=12, strength=10000.0)

manager = RHCManager(env, CostFunction, GTCPSO_HDPLO, params, dt=2.0)

manager.add_event(0.0, 'UAV_ARRIVE', {'uav': u1})
manager.add_event(0.0, 'UAV_ARRIVE', {'uav': u2})
manager.add_event(4.0, 'UAV_ARRIVE', {'uav': u3})

# DISCRETE DYNAMICS: Spawn threat massively blocking the path at t=4.0
manager.add_event(4.0, 'THREAT_ARRIVE', {'threat': popup_threat})

executed_paths = manager.run_simulation(max_time=40.0)

print("Demo calculation finished. Compiling final interactive visualizations...")

final_uavs = [u1, u2, u3]
uav_arrival_times = {'UAV1': 0.0, 'UAV2': 0.0, 'UAV3': 4.0}
paths_array = []
path_times = []

for u in final_uavs:
    if u.uav_id in executed_paths:
        p = np.array(executed_paths[u.uav_id])
        if len(p) < 2:
            p = np.array([u.start, u.goal])
        paths_array.append(p)
        
        # Calculate simulation time per coordinate
        t_start = uav_arrival_times[u.uav_id]
        pt = [f"{t_start + k * 2.0:.1f}" for k in range(len(p))]
        path_times.append(pt)

fake_results = [{
    'algorithm': 'D-GTCPSO (RHC Execution)',
    'best_paths': paths_array,
    'path_times': path_times
}]

# First threat was static (0.0s), second was dynamic popup (4.0s)
fig = create_3d_scene(env, fake_results, scenario_name='Dynamic RHC Demo - Static Pathing', threat_times=[0.0, 4.0])
out_path = os.path.abspath('demo_static_plot.html')
fig.write_html(out_path, include_plotlyjs='cdn')
print(f"Static 3D path visualization saved to: {out_path}")
