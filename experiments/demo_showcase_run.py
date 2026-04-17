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


print("Starting Pristine Showcase Dynamic Optimization Demo...")

env = Environment(bounds={'x': (0, 100), 'y': (0, 100), 'z': (0, 50)})
# Central static threat
env.add_threat([50, 50, 0], radius=15, strength=10000.0)

# Two UAVs flying straight down their respectful corridors
u1 = UAV('UAV1', [20, 10, 10], [20, 90, 10])
u2 = UAV('UAV2', [80, 10, 10], [80, 90, 10])

params = {
    'num_particles': 40, 
    'num_waypoints': 6, 
    'max_iterations': 60,
    'gt_probability': 0.2
}

popup_threat_1 = Threat(center=(20, 40, 0), radius=10, strength=10000.0)
popup_threat_2 = Threat(center=(80, 40, 0), radius=10, strength=10000.0)

manager = RHCManager(env, CostFunction, GTCPSO_HDPLO, params, dt=2.0)

manager.add_event(0.0, 'UAV_ARRIVE', {'uav': u1})
manager.add_event(0.0, 'UAV_ARRIVE', {'uav': u2})
  
manager.add_event(2.0, 'THREAT_ARRIVE', {'threat': popup_threat_1})
manager.add_event(4.0, 'THREAT_ARRIVE', {'threat': popup_threat_2})

executed_paths = manager.run_simulation(max_time=40.0)

print("Demo calculation finished. Compiling final interactive visualizations...")

final_uavs = [u1, u2]
uav_arrival_times = {'UAV1': 0.0, 'UAV2': 0.0}
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
    'algorithm': 'D-GTCPSO (Showcase)',
    'best_paths': paths_array,
    'path_times': path_times
}]

# First threat was static (0.0s), popups at 2.0s and 4.0s
fig = create_3d_scene(env, fake_results, scenario_name='Clean Threat Avoidance Showcase', threat_times=[0.0, 2.0, 4.0])
out_path = os.path.abspath('demo_showcase.html')
fig.write_html(out_path, include_plotlyjs='cdn')
print(f"Static 3D path visualization saved to: {out_path}")
