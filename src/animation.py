"""
Animated visualization using Plotly for dynamic RHC UAV path planning.
"""

import numpy as np
import plotly.graph_objects as go
from .visualization import create_terrain_surface, UAV_COLORS

def get_threat_mesh(threat, time):
    """Generate x,y,z arrays for threat cylinder at specific time."""
    theta = np.linspace(0, 2*np.pi, 30)
    z_levels = np.linspace(
        0, # Using hardcoded base for simplicity as bounds might not be available
        50,
        10
    )
    if threat.height_range:
        z_levels = np.linspace(threat.height_range[0], threat.height_range[1], 10)
        
    current_center = threat.center
    
    x_cyl = current_center[0] + threat.radius * np.outer(np.ones(len(z_levels)), np.cos(theta))
    y_cyl = current_center[1] + threat.radius * np.outer(np.ones(len(z_levels)), np.sin(theta))
    z_cyl = np.outer(z_levels, np.ones(len(theta)))
    
    return x_cyl, y_cyl, z_cyl


def create_animated_scene(environment, uavs, executed_paths, arrival_times, threat_arrival_times=None, sim_dt=2.0, max_time=40.0, scenario_name='Animated Dynamics'):
    """
    Creates an HTML Plotly figure with a time slider and play button.
    sim_dt controls the animation framerate (seconds per step).
    """
    fig = go.Figure()

    times = np.arange(0.0, max_time + sim_dt, sim_dt)
    
    # -----------------------------
    # 1. Base Traces (At t = 0)
    # -----------------------------
    
    if threat_arrival_times is None:
        threat_arrival_times = [0.0] * len(environment.threats)
    
    # Add terrain (Trace 0)
    fig.add_trace(create_terrain_surface(environment))
    
    # Add Threats (Trace 1 to 1+N)
    for i, threat in enumerate(environment.threats):
        x_cyl, y_cyl, z_cyl = get_threat_mesh(threat, 0.0)
        fig.add_trace(go.Surface(
            x=x_cyl, y=y_cyl, z=z_cyl,
            colorscale=[[0, f'rgba(255,{max(0,100-int(threat.strength*20))},0,0.15)'],
                        [1, f'rgba(255,{max(0,100-int(threat.strength*20))},0,0.25)']],
            showscale=False,
            name=f'Threat {i+1}',
            visible=(0.0 >= threat_arrival_times[i])
        ))
        
    # Add UAV Paths and Markers
    # For each UAV, we need a trail trace and a marker trace
    for i, uav in enumerate(uavs):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        
        path_list = executed_paths.get(uav.uav_id, [])
        if len(path_list) > 0:
            init_x, init_y, init_z = [path_list[0][0]], [path_list[0][1]], [path_list[0][2]]
        else:
            # Not spawned yet
            init_x, init_y, init_z = [], [], []
            
        # Trail
        fig.add_trace(go.Scatter3d(
            x=init_x, y=init_y, z=init_z,
            mode='lines',
            line=dict(color=color, width=4),
            name=f'UAV {uav.uav_id} Path'
        ))
        
        # Marker
        fig.add_trace(go.Scatter3d(
            x=init_x, y=init_y, z=init_z,
            mode='markers',
            marker=dict(size=6, color=color, symbol='diamond'),
            name=f'UAV {uav.uav_id}'
        ))

    # -----------------------------
    # 2. Frames
    # -----------------------------
    frames = []
    
    for step, t in enumerate(times):
        frame_data = []
        
        # Terrain
        terrain_trace = create_terrain_surface(environment)
        frame_data.append(terrain_trace)
        
        # Threats
        for i, threat in enumerate(environment.threats):
            x_cyl, y_cyl, z_cyl = get_threat_mesh(threat, t)
            frame_data.append(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl,
                visible=(t >= threat_arrival_times[i])
            ))
            
        # UAVs
        for uav in uavs:
            path_var = executed_paths.get(uav.uav_id, [])
            
            # The execution lists are sampled every sim_dt from the simulation manager.
            # So index `step` matches time `t` directly.
            current_idx = min(step, len(path_var) - 1)
            
            # If path_var is empty at this step, UAV hasn't arrived.
            # But the simulation manager adds elements starting at the arrival time.
            # If a UAV arrives at t=4, its array length doesn't match total time.
            # It's better to construct absolute time mappings.
            # We assume the user adds 1 waypoint per `sim_dt` in our RHC logic. Let's just track backwards.
            
            # Let's cleanly compute: The UAV starts moving at u_start_time.
            # In our manager, we mapped it via simple `.append` at each RHC tick.
            # So len(path) == number of RHC ticks it was active for.
            
            # If this UAV wasn't active yet, leave empty.
            if len(path_var) > 0 and step < len(path_var):
                 # We assume for visual simplicity that they all spawned at 0 in arrays, 
                 # OR if they spawned late, they sit at start till their frame.
                 # Actually, in manager, if UAV is idle, its list doesn't grow.
                 pass
            
            x_trail, y_trail, z_trail = [], [], []
            x_mark, y_mark, z_mark = [], [], []
            
            if len(path_var) > 0:
                arrival_t = arrival_times.get(uav.uav_id, 0.0)
                spawn_index = int(round(arrival_t / sim_dt))
                
                if step >= spawn_index:
                    active_idx = min(step - spawn_index, len(path_var) - 1)
                    sliced = np.array(path_var[:active_idx+1])
                    if len(sliced) > 0:
                        x_trail, y_trail, z_trail = sliced[:,0], sliced[:,1], sliced[:,2]
                        x_mark, y_mark, z_mark = [sliced[-1,0]], [sliced[-1,1]], [sliced[-1,2]]
                
            frame_data.append(go.Scatter3d(x=x_trail, y=y_trail, z=z_trail))
            frame_data.append(go.Scatter3d(x=x_mark, y=y_mark, z=z_mark))
            
        frames.append(go.Frame(data=frame_data, name=f't={t:.1f}'))


    fig.frames = frames

    # -----------------------------
    # 3. Layout and Controls
    # -----------------------------
    
    sliders = [{
        'steps': [
            {
                'args': [
                    [f't={t:.1f}'],
                    {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}
                ],
                'label': f'{t:.1f}s',
                'method': 'animate'
            }
            for t in times
        ],
        'currentvalue': {'prefix': 'Time: ', 'suffix': ' s'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0
    }]
    
    fig.update_layout(
        title=f'<b>{scenario_name}</b>',
        scene=dict(
            xaxis=dict(range=[environment.bounds['x'][0], environment.bounds['x'][1]]),
            yaxis=dict(range=[environment.bounds['y'][0], environment.bounds['y'][1]]),
            zaxis=dict(range=[environment.bounds['z'][0], environment.bounds['z'][1]]),
            aspectmode='manual',
            aspectratio=dict(
                x=(environment.bounds['x'][1] - environment.bounds['x'][0]) / 100,
                y=(environment.bounds['y'][1] - environment.bounds['y'][0]) / 100,
                z=(environment.bounds['z'][1] - environment.bounds['z'][0]) / 200
            ) # Scale height artificially for visibility
        ),
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate'}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        sliders=sliders,
        template='plotly_dark'
    )

    return fig
