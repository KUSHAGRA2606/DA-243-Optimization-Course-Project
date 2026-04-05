"""
Interactive HTML visualization using Plotly for UAV path planning results.
Shows 3D paths, threats, terrain, no-fly zones, bottleneck waypoints,
convergence curves, and cost breakdowns.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os


def create_terrain_surface(environment, opacity=0.4):
    """Create a 3D terrain surface trace."""
    terrain = environment.terrain
    x = np.linspace(terrain.x_range[0], terrain.x_range[1], terrain.nx)
    y = np.linspace(terrain.y_range[0], terrain.y_range[1], terrain.ny)
    
    return go.Surface(
        x=x, y=y, z=terrain.heightmap.T,
        colorscale='earth',
        opacity=opacity,
        showscale=False,
        name='Terrain',
        hoverinfo='skip'
    )


def create_threat_traces(environment):
    """Create cylindrical threat zone visualizations."""
    traces = []
    
    for i, threat in enumerate(environment.threats):
        # Create cylinder mesh for threat zone
        theta = np.linspace(0, 2*np.pi, 30)
        z_levels = np.linspace(
            environment.bounds['z'][0],
            environment.bounds['z'][1],
            10
        )
        
        if threat.height_range:
            z_levels = np.linspace(threat.height_range[0], threat.height_range[1], 10)
        
        x_cyl = threat.center[0] + threat.radius * np.outer(np.ones(len(z_levels)), np.cos(theta))
        y_cyl = threat.center[1] + threat.radius * np.outer(np.ones(len(z_levels)), np.sin(theta))
        z_cyl = np.outer(z_levels, np.ones(len(theta)))
        
        traces.append(go.Surface(
            x=x_cyl, y=y_cyl, z=z_cyl,
            colorscale=[[0, f'rgba(255,{max(0,100-int(threat.strength*20))},0,0.15)'],
                        [1, f'rgba(255,{max(0,100-int(threat.strength*20))},0,0.25)']],
            showscale=False,
            name=f'Threat {i+1} (str={threat.strength:.1f})',
            hovertemplate=f'Threat {i+1}<br>Strength: {threat.strength}<br>Radius: {threat.radius}<extra></extra>'
        ))
        
        # Threat center marker
        traces.append(go.Scatter3d(
            x=[threat.center[0]], y=[threat.center[1]], z=[threat.center[2]],
            mode='markers',
            marker=dict(size=6, color='red', symbol='diamond'),
            name=f'Threat {i+1} Center',
            showlegend=False,
            hovertemplate=f'Threat {i+1} Center<br>({threat.center[0]:.0f}, {threat.center[1]:.0f}, {threat.center[2]:.0f})<extra></extra>'
        ))
    
    return traces


def create_nfz_traces(environment):
    """Create no-fly zone visualization."""
    traces = []
    
    for i, nfz in enumerate(environment.no_fly_zones):
        if nfz.shape == 'cylinder':
            theta = np.linspace(0, 2*np.pi, 30)
            z_bot = nfz.center[2] - nfz.size[1] / 2
            z_top = nfz.center[2] + nfz.size[1] / 2
            z_levels = np.linspace(z_bot, z_top, 5)
            
            x_cyl = nfz.center[0] + nfz.size[0] * np.outer(np.ones(len(z_levels)), np.cos(theta))
            y_cyl = nfz.center[1] + nfz.size[0] * np.outer(np.ones(len(z_levels)), np.sin(theta))
            z_cyl = np.outer(z_levels, np.ones(len(theta)))
            
            traces.append(go.Surface(
                x=x_cyl, y=y_cyl, z=z_cyl,
                colorscale=[[0, 'rgba(128,128,128,0.3)'], [1, 'rgba(128,128,128,0.4)']],
                showscale=False,
                name=f'No-Fly Zone {i+1}'
            ))
        else:  # box
            half = nfz.size / 2
            corners = nfz.center + np.array([
                [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]
            ]) * half
            
            traces.append(go.Mesh3d(
                x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                i=[0,0,0,0,4,4,0,1,2,3,0,1],
                j=[1,2,4,1,5,6,4,5,6,7,3,2],
                k=[2,3,5,5,6,7,1,2,3,4,4,6],
                color='gray',
                opacity=0.3,
                name=f'No-Fly Zone {i+1}'
            ))
    
    return traces


UAV_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


def create_path_traces(paths, uavs, algorithm_name, bottleneck_info=None,
                       show_waypoints=True, dash=None):
    """Create path traces for all UAVs."""
    traces = []
    
    for u, (path, uav) in enumerate(zip(paths, uavs)):
        color = UAV_COLORS[u % len(UAV_COLORS)]
        
        # Path line
        line_style = dict(color=color, width=4)
        if dash:
            line_style['dash'] = dash
        
        traces.append(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines',
            line=line_style,
            name=f'{algorithm_name} - UAV {uav.uav_id}',
            hovertemplate=(
                f'{algorithm_name} - UAV {uav.uav_id}<br>'
                'X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
            )
        ))
        
        if show_waypoints:
            # Waypoint markers
            traces.append(go.Scatter3d(
                x=path[1:-1, 0], y=path[1:-1, 1], z=path[1:-1, 2],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.7),
                name=f'{algorithm_name} - UAV {uav.uav_id} Waypoints',
                showlegend=False,
                hovertemplate=(
                    f'Waypoint (UAV {uav.uav_id})<br>'
                    'X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
                )
            ))
        
        # Start marker
        traces.append(go.Scatter3d(
            x=[path[0, 0]], y=[path[0, 1]], z=[path[0, 2]],
            mode='markers+text',
            marker=dict(size=8, color='green', symbol='circle'),
            text=[f'S{uav.uav_id}'],
            textposition='top center',
            name=f'Start UAV {uav.uav_id}',
            showlegend=False,
        ))
        
        # Goal marker
        traces.append(go.Scatter3d(
            x=[path[-1, 0]], y=[path[-1, 1]], z=[path[-1, 2]],
            mode='markers+text',
            marker=dict(size=8, color='blue', symbol='square'),
            text=[f'G{uav.uav_id}'],
            textposition='top center',
            name=f'Goal UAV {uav.uav_id}',
            showlegend=False,
        ))
        
        # Bottleneck marker
        if bottleneck_info:
            for bi in bottleneck_info:
                if bi['uav'] == u:
                    b_idx = bi['bottleneck_idx']
                    traces.append(go.Scatter3d(
                        x=[path[b_idx, 0]], y=[path[b_idx, 1]], z=[path[b_idx, 2]],
                        mode='markers',
                        marker=dict(size=10, color='yellow', symbol='x',
                                    line=dict(width=2, color='red')),
                        name=f'Bottleneck UAV {uav.uav_id}',
                        hovertemplate=(
                            f'BOTTLENECK - UAV {uav.uav_id}<br>'
                            f'Waypoint Index: {b_idx}<br>'
                            f'Cost: {bi["wp_costs"][b_idx]:.3f}<br>'
                            'X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
                        )
                    ))
    
    return traces


def create_3d_scene(environment, results_list, scenario_name='Scenario'):
    """
    Create a full interactive 3D visualization.
    
    results_list: list of result dicts from different algorithms
    """
    fig = go.Figure()
    
    # Add terrain
    fig.add_trace(create_terrain_surface(environment))
    
    # Add threats
    for trace in create_threat_traces(environment):
        fig.add_trace(trace)
    
    # Add no-fly zones
    for trace in create_nfz_traces(environment):
        fig.add_trace(trace)
    
    # Add paths for each algorithm
    dash_styles = [None, 'dash', 'dot']
    for i, results in enumerate(results_list):
        bottleneck_info = results.get('bottleneck_info', None)
        dash = dash_styles[i % len(dash_styles)]
        
        for trace in create_path_traces(
            results['best_paths'], 
            [type('UAV', (), {'uav_id': u})() for u in range(len(results['best_paths']))],
            results['algorithm'],
            bottleneck_info=bottleneck_info,
            dash=dash
        ):
            fig.add_trace(trace)
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'<b>Cooperative UAV Path Planning - {scenario_name}</b>',
            font=dict(size=20, family='Inter, sans-serif'),
            x=0.5
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Altitude (m)',
            aspectmode='manual',
            aspectratio=dict(
                x=(environment.bounds['x'][1] - environment.bounds['x'][0]) / 100,
                y=(environment.bounds['y'][1] - environment.bounds['y'][0]) / 100,
                z=(environment.bounds['z'][1] - environment.bounds['z'][0]) / 200
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        template='plotly_dark',
        width=1200,
        height=800,
        legend=dict(
            x=0.01, y=0.99,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(size=10)
        )
    )
    
    return fig


def create_convergence_plot(results_list, scenario_name='Scenario'):
    """Create convergence curves comparison."""
    fig = go.Figure()
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    for i, results in enumerate(results_list):
        fig.add_trace(go.Scatter(
            y=results['convergence'],
            mode='lines',
            name=results['algorithm'],
            line=dict(color=colors[i % len(colors)], width=2.5),
            hovertemplate=f'{results["algorithm"]}<br>Iter: %{{x}}<br>Cost: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Convergence Curves - {scenario_name}</b>',
            font=dict(size=18, family='Inter, sans-serif'),
            x=0.5
        ),
        xaxis_title='Iteration',
        yaxis_title='Best Cost',
        template='plotly_dark',
        width=900,
        height=500,
        legend=dict(x=0.7, y=0.95),
        hovermode='x unified'
    )
    
    return fig


def create_cost_breakdown_chart(results_list, scenario_name='Scenario'):
    """Create stacked bar chart of cost components."""
    components = ['path_length', 'threat_exposure', 'smoothness',
                  'altitude_variation', 'fuel_consumption',
                  'penalty_turn', 'penalty_climb', 'penalty_boundary',
                  'penalty_terrain', 'penalty_nfz']
    
    component_labels = ['Path Length', 'Threat Exposure', 'Smoothness',
                        'Altitude Var.', 'Fuel', 'Turn Penalty',
                        'Climb Penalty', 'Boundary Penalty',
                        'Terrain Penalty', 'NFZ Penalty']
    
    colors = px.colors.qualitative.Set3
    
    fig = go.Figure()
    
    algorithms = [r['algorithm'] for r in results_list]
    
    for c_idx, (comp, label) in enumerate(zip(components, component_labels)):
        values = []
        for results in results_list:
            total = sum(bd.get(comp, 0) for bd in results['cost_breakdowns'])
            values.append(total)
        
        fig.add_trace(go.Bar(
            name=label,
            x=algorithms,
            y=values,
            marker_color=colors[c_idx % len(colors)],
            hovertemplate=f'{label}<br>%{{x}}: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Cost Component Breakdown - {scenario_name}</b>',
            font=dict(size=18, family='Inter, sans-serif'),
            x=0.5
        ),
        barmode='stack',
        xaxis_title='Algorithm',
        yaxis_title='Cost',
        template='plotly_dark',
        width=900,
        height=500,
        legend=dict(x=1.02, y=1, font=dict(size=9))
    )
    
    return fig


def create_bottleneck_analysis(results, uavs, scenario_name='Scenario'):
    """Create bottleneck waypoint cost visualization."""
    if 'bottleneck_info' not in results or not results['bottleneck_info']:
        return None
    
    fig = make_subplots(
        rows=1, cols=len(results['bottleneck_info']),
        subplot_titles=[f'UAV {bi["uav"]}' for bi in results['bottleneck_info']]
    )
    
    for i, bi in enumerate(results['bottleneck_info']):
        wp_costs = np.array(bi['wp_costs'])
        x_vals = list(range(len(wp_costs)))
        
        colors = ['green' if j == 0 or j == len(wp_costs)-1 
                  else ('red' if j == bi['bottleneck_idx'] else '#4ecdc4')
                  for j in x_vals]
        
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=wp_costs,
                marker_color=colors,
                name=f'UAV {bi["uav"]} WP Costs',
                hovertemplate='WP %{x}<br>Cost: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        # Highlight bottleneck
        fig.add_trace(
            go.Scatter(
                x=[bi['bottleneck_idx']],
                y=[wp_costs[bi['bottleneck_idx']]],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='star'),
                text=['BOTTLENECK'],
                textposition='top center',
                textfont=dict(color='red', size=10),
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=dict(
            text=f'<b>Per-Waypoint Cost Analysis ({results["algorithm"]}) - {scenario_name}</b>',
            font=dict(size=16, family='Inter, sans-serif'),
            x=0.5
        ),
        template='plotly_dark',
        width=400 * len(results['bottleneck_info']),
        height=400
    )
    
    return fig


def create_diversity_plot(hdplo_results, scenario_name='Scenario'):
    """Plot diversity and crossover rate history for HDPLO."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Population Diversity', 'Adaptive Crossover Rate'],
        vertical_spacing=0.15
    )
    
    if 'diversity_history' in hdplo_results and hdplo_results['diversity_history']:
        fig.add_trace(
            go.Scatter(
                y=hdplo_results['diversity_history'],
                mode='lines',
                line=dict(color='#45b7d1', width=2),
                name='Diversity',
                hovertemplate='Step: %{x}<br>Diversity: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    if 'crossover_rate_history' in hdplo_results and hdplo_results['crossover_rate_history']:
        fig.add_trace(
            go.Scatter(
                y=hdplo_results['crossover_rate_history'],
                mode='lines',
                line=dict(color='#ff6b6b', width=2),
                name='Crossover Rate',
                hovertemplate='Step: %{x}<br>Rate: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title=dict(
            text=f'<b>HDPLO Diversity & Crossover Adaptation - {scenario_name}</b>',
            font=dict(size=16, family='Inter, sans-serif'),
            x=0.5
        ),
        template='plotly_dark',
        width=900,
        height=600,
        showlegend=True
    )
    
    return fig


def create_full_dashboard(environment, results_list, uavs, scenario_name='Scenario',
                          output_dir='results'):
    """
    Create and save all visualization HTML files.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # 1. 3D Scene
    fig_3d = create_3d_scene(environment, results_list, scenario_name)
    path_3d = os.path.join(output_dir, f'{scenario_name.lower().replace(" ", "_")}_3d_paths.html')
    fig_3d.write_html(path_3d, include_plotlyjs='cdn')
    saved_files.append(path_3d)
    print(f"  Saved: {path_3d}")
    
    # 2. Convergence
    fig_conv = create_convergence_plot(results_list, scenario_name)
    path_conv = os.path.join(output_dir, f'{scenario_name.lower().replace(" ", "_")}_convergence.html')
    fig_conv.write_html(path_conv, include_plotlyjs='cdn')
    saved_files.append(path_conv)
    print(f"  Saved: {path_conv}")
    
    # 3. Cost Breakdown
    fig_cost = create_cost_breakdown_chart(results_list, scenario_name)
    path_cost = os.path.join(output_dir, f'{scenario_name.lower().replace(" ", "_")}_cost_breakdown.html')
    fig_cost.write_html(path_cost, include_plotlyjs='cdn')
    saved_files.append(path_cost)
    print(f"  Saved: {path_cost}")
    
    # 4. Bottleneck Analysis for GTCPSO and HDPLO
    for results in results_list:
        if 'bottleneck_info' in results and results['bottleneck_info']:
            fig_bn = create_bottleneck_analysis(results, uavs, scenario_name)
            if fig_bn:
                alg_name = results['algorithm'].lower().replace(' ', '_').replace('+', 'plus')
                path_bn = os.path.join(output_dir,
                    f'{scenario_name.lower().replace(" ", "_")}_{alg_name}_bottleneck.html')
                fig_bn.write_html(path_bn, include_plotlyjs='cdn')
                saved_files.append(path_bn)
                print(f"  Saved: {path_bn}")
    
    # 5. Diversity plot for HDPLO
    for results in results_list:
        if 'diversity_history' in results and results['diversity_history']:
            fig_div = create_diversity_plot(results, scenario_name)
            path_div = os.path.join(output_dir,
                f'{scenario_name.lower().replace(" ", "_")}_hdplo_diversity.html')
            fig_div.write_html(path_div, include_plotlyjs='cdn')
            saved_files.append(path_div)
            print(f"  Saved: {path_div}")
    
    return saved_files
