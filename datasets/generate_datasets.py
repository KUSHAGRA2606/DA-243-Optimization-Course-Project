"""
Dataset generation for UAV path planning scenarios.
Creates 4 scenarios of increasing difficulty.
"""

import json
import os
import numpy as np


def create_scenario_simple():
    """Simple: 2 UAVs, 3 threats, 1 no-fly zone, flat terrain."""
    return {
        'name': 'Simple',
        'description': '2 UAVs, 3 threats, 1 no-fly zone, flat terrain (100x100x50)',
        'environment': {
            'bounds': {'x': [0, 100], 'y': [0, 100], 'z': [5, 50]},
            'terrain_type': 'flat',
            'threats': [
                {'center': [30, 50, 25], 'radius': 12, 'strength': 4.0, 'alpha': 2.0, 'height_range': None},
                {'center': [60, 30, 25], 'radius': 10, 'strength': 3.0, 'alpha': 2.0, 'height_range': None},
                {'center': [70, 70, 25], 'radius': 8, 'strength': 5.0, 'alpha': 2.0, 'height_range': None},
            ],
            'no_fly_zones': [
                {'center': [50, 50, 25], 'size': [8, 50], 'shape': 'cylinder'},
            ]
        },
        'uavs': [
            {'uav_id': 0, 'start': [5, 10, 20], 'goal': [95, 90, 20],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 8.0, 'fuel_capacity': 1000.0},
            {'uav_id': 1, 'start': [5, 90, 25], 'goal': [95, 10, 25],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 8.0, 'fuel_capacity': 1000.0},
        ]
    }


def create_scenario_moderate():
    """Moderate: 3 UAVs, 6 threats, 2 no-fly zones, hilly terrain."""
    return {
        'name': 'Moderate',
        'description': '3 UAVs, 6 threats, 2 no-fly zones, hilly terrain (150x150x80)',
        'environment': {
            'bounds': {'x': [0, 150], 'y': [0, 150], 'z': [5, 80]},
            'terrain_type': 'hills',
            'threats': [
                {'center': [30, 40, 30], 'radius': 15, 'strength': 4.0, 'alpha': 2.0, 'height_range': None},
                {'center': [60, 80, 35], 'radius': 12, 'strength': 5.0, 'alpha': 2.0, 'height_range': None},
                {'center': [100, 50, 40], 'radius': 18, 'strength': 3.5, 'alpha': 2.0, 'height_range': None},
                {'center': [80, 120, 30], 'radius': 10, 'strength': 6.0, 'alpha': 2.0, 'height_range': None},
                {'center': [40, 110, 35], 'radius': 14, 'strength': 4.5, 'alpha': 2.0, 'height_range': None},
                {'center': [120, 100, 40], 'radius': 11, 'strength': 3.0, 'alpha': 2.0, 'height_range': None},
            ],
            'no_fly_zones': [
                {'center': [75, 75, 40], 'size': [10, 80], 'shape': 'cylinder'},
                {'center': [110, 30, 40], 'size': [8, 60], 'shape': 'cylinder'},
            ]
        },
        'uavs': [
            {'uav_id': 0, 'start': [5, 5, 25], 'goal': [145, 145, 25],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1200.0},
            {'uav_id': 1, 'start': [5, 145, 30], 'goal': [145, 5, 30],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1200.0},
            {'uav_id': 2, 'start': [75, 5, 35], 'goal': [75, 145, 35],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1200.0},
        ]
    }


def create_scenario_complex():
    """Complex: 5 UAVs, 10 threats, 4 no-fly zones, mountainous terrain."""
    return {
        'name': 'Complex',
        'description': '5 UAVs, 10 threats, 4 no-fly zones, mountains (200x200x100)',
        'environment': {
            'bounds': {'x': [0, 200], 'y': [0, 200], 'z': [5, 100]},
            'terrain_type': 'mountains',
            'threats': [
                {'center': [40, 50, 40], 'radius': 18, 'strength': 5.0, 'alpha': 2.0, 'height_range': None},
                {'center': [80, 30, 45], 'radius': 15, 'strength': 4.0, 'alpha': 2.0, 'height_range': None},
                {'center': [120, 80, 50], 'radius': 20, 'strength': 6.0, 'alpha': 2.0, 'height_range': None},
                {'center': [60, 130, 35], 'radius': 12, 'strength': 3.5, 'alpha': 2.0, 'height_range': None},
                {'center': [160, 60, 45], 'radius': 16, 'strength': 5.5, 'alpha': 2.0, 'height_range': None},
                {'center': [100, 160, 40], 'radius': 14, 'strength': 4.5, 'alpha': 2.0, 'height_range': None},
                {'center': [30, 170, 35], 'radius': 10, 'strength': 7.0, 'alpha': 2.0, 'height_range': None},
                {'center': [180, 140, 50], 'radius': 13, 'strength': 4.0, 'alpha': 2.0, 'height_range': None},
                {'center': [140, 180, 45], 'radius': 11, 'strength': 3.0, 'alpha': 2.0, 'height_range': None},
                {'center': [100, 100, 50], 'radius': 22, 'strength': 8.0, 'alpha': 2.0, 'height_range': None},
            ],
            'no_fly_zones': [
                {'center': [50, 100, 50], 'size': [12, 100], 'shape': 'cylinder'},
                {'center': [150, 50, 50], 'size': [10, 80], 'shape': 'cylinder'},
                {'center': [100, 30, 50], 'size': [9, 70], 'shape': 'cylinder'},
                {'center': [130, 150, 50], 'size': [11, 90], 'shape': 'cylinder'},
            ]
        },
        'uavs': [
            {'uav_id': 0, 'start': [5, 5, 30], 'goal': [195, 195, 30],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1500.0},
            {'uav_id': 1, 'start': [5, 195, 35], 'goal': [195, 5, 35],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1500.0},
            {'uav_id': 2, 'start': [100, 5, 40], 'goal': [100, 195, 40],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1500.0},
            {'uav_id': 3, 'start': [5, 100, 35], 'goal': [195, 100, 35],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1500.0},
            {'uav_id': 4, 'start': [50, 5, 45], 'goal': [150, 195, 45],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 10.0, 'fuel_capacity': 1500.0},
        ]
    }


def create_scenario_extreme():
    """Extreme: 5 UAVs, 15 threats, 6 no-fly zones, rugged terrain."""
    return {
        'name': 'Extreme',
        'description': '5 UAVs, 15 threats, 6 no-fly zones, rugged terrain (250x250x120)',
        'environment': {
            'bounds': {'x': [0, 250], 'y': [0, 250], 'z': [5, 120]},
            'terrain_type': 'rugged',
            'threats': [
                {'center': [40, 40, 40], 'radius': 20, 'strength': 5.0, 'alpha': 2.0, 'height_range': None},
                {'center': [80, 60, 50], 'radius': 15, 'strength': 6.0, 'alpha': 2.0, 'height_range': None},
                {'center': [130, 40, 55], 'radius': 18, 'strength': 4.5, 'alpha': 2.0, 'height_range': None},
                {'center': [50, 120, 45], 'radius': 16, 'strength': 7.0, 'alpha': 2.0, 'height_range': None},
                {'center': [100, 100, 60], 'radius': 25, 'strength': 8.0, 'alpha': 2.0, 'height_range': None},
                {'center': [180, 80, 50], 'radius': 14, 'strength': 5.5, 'alpha': 2.0, 'height_range': None},
                {'center': [200, 150, 55], 'radius': 17, 'strength': 4.0, 'alpha': 2.0, 'height_range': None},
                {'center': [60, 200, 45], 'radius': 13, 'strength': 6.5, 'alpha': 2.0, 'height_range': None},
                {'center': [150, 180, 50], 'radius': 19, 'strength': 5.0, 'alpha': 2.0, 'height_range': None},
                {'center': [220, 30, 55], 'radius': 12, 'strength': 3.5, 'alpha': 2.0, 'height_range': None},
                {'center': [30, 180, 40], 'radius': 15, 'strength': 7.5, 'alpha': 2.0, 'height_range': None},
                {'center': [170, 220, 50], 'radius': 11, 'strength': 4.0, 'alpha': 2.0, 'height_range': None},
                {'center': [120, 150, 55], 'radius': 16, 'strength': 6.0, 'alpha': 2.0, 'height_range': None},
                {'center': [200, 200, 60], 'radius': 14, 'strength': 5.0, 'alpha': 2.0, 'height_range': None},
                {'center': [90, 220, 45], 'radius': 10, 'strength': 8.5, 'alpha': 2.0, 'height_range': None},
            ],
            'no_fly_zones': [
                {'center': [60, 60, 60], 'size': [12, 120], 'shape': 'cylinder'},
                {'center': [125, 125, 60], 'size': [15, 100], 'shape': 'cylinder'},
                {'center': [180, 50, 60], 'size': [10, 80], 'shape': 'cylinder'},
                {'center': [50, 180, 60], 'size': [11, 90], 'shape': 'cylinder'},
                {'center': [200, 180, 60], 'size': [9, 70], 'shape': 'cylinder'},
                {'center': [100, 50, 60], 'size': [13, 110], 'shape': 'cylinder'},
            ]
        },
        'uavs': [
            {'uav_id': 0, 'start': [5, 5, 35], 'goal': [245, 245, 35],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 12.0, 'fuel_capacity': 2000.0},
            {'uav_id': 1, 'start': [5, 245, 40], 'goal': [245, 5, 40],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 12.0, 'fuel_capacity': 2000.0},
            {'uav_id': 2, 'start': [125, 5, 45], 'goal': [125, 245, 45],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 12.0, 'fuel_capacity': 2000.0},
            {'uav_id': 3, 'start': [5, 125, 40], 'goal': [245, 125, 40],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 12.0, 'fuel_capacity': 2000.0},
            {'uav_id': 4, 'start': [60, 5, 50], 'goal': [190, 245, 50],
             'max_turn_angle': 1.047, 'max_climb_angle': 0.524,
             'safe_distance': 12.0, 'fuel_capacity': 2000.0},
        ]
    }


def save_all_datasets(output_dir='datasets'):
    """Save all scenario datasets as JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = {
        'scenario_simple': create_scenario_simple(),
        'scenario_moderate': create_scenario_moderate(),
        'scenario_complex': create_scenario_complex(),
        'scenario_extreme': create_scenario_extreme(),
    }
    
    for name, data in scenarios.items():
        path = os.path.join(output_dir, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")
    
    return scenarios


if __name__ == '__main__':
    save_all_datasets()
