"""
Utility functions for coordinate transformations and helpers.
Supports cylindrical <-> Cartesian conversions for GTCPSO.
"""

import numpy as np


def cylindrical_to_cartesian(rho, phi, z):
    """Convert cylindrical (rho, phi, z) to Cartesian (x, y, z)."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z


def cartesian_to_cylindrical(x, y, z):
    """Convert Cartesian (x, y, z) to cylindrical (rho, phi, z)."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi, z


def wrap_angle(phi):
    """Wrap angle to [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def euclidean_distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))


def path_length(waypoints):
    """
    Total Euclidean length of a path given as (N, 3) array of Cartesian waypoints.
    """
    diffs = np.diff(waypoints, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(segment_lengths)


def segment_lengths(waypoints):
    """
    Return array of segment lengths for a path (N, 3) -> (N-1,) array.
    """
    diffs = np.diff(waypoints, axis=0)
    return np.sqrt(np.sum(diffs**2, axis=1))


def turning_angles(waypoints):
    """
    Compute turning angle at each interior waypoint (vectorized).
    Returns (N-2,) array of angles in radians.
    """
    if len(waypoints) < 3:
        return np.array([])
    
    v1 = waypoints[1:-1] - waypoints[:-2]   # (N-2, 3)
    v2 = waypoints[2:] - waypoints[1:-1]    # (N-2, 3)
    
    norm1 = np.linalg.norm(v1, axis=1)      # (N-2,)
    norm2 = np.linalg.norm(v2, axis=1)      # (N-2,)
    
    # Avoid division by zero
    valid = (norm1 > 1e-10) & (norm2 > 1e-10)
    
    angles = np.zeros(len(v1))
    if np.any(valid):
        dots = np.sum(v1[valid] * v2[valid], axis=1)
        cos_angles = np.clip(dots / (norm1[valid] * norm2[valid]), -1.0, 1.0)
        angles[valid] = np.arccos(cos_angles)
    
    return angles


def climb_angles(waypoints):
    """
    Compute climb/dive angle at each segment (vectorized).
    Returns (N-1,) array of angles in radians.
    """
    if len(waypoints) < 2:
        return np.array([])
    
    diffs = np.diff(waypoints, axis=0)  # (N-1, 3)
    horizontal_dist = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    
    angles = np.arctan2(diffs[:, 2], np.maximum(horizontal_dist, 1e-10))
    
    return angles


def interpolate_path(waypoints, num_points=100):
    """
    Linearly interpolate a path to have num_points evenly spaced points.
    """
    seg_lens = segment_lengths(waypoints)
    cumulative = np.concatenate([[0], np.cumsum(seg_lens)])
    total_length = cumulative[-1]
    
    if total_length < 1e-10:
        return np.tile(waypoints[0], (num_points, 1))
    
    t_original = cumulative / total_length
    t_new = np.linspace(0, 1, num_points)
    
    interpolated = np.zeros((num_points, 3))
    for dim in range(3):
        interpolated[:, dim] = np.interp(t_new, t_original, waypoints[:, dim])
    
    return interpolated


def generate_random_path(start, goal, num_waypoints, bounds, rng=None):
    """
    Generate a random path from start to goal with num_waypoints intermediate points.
    Total path has num_waypoints + 2 points (including start and goal).
    
    bounds: dict with 'x': (min, max), 'y': (min, max), 'z': (min, max)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    waypoints = np.zeros((num_waypoints + 2, 3))
    waypoints[0] = start
    waypoints[-1] = goal
    
    # Generate intermediate waypoints with linear interpolation + random perturbation
    for i in range(1, num_waypoints + 1):
        t = i / (num_waypoints + 1)
        base = (1 - t) * np.array(start) + t * np.array(goal)
        
        perturbation = np.array([
            rng.uniform(-0.3, 0.3) * (bounds['x'][1] - bounds['x'][0]),
            rng.uniform(-0.3, 0.3) * (bounds['y'][1] - bounds['y'][0]),
            rng.uniform(-0.15, 0.15) * (bounds['z'][1] - bounds['z'][0]),
        ])
        
        point = base + perturbation
        # Clip to bounds
        point[0] = np.clip(point[0], bounds['x'][0], bounds['x'][1])
        point[1] = np.clip(point[1], bounds['y'][0], bounds['y'][1])
        point[2] = np.clip(point[2], bounds['z'][0], bounds['z'][1])
        
        waypoints[i] = point
    
    return waypoints
