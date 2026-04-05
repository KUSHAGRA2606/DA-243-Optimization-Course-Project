"""
Multi-objective cost function for UAV path planning.
Includes all components: path length, threat exposure, smoothness,
altitude cost, fuel consumption, and constraint violation penalties.

Also provides per-waypoint cost breakdown for bottleneck detection.
"""

import numpy as np
from . import utils


class CostFunction:
    """
    Multi-objective cost function:
    J = w1*L + w2*T + w3*S + w4*A + w5*F + P_constraints
    
    Components:
        L - Path length cost
        T - Threat exposure cost
        S - Smoothness cost (curvature/turning penalty)
        A - Altitude variation cost
        F - Fuel consumption cost
        P - Constraint violation penalties
            - Turning angle violations
            - Climb/dive angle violations
            - Inter-UAV collision violations
            - Boundary violations
            - Terrain clearance violations
            - No-fly zone violations
    """
    
    def __init__(self, environment, uavs,
                 w_length=1.0,
                 w_threat=5.0,
                 w_smooth=2.0,
                 w_altitude=0.5,
                 w_fuel=0.3,
                 penalty_turn=50.0,
                 penalty_climb=50.0,
                 penalty_collision=200.0,
                 penalty_boundary=100.0,
                 penalty_terrain=150.0,
                 penalty_nfz=500.0):
        
        self.env = environment
        self.uavs = uavs
        self.num_uavs = len(uavs)
        
        # Objective weights
        self.w_length = w_length
        self.w_threat = w_threat
        self.w_smooth = w_smooth
        self.w_altitude = w_altitude
        self.w_fuel = w_fuel
        
        # Penalty weights
        self.penalty_turn = penalty_turn
        self.penalty_climb = penalty_climb
        self.penalty_collision = penalty_collision
        self.penalty_boundary = penalty_boundary
        self.penalty_terrain = penalty_terrain
        self.penalty_nfz = penalty_nfz
    
    def path_length_cost(self, waypoints):
        """
        Cost component: Total Euclidean path length.
        Normalized by straight-line distance from start to goal.
        """
        total_length = utils.path_length(waypoints)
        straight_line = utils.euclidean_distance(waypoints[0], waypoints[-1])
        if straight_line < 1e-10:
            return total_length
        return total_length / straight_line  # Ratio - 1.0 is optimal
    
    def threat_exposure_cost(self, waypoints):
        """
        Cost component: Cumulative threat exposure along path.
        Samples threat cost at each waypoint and between waypoints.
        """
        total_cost = 0.0
        
        # At each waypoint
        for wp in waypoints:
            total_cost += self.env.threat_cost_at(wp)
        
        # Between waypoints (midpoint sampling)
        for i in range(len(waypoints) - 1):
            mid = (waypoints[i] + waypoints[i + 1]) / 2
            total_cost += self.env.threat_cost_at(mid) * 0.5
        
        return total_cost
    
    def smoothness_cost(self, waypoints, uav):
        """
        Cost component: Path smoothness measured by turning angles.
        Higher turning angles = less smooth = higher cost.
        """
        angles = utils.turning_angles(waypoints)
        if len(angles) == 0:
            return 0.0
        
        # Sum of squared turning angles (smoother paths have lower cost)
        return np.sum(angles**2)
    
    def altitude_variation_cost(self, waypoints):
        """
        Cost component: Penalize excessive altitude changes.
        """
        if len(waypoints) < 2:
            return 0.0
        
        dz = np.abs(np.diff(waypoints[:, 2]))
        return np.sum(dz**2) / len(dz)
    
    def fuel_consumption_cost(self, waypoints):
        """
        Cost component: Estimated fuel consumption.
        Based on path length + altitude gain penalty.
        """
        total_length = utils.path_length(waypoints)
        
        # Extra fuel for climbing
        altitude_gain = 0.0
        for i in range(len(waypoints) - 1):
            dz = waypoints[i + 1, 2] - waypoints[i, 2]
            if dz > 0:
                altitude_gain += dz
        
        return total_length * 0.01 + altitude_gain * 0.05
    
    def turning_angle_penalty(self, waypoints, uav):
        """Penalty for exceeding max turning angle (vectorized)."""
        angles = utils.turning_angles(waypoints)
        if len(angles) == 0:
            return 0.0
        excess = np.maximum(0, angles - uav.max_turn_angle)
        return np.sum(excess**2)
    
    def climb_angle_penalty(self, waypoints, uav):
        """Penalty for exceeding max climb/dive angle (vectorized)."""
        angles = utils.climb_angles(waypoints)
        if len(angles) == 0:
            return 0.0
        excess = np.maximum(0, np.abs(angles) - uav.max_climb_angle)
        return np.sum(excess**2)
    
    def boundary_penalty(self, waypoints):
        """Penalty for going outside operational bounds."""
        penalty = 0.0
        bounds = self.env.bounds
        for wp in waypoints:
            for i, dim in enumerate(['x', 'y', 'z']):
                if wp[i] < bounds[dim][0]:
                    penalty += (bounds[dim][0] - wp[i])**2
                elif wp[i] > bounds[dim][1]:
                    penalty += (wp[i] - bounds[dim][1])**2
        return penalty
    
    def terrain_penalty(self, waypoints):
        """Penalty for flying too close to or below terrain."""
        penalty = 0.0
        for wp in waypoints:
            clearance = self.env.terrain_clearance(wp)
            if clearance < 0:
                penalty += clearance**2
        return penalty
    
    def nfz_penalty(self, waypoints):
        """Penalty for entering no-fly zones."""
        penalty = 0.0
        for wp in waypoints:
            penalty += self.env.nfz_penalty_at(wp)
        return penalty
    
    def collision_penalty(self, all_paths):
        """
        Penalty for inter-UAV collisions (vectorized for speed).
        all_paths: list of (N, 3) waypoint arrays, one per UAV.
        """
        penalty = 0.0
        num_check_points = 15  # Reduced for speed, still sufficient
        
        for i in range(len(all_paths)):
            for j in range(i + 1, len(all_paths)):
                safe_dist = max(self.uavs[i].safe_distance, self.uavs[j].safe_distance)
                
                # Interpolate both paths (vectorized)
                path_i = utils.interpolate_path(all_paths[i], num_points=num_check_points)
                path_j = utils.interpolate_path(all_paths[j], num_points=num_check_points)
                
                # Vectorized distance computation
                diffs = path_i - path_j
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                
                violations = dists < safe_dist
                if np.any(violations):
                    penalty += np.sum((safe_dist - dists[violations])**2)
        
        return penalty
    
    def evaluate_single(self, waypoints, uav_index):
        """
        Evaluate cost for a single UAV path (without inter-UAV collision).
        Returns total cost and component breakdown.
        """
        uav = self.uavs[uav_index]
        
        # Objective components
        l_cost = self.w_length * self.path_length_cost(waypoints)
        t_cost = self.w_threat * self.threat_exposure_cost(waypoints)
        s_cost = self.w_smooth * self.smoothness_cost(waypoints, uav)
        a_cost = self.w_altitude * self.altitude_variation_cost(waypoints)
        f_cost = self.w_fuel * self.fuel_consumption_cost(waypoints)
        
        # Penalty components
        p_turn = self.penalty_turn * self.turning_angle_penalty(waypoints, uav)
        p_climb = self.penalty_climb * self.climb_angle_penalty(waypoints, uav)
        p_bound = self.penalty_boundary * self.boundary_penalty(waypoints)
        p_terrain = self.penalty_terrain * self.terrain_penalty(waypoints)
        p_nfz = self.penalty_nfz * self.nfz_penalty(waypoints)
        
        total = l_cost + t_cost + s_cost + a_cost + f_cost + p_turn + p_climb + p_bound + p_terrain + p_nfz
        
        breakdown = {
            'total': total,
            'path_length': l_cost,
            'threat_exposure': t_cost,
            'smoothness': s_cost,
            'altitude_variation': a_cost,
            'fuel_consumption': f_cost,
            'penalty_turn': p_turn,
            'penalty_climb': p_climb,
            'penalty_boundary': p_bound,
            'penalty_terrain': p_terrain,
            'penalty_nfz': p_nfz,
            'total_penalties': p_turn + p_climb + p_bound + p_terrain + p_nfz
        }
        
        return total, breakdown
    
    def evaluate_cooperative(self, all_paths):
        """
        Evaluate total cooperative cost for all UAV paths.
        Includes inter-UAV collision penalty.
        
        all_paths: list of (N, 3) waypoint arrays
        Returns: total cost, list of per-UAV breakdowns
        """
        total_cost = 0.0
        breakdowns = []
        
        for i, path in enumerate(all_paths):
            cost, breakdown = self.evaluate_single(path, i)
            total_cost += cost
            breakdowns.append(breakdown)
        
        # Inter-UAV collision penalty
        coll_penalty = self.penalty_collision * self.collision_penalty(all_paths)
        total_cost += coll_penalty
        
        return total_cost, breakdowns, coll_penalty
    
    def per_waypoint_cost(self, waypoints, uav_index):
        """
        Compute cost contribution of each waypoint.
        Used for Gene Targeting bottleneck detection.
        Returns (N,) array of per-waypoint costs.
        """
        uav = self.uavs[uav_index]
        n = len(waypoints)
        costs = np.zeros(n)
        
        for i in range(n):
            wp = waypoints[i]
            
            # Threat at this waypoint
            costs[i] += self.w_threat * self.env.threat_cost_at(wp)
            
            # Segment length contribution (before and after)
            if i > 0:
                seg_len = utils.euclidean_distance(waypoints[i-1], wp)
                costs[i] += self.w_length * seg_len * 0.5 / max(1.0, utils.euclidean_distance(waypoints[0], waypoints[-1]))
            if i < n - 1:
                seg_len = utils.euclidean_distance(wp, waypoints[i+1])
                costs[i] += self.w_length * seg_len * 0.5 / max(1.0, utils.euclidean_distance(waypoints[0], waypoints[-1]))
            
            # Turning angle at waypoint
            if 0 < i < n - 1:
                v1 = waypoints[i] - waypoints[i-1]
                v2 = waypoints[i+1] - waypoints[i]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                    angle = np.arccos(cos_a)
                    costs[i] += self.w_smooth * angle**2
                    if angle > uav.max_turn_angle:
                        costs[i] += self.penalty_turn * (angle - uav.max_turn_angle)**2
            
            # Terrain clearance
            clearance = self.env.terrain_clearance(wp)
            if clearance < 0:
                costs[i] += self.penalty_terrain * clearance**2
            
            # No-fly zone
            costs[i] += self.penalty_nfz * self.env.nfz_penalty_at(wp) / max(1.0, 1000.0)
            
            # Boundary violation
            bounds = self.env.bounds
            for d, dim in enumerate(['x', 'y', 'z']):
                if wp[d] < bounds[dim][0]:
                    costs[i] += self.penalty_boundary * (bounds[dim][0] - wp[d])**2
                elif wp[d] > bounds[dim][1]:
                    costs[i] += self.penalty_boundary * (wp[d] - bounds[dim][1])**2
        
        return costs
