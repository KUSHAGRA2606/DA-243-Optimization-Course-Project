"""
GTCPSO: Gene Targeting Cylindrical coordinate Particle Swarm Optimization.

Implements the main paper's algorithm:
1. Cylindrical coordinate representation (rho, phi, z) for waypoints
2. Gene Targeting: detect bottleneck waypoint and replace with improved vector
3. Two-phase update: exploration (early) vs exploitation (late)
"""

import numpy as np
from . import utils
from .cost_function import CostFunction


class GTCPSO:
    """
    Gene Targeting Cylindrical coordinate PSO for cooperative UAV path planning.
    """
    
    def __init__(self, environment, uavs, cost_function,
                 num_particles=50,
                 num_waypoints=10,
                 max_iterations=200,
                 w_start=0.9,
                 w_end=0.4,
                 c1=2.0,
                 c2=2.0,
                 gt_probability=0.3,      # Probability of applying gene targeting per iteration
                 gt_phase_switch=0.5,     # Fraction of iterations to switch from phase 1 to 2
                 seed=None):
        
        self.env = environment
        self.uavs = uavs
        self.cost_fn = cost_function
        self.num_particles = num_particles
        self.num_waypoints = num_waypoints
        self.max_iterations = max_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.gt_probability = gt_probability
        self.gt_phase_switch = gt_phase_switch
        self.rng = np.random.default_rng(seed)
        
        self.num_uavs = len(uavs)
        
        # Swarm state
        self.particles = []          # List of particles; each is list of paths (cylindrical)
        self.particles_cart = []     # Cartesian version for evaluation
        self.velocities = []
        self.pbest_positions = []
        self.pbest_positions_cart = []
        self.pbest_costs = []
        self.gbest_position = None
        self.gbest_position_cart = None
        self.gbest_cost = float('inf')
        
        self.convergence_history = []
        self.cost_breakdown_history = []
        self.bottleneck_history = []    # Track bottleneck waypoints
    
    def _cart_to_cyl_path(self, cart_path, start, goal):
        """
        Convert Cartesian path to cylindrical coordinates relative to the
        line from start to goal (the reference axis).
        
        Each intermediate waypoint is encoded as (rho, phi, z_offset) where:
        - rho: radial distance from the reference line
        - phi: angular position around reference line
        - z_offset: height offset from linearly interpolated altitude
        """
        n = len(cart_path)
        cyl_path = np.zeros((n, 3))
        
        # Start and goal stay fixed
        cyl_path[0] = [0, 0, 0]
        cyl_path[-1] = [0, 0, 0]
        
        direction = goal - start
        total_dist = np.linalg.norm(direction[:2])
        
        if total_dist < 1e-10:
            # Start and goal at same XY location
            for i in range(1, n - 1):
                cyl_path[i] = [0, 0, cart_path[i, 2] - start[2]]
            return cyl_path
        
        dir_norm = direction[:2] / total_dist
        # Perpendicular direction
        perp = np.array([-dir_norm[1], dir_norm[0]])
        
        for i in range(1, n - 1):
            t = i / (n - 1)
            # Linear interpolation point
            ref_point = start + t * direction
            # Offset from reference
            offset = cart_path[i] - ref_point
            
            # Project to along-track and cross-track
            along = np.dot(offset[:2], dir_norm)
            cross = np.dot(offset[:2], perp)
            
            rho = np.sqrt(along**2 + cross**2)
            phi = np.arctan2(cross, along)
            z_off = offset[2]
            
            cyl_path[i] = [rho, phi, z_off]
        
        return cyl_path
    
    def _cyl_to_cart_path(self, cyl_path, start, goal):
        """Convert cylindrical path back to Cartesian."""
        n = len(cyl_path)
        cart_path = np.zeros((n, 3))
        
        cart_path[0] = start.copy()
        cart_path[-1] = goal.copy()
        
        direction = goal - start
        total_dist = np.linalg.norm(direction[:2])
        
        if total_dist < 1e-10:
            for i in range(1, n - 1):
                cart_path[i] = start.copy()
                cart_path[i, 2] += cyl_path[i, 2]
            return cart_path
        
        dir_norm = direction[:2] / total_dist
        perp = np.array([-dir_norm[1], dir_norm[0]])
        
        for i in range(1, n - 1):
            t = i / (n - 1)
            ref_point = start + t * direction
            
            rho, phi, z_off = cyl_path[i]
            
            # Convert back from along/cross
            along = rho * np.cos(phi)
            cross = rho * np.sin(phi)
            
            offset_xy = along * dir_norm + cross * perp
            
            cart_path[i, 0] = ref_point[0] + offset_xy[0]
            cart_path[i, 1] = ref_point[1] + offset_xy[1]
            cart_path[i, 2] = ref_point[2] + z_off
        
        return cart_path
    
    def _initialize(self):
        """Initialize swarm in cylindrical coordinates."""
        self.particles = []
        self.particles_cart = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_positions_cart = []
        self.pbest_costs = []
        
        for p in range(self.num_particles):
            particle_cyl = []
            particle_cart = []
            particle_vel = []
            
            for uav in self.uavs:
                # Generate random Cartesian path first
                cart_path = utils.generate_random_path(
                    uav.start, uav.goal, self.num_waypoints,
                    self.env.bounds, self.rng
                )
                
                # Convert to cylindrical
                cyl_path = self._cart_to_cyl_path(cart_path, uav.start, uav.goal)
                
                particle_cyl.append(cyl_path.copy())
                particle_cart.append(cart_path.copy())
                
                # Initialize velocity in cylindrical space
                vel = np.zeros_like(cyl_path)
                vel[1:-1, 0] = self.rng.uniform(-5, 5, self.num_waypoints)   # rho velocity
                vel[1:-1, 1] = self.rng.uniform(-0.3, 0.3, self.num_waypoints)  # phi velocity
                vel[1:-1, 2] = self.rng.uniform(-3, 3, self.num_waypoints)   # z velocity
                particle_vel.append(vel)
            
            self.particles.append(particle_cyl)
            self.particles_cart.append(particle_cart)
            self.velocities.append(particle_vel)
            self.pbest_positions.append([p.copy() for p in particle_cyl])
            self.pbest_positions_cart.append([p.copy() for p in particle_cart])
            
            cost, _, _ = self.cost_fn.evaluate_cooperative(particle_cart)
            self.pbest_costs.append(cost)
            
            if cost < self.gbest_cost:
                self.gbest_cost = cost
                self.gbest_position = [p.copy() for p in particle_cyl]
                self.gbest_position_cart = [p.copy() for p in particle_cart]
    
    def _detect_bottleneck(self, uav_index):
        """
        Gene Detector: Identify the bottleneck waypoint in the global best path.
        The bottleneck is the waypoint with the highest per-waypoint cost.
        Returns the index of the bottleneck waypoint.
        """
        path = self.gbest_position_cart[uav_index]
        costs = self.cost_fn.per_waypoint_cost(path, uav_index)
        
        # Only consider interior waypoints (not start/goal)
        interior_costs = costs[1:-1]
        bottleneck_idx = np.argmax(interior_costs) + 1  # Offset by 1 for start
        
        return bottleneck_idx, costs
    
    def _gene_targeting(self, uav_index, iteration):
        """
        Gene Targeting operation: replace bottleneck waypoint with improved vector.
        
        Two-phase strategy:
        Phase 1 (exploration): Large perturbation, broad search
        Phase 2 (exploitation): Small perturbation, fine-tuning
        """
        uav = self.uavs[uav_index]
        bottleneck_idx, wp_costs = self._detect_bottleneck(uav_index)
        
        self.bottleneck_history.append({
            'iteration': iteration,
            'uav': uav_index,
            'bottleneck_idx': bottleneck_idx,
            'bottleneck_cost': wp_costs[bottleneck_idx],
            'all_wp_costs': wp_costs.tolist()
        })
        
        # Current bottleneck position (cylindrical)
        current_cyl = self.gbest_position[uav_index][bottleneck_idx].copy()
        
        # Determine phase
        phase_ratio = iteration / self.max_iterations
        is_phase2 = phase_ratio >= self.gt_phase_switch
        
        best_cost = self.gbest_cost
        best_replacement = current_cyl.copy()
        
        # Generate candidate replacements
        num_candidates = 10 if is_phase2 else 20
        
        for _ in range(num_candidates):
            if is_phase2:
                # Phase 2 (exploitation): Small perturbation around current
                perturbation = np.array([
                    self.rng.normal(0, 2),       # small rho change
                    self.rng.normal(0, 0.1),     # small phi change
                    self.rng.normal(0, 1),       # small z change
                ])
                candidate = current_cyl + perturbation
            else:
                # Phase 1 (exploration): Sample from good particles
                # Pick a random personal best and use its waypoint
                random_pbest_idx = self.rng.integers(0, self.num_particles)
                donor = self.pbest_positions[random_pbest_idx][uav_index][bottleneck_idx].copy()
                
                # Blend with current
                blend = self.rng.uniform(0.3, 0.7)
                candidate = blend * donor + (1 - blend) * current_cyl
                
                # Add exploration noise
                candidate += np.array([
                    self.rng.normal(0, 5),
                    self.rng.normal(0, 0.3),
                    self.rng.normal(0, 3),
                ])
            
            # Ensure rho >= 0
            candidate[0] = max(0, candidate[0])
            candidate[1] = utils.wrap_angle(candidate[1])
            
            # Test this replacement
            test_path_cyl = [p.copy() for p in self.gbest_position]
            test_path_cyl[uav_index][bottleneck_idx] = candidate
            
            # Convert to Cartesian and evaluate
            test_path_cart = []
            for u, uav_obj in enumerate(self.uavs):
                test_path_cart.append(self._cyl_to_cart_path(
                    test_path_cyl[u], uav_obj.start, uav_obj.goal
                ))
            
            cost, _, _ = self.cost_fn.evaluate_cooperative(test_path_cart)
            
            if cost < best_cost:
                best_cost = cost
                best_replacement = candidate.copy()
        
        # Apply best replacement if it improves
        if best_cost < self.gbest_cost:
            self.gbest_position[uav_index][bottleneck_idx] = best_replacement
            self.gbest_position_cart[uav_index] = self._cyl_to_cart_path(
                self.gbest_position[uav_index], uav.start, uav.goal
            )
            self.gbest_cost = best_cost
    
    def _update_velocity(self, particle_idx, iteration):
        """PSO velocity update in cylindrical coordinates."""
        w = self.w_start - (self.w_start - self.w_end) * iteration / self.max_iterations
        
        for u in range(self.num_uavs):
            r1 = self.rng.random(self.particles[particle_idx][u].shape)
            r2 = self.rng.random(self.particles[particle_idx][u].shape)
            
            cognitive = self.c1 * r1 * (self.pbest_positions[particle_idx][u] - self.particles[particle_idx][u])
            social = self.c2 * r2 * (self.gbest_position[u] - self.particles[particle_idx][u])
            
            self.velocities[particle_idx][u] = (
                w * self.velocities[particle_idx][u] + cognitive + social
            )
            
            # Clamp velocity
            v_max = np.array([10.0, 0.5, 5.0])  # rho, phi, z max velocities
            self.velocities[particle_idx][u] = np.clip(
                self.velocities[particle_idx][u], -v_max, v_max
            )
    
    def _update_position(self, particle_idx):
        """Update particle position in cylindrical coordinates."""
        for u in range(self.num_uavs):
            uav = self.uavs[u]
            
            # Update cylindrical coordinates
            self.particles[particle_idx][u][1:-1] += self.velocities[particle_idx][u][1:-1]
            
            # Enforce constraints
            for i in range(1, self.num_waypoints + 1):
                # rho >= 0
                self.particles[particle_idx][u][i, 0] = max(0, self.particles[particle_idx][u][i, 0])
                # Wrap phi
                self.particles[particle_idx][u][i, 1] = utils.wrap_angle(
                    self.particles[particle_idx][u][i, 1]
                )
            
            # Convert to Cartesian
            self.particles_cart[particle_idx][u] = self._cyl_to_cart_path(
                self.particles[particle_idx][u], uav.start, uav.goal
            )
            
            # Clip Cartesian to bounds
            for i in range(1, self.num_waypoints + 1):
                self.particles_cart[particle_idx][u][i, 0] = np.clip(
                    self.particles_cart[particle_idx][u][i, 0],
                    self.env.bounds['x'][0], self.env.bounds['x'][1]
                )
                self.particles_cart[particle_idx][u][i, 1] = np.clip(
                    self.particles_cart[particle_idx][u][i, 1],
                    self.env.bounds['y'][0], self.env.bounds['y'][1]
                )
                self.particles_cart[particle_idx][u][i, 2] = np.clip(
                    self.particles_cart[particle_idx][u][i, 2],
                    self.env.bounds['z'][0], self.env.bounds['z'][1]
                )
            
            # Update cylindrical from clipped Cartesian
            self.particles[particle_idx][u] = self._cart_to_cyl_path(
                self.particles_cart[particle_idx][u], uav.start, uav.goal
            )
    
    def optimize(self, verbose=True):
        """Run GTCPSO optimization."""
        self._initialize()
        
        for it in range(self.max_iterations):
            # Standard PSO update
            for p in range(self.num_particles):
                self._update_velocity(p, it)
                self._update_position(p)
                
                cost, breakdowns, coll = self.cost_fn.evaluate_cooperative(
                    self.particles_cart[p]
                )
                
                if cost < self.pbest_costs[p]:
                    self.pbest_costs[p] = cost
                    self.pbest_positions[p] = [path.copy() for path in self.particles[p]]
                    self.pbest_positions_cart[p] = [path.copy() for path in self.particles_cart[p]]
                
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest_position = [path.copy() for path in self.particles[p]]
                    self.gbest_position_cart = [path.copy() for path in self.particles_cart[p]]
            
            # Gene Targeting operation
            if self.rng.random() < self.gt_probability:
                for u in range(self.num_uavs):
                    self._gene_targeting(u, it)
            
            self.convergence_history.append(self.gbest_cost)
            
            # Store breakdown
            _, breakdowns, coll = self.cost_fn.evaluate_cooperative(self.gbest_position_cart)
            self.cost_breakdown_history.append({
                'iteration': it,
                'total': self.gbest_cost,
                'breakdowns': breakdowns,
                'collision_penalty': coll
            })
            
            if verbose and (it % 20 == 0 or it == self.max_iterations - 1):
                print(f"  GTCPSO Iter {it:4d}/{self.max_iterations}: Best Cost = {self.gbest_cost:.4f}")
        
        return self.gbest_position_cart, self.gbest_cost
    
    def get_results(self):
        """Return results dict."""
        _, breakdowns, coll = self.cost_fn.evaluate_cooperative(self.gbest_position_cart)
        
        # Get final bottleneck info
        bottleneck_info = []
        for u in range(self.num_uavs):
            idx, costs = self._detect_bottleneck(u)
            bottleneck_info.append({
                'uav': u,
                'bottleneck_idx': idx,
                'wp_costs': costs.tolist()
            })
        
        return {
            'algorithm': 'GTCPSO',
            'best_cost': self.gbest_cost,
            'best_paths': self.gbest_position_cart,
            'convergence': self.convergence_history,
            'cost_breakdowns': breakdowns,
            'collision_penalty': coll,
            'path_lengths': [utils.path_length(p) for p in self.gbest_position_cart],
            'bottleneck_info': bottleneck_info,
            'bottleneck_history': self.bottleneck_history,
        }
