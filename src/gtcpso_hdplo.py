"""
GTCPSO + HDPLO Improvements: Enhanced cooperative UAV path planning.

Inherits GTCPSO's cylindrical coordinate representation and adds:
1. Adaptive Crossover Mechanism - prevents premature convergence
2. CMA-ES Local Refinement - replaces simple GT local fix
3. Direction-Guided Search - velocity update with direction prediction
4. Diversity Control - monitors and maintains population diversity
"""

import numpy as np
from . import utils
from .gtcpso import GTCPSO
from .cost_function import CostFunction


class GTCPSO_HDPLO(GTCPSO):
    """
    GTCPSO enhanced with HDPLO improvements for superior path planning.
    """
    
    def __init__(self, environment, uavs, cost_function,
                 num_particles=50,
                 num_waypoints=10,
                 max_iterations=200,
                 w_start=0.9,
                 w_end=0.4,
                 c1=2.0,
                 c2=2.0,
                 gt_probability=0.3,
                 gt_phase_switch=0.5,
                 # HDPLO-specific parameters
                 crossover_rate_init=0.5,
                 crossover_rate_min=0.1,
                 crossover_rate_max=0.9,
                 diversity_threshold=0.1,
                 cma_sigma_init=3.0,
                 cma_population=10,
                 cma_iterations=15,
                 direction_weight=0.5,
                 direction_history_len=5,
                 seed=None):
        
        super().__init__(
            environment, uavs, cost_function,
            num_particles, num_waypoints, max_iterations,
            w_start, w_end, c1, c2,
            gt_probability, gt_phase_switch, seed
        )
        
        # Adaptive Crossover parameters
        self.crossover_rate = crossover_rate_init
        self.crossover_rate_min = crossover_rate_min
        self.crossover_rate_max = crossover_rate_max
        
        # Diversity Control
        self.diversity_threshold = diversity_threshold
        self.diversity_history = []
        
        # CMA-ES parameters
        self.cma_sigma_init = cma_sigma_init
        self.cma_population = cma_population
        self.cma_iterations = cma_iterations
        
        # Direction-Guided Search
        self.direction_weight = direction_weight
        self.direction_history_len = direction_history_len
        self.gbest_trajectory = []  # History of gbest positions for direction prediction
        
        # Tracking
        self.diversity_values = []
        self.crossover_rates = []
        self.cma_improvements = []
    
    def _compute_diversity(self):
        """
        Compute population diversity as average pairwise distance 
        between particles (normalized).
        """
        if len(self.particles) < 2:
            return 1.0
        
        total_dist = 0.0
        count = 0
        
        # Sample pairs to keep computation manageable
        indices = list(range(min(self.num_particles, 20)))
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                for u in range(self.num_uavs):
                    diff = (self.particles[indices[i]][u][1:-1] - 
                            self.particles[indices[j]][u][1:-1])
                    total_dist += np.mean(np.abs(diff))
                    count += 1
        
        if count == 0:
            return 0.0
        
        diversity = total_dist / count
        
        # Normalize by search space size
        search_range = max(
            self.env.bounds['x'][1] - self.env.bounds['x'][0],
            self.env.bounds['y'][1] - self.env.bounds['y'][0],
            self.env.bounds['z'][1] - self.env.bounds['z'][0]
        )
        
        return diversity / max(search_range * 0.1, 1e-10)
    
    def _adaptive_crossover(self, iteration):
        """
        HDPLO Improvement 1: Adaptive Crossover Mechanism.
        
        Performs crossover between top particles to generate new solutions.
        Crossover rate adapts based on population diversity and convergence progress.
        """
        diversity = self._compute_diversity()
        self.diversity_values.append(diversity)
        
        # Adapt crossover rate based on diversity
        if diversity < self.diversity_threshold:
            # Low diversity → increase crossover to boost exploration
            self.crossover_rate = min(
                self.crossover_rate * 1.2, self.crossover_rate_max
            )
        else:
            # Good diversity → decrease crossover for more exploitation
            self.crossover_rate = max(
                self.crossover_rate * 0.95, self.crossover_rate_min
            )
        
        self.crossover_rates.append(self.crossover_rate)
        
        # Rank particles by cost
        costs = []
        for p in range(self.num_particles):
            cost, _, _ = self.cost_fn.evaluate_cooperative(self.particles_cart[p])
            costs.append(cost)
        
        sorted_indices = np.argsort(costs)
        
        # Perform crossover on bottom-half particles using top-half parents
        half = self.num_particles // 2
        
        for p_idx in range(half, self.num_particles):
            if self.rng.random() > self.crossover_rate:
                continue
            
            # Select two parents from top half
            p1_idx = sorted_indices[self.rng.integers(0, half)]
            p2_idx = sorted_indices[self.rng.integers(0, half)]
            
            child_idx = sorted_indices[p_idx]
            
            for u in range(self.num_uavs):
                for wp in range(1, self.num_waypoints + 1):
                    # Uniform crossover with adaptive blending
                    if self.rng.random() < 0.5:
                        alpha = self.rng.uniform(0.3, 0.7)
                        self.particles[child_idx][u][wp] = (
                            alpha * self.particles[p1_idx][u][wp] +
                            (1 - alpha) * self.particles[p2_idx][u][wp]
                        )
                    
                    # Add mutation proportional to diversity need
                    if diversity < self.diversity_threshold:
                        mutation_scale = np.array([3.0, 0.2, 2.0])
                        self.particles[child_idx][u][wp] += (
                            self.rng.normal(0, 1, 3) * mutation_scale
                        )
                
                # Ensure rho >= 0 and wrap phi
                self.particles[child_idx][u][1:-1, 0] = np.maximum(
                    0, self.particles[child_idx][u][1:-1, 0]
                )
                for wp in range(1, self.num_waypoints + 1):
                    self.particles[child_idx][u][wp, 1] = utils.wrap_angle(
                        self.particles[child_idx][u][wp, 1]
                    )
                
                # Update Cartesian
                self.particles_cart[child_idx][u] = self._cyl_to_cart_path(
                    self.particles[child_idx][u],
                    self.uavs[u].start, self.uavs[u].goal
                )
    
    def _cma_es_local_refinement(self, uav_index, bottleneck_idx):
        """
        HDPLO Improvement 2: CMA-ES Local Refinement.
        
        Instead of simple waypoint replacement (GT approach), uses CMA-ES 
        to optimize a window of waypoints around the bottleneck.
        This provides much stronger local optimization.
        """
        uav = self.uavs[uav_index]
        
        # Define window around bottleneck (±1 waypoints)
        window_start = max(1, bottleneck_idx - 1)
        window_end = min(self.num_waypoints, bottleneck_idx + 1)
        window_size = window_end - window_start + 1
        dim = window_size * 3  # Each waypoint has 3 cylindrical coords
        
        # Initial mean = current waypoints in window
        mean = np.zeros(dim)
        for i, wp_idx in enumerate(range(window_start, window_end + 1)):
            mean[i*3:(i+1)*3] = self.gbest_position[uav_index][wp_idx]
        
        # CMA-ES simplified (without full covariance adaptation for speed)
        sigma = self.cma_sigma_init
        best_solution = mean.copy()
        best_cost = self.gbest_cost
        
        # Covariance matrix (start as identity, adapt diagonally)
        cov_diag = np.ones(dim) * sigma**2
        # Different scales for rho, phi, z
        for i in range(window_size):
            cov_diag[i*3 + 0] *= 4.0   # rho
            cov_diag[i*3 + 1] *= 0.1   # phi (smaller scale)
            cov_diag[i*3 + 2] *= 2.0   # z
        
        for gen in range(self.cma_iterations):
            # Generate population
            candidates = []
            candidate_costs = []
            
            for _ in range(self.cma_population):
                # Sample from diagonal gaussian
                sample = mean + self.rng.normal(0, 1, dim) * np.sqrt(cov_diag)
                
                # Apply to test path
                test_cyl = [p.copy() for p in self.gbest_position]
                for i, wp_idx in enumerate(range(window_start, window_end + 1)):
                    test_cyl[uav_index][wp_idx] = sample[i*3:(i+1)*3]
                    test_cyl[uav_index][wp_idx, 0] = max(0, test_cyl[uav_index][wp_idx, 0])
                    test_cyl[uav_index][wp_idx, 1] = utils.wrap_angle(test_cyl[uav_index][wp_idx, 1])
                
                # Evaluate
                test_cart = []
                for u, uav_obj in enumerate(self.uavs):
                    test_cart.append(self._cyl_to_cart_path(
                        test_cyl[u], uav_obj.start, uav_obj.goal
                    ))
                
                cost, _, _ = self.cost_fn.evaluate_cooperative(test_cart)
                candidates.append(sample)
                candidate_costs.append(cost)
            
            # Sort by cost
            sorted_idx = np.argsort(candidate_costs)
            
            # Update mean from best half
            mu = max(2, self.cma_population // 2)
            selected = np.array([candidates[sorted_idx[i]] for i in range(mu)])
            
            # Weighted recombination (rank-based weights)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            
            new_mean = np.sum(selected * weights[:, np.newaxis], axis=0)
            
            # Adapt step size (simplified)
            if candidate_costs[sorted_idx[0]] < best_cost:
                best_cost = candidate_costs[sorted_idx[0]]
                best_solution = candidates[sorted_idx[0]].copy()
                sigma *= 1.05  # Increase step if improving
            else:
                sigma *= 0.85  # Decrease step if not improving
            
            mean = new_mean
            cov_diag = np.var(selected, axis=0) + 1e-10
        
        # Apply best solution if it improves
        improvement = self.gbest_cost - best_cost
        self.cma_improvements.append(improvement)
        
        if best_cost < self.gbest_cost:
            for i, wp_idx in enumerate(range(window_start, window_end + 1)):
                self.gbest_position[uav_index][wp_idx] = best_solution[i*3:(i+1)*3]
                self.gbest_position[uav_index][wp_idx, 0] = max(
                    0, self.gbest_position[uav_index][wp_idx, 0]
                )
                self.gbest_position[uav_index][wp_idx, 1] = utils.wrap_angle(
                    self.gbest_position[uav_index][wp_idx, 1]
                )
            
            self.gbest_position_cart[uav_index] = self._cyl_to_cart_path(
                self.gbest_position[uav_index], uav.start, uav.goal
            )
            self.gbest_cost = best_cost
            return True
        
        return False
    
    def _compute_direction_prediction(self):
        """
        HDPLO Improvement 3: Direction-Guided Search.
        
        Predict the direction of improvement from historical gbest trajectory.
        Returns a direction vector that particles can use to accelerate convergence.
        """
        if len(self.gbest_trajectory) < 2:
            return None
        
        # Use recent history to estimate direction
        history_len = min(self.direction_history_len, len(self.gbest_trajectory))
        recent = self.gbest_trajectory[-history_len:]
        
        # Compute weighted direction from old to new positions
        directions = []
        for u in range(self.num_uavs):
            dir_u = np.zeros_like(recent[-1][u])
            total_weight = 0
            
            for i in range(1, len(recent)):
                weight = i / len(recent)  # More recent = higher weight
                diff = recent[i][u] - recent[i-1][u]
                dir_u += weight * diff
                total_weight += weight
            
            if total_weight > 0:
                dir_u /= total_weight
            
            directions.append(dir_u)
        
        return directions
    
    def _update_velocity_with_direction(self, particle_idx, iteration, direction_pred):
        """
        Enhanced velocity update with direction prediction term.
        
        V_new = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X) + c3*r3*direction
        """
        w = self.w_start - (self.w_start - self.w_end) * iteration / self.max_iterations
        
        for u in range(self.num_uavs):
            r1 = self.rng.random(self.particles[particle_idx][u].shape)
            r2 = self.rng.random(self.particles[particle_idx][u].shape)
            
            cognitive = self.c1 * r1 * (
                self.pbest_positions[particle_idx][u] - self.particles[particle_idx][u]
            )
            social = self.c2 * r2 * (
                self.gbest_position[u] - self.particles[particle_idx][u]
            )
            
            # Direction prediction component
            direction_term = np.zeros_like(self.particles[particle_idx][u])
            if direction_pred is not None:
                r3 = self.rng.random(self.particles[particle_idx][u].shape)
                direction_term = self.direction_weight * r3 * direction_pred[u]
            
            self.velocities[particle_idx][u] = (
                w * self.velocities[particle_idx][u] + 
                cognitive + social + direction_term
            )
            
            # Clamp velocity
            v_max = np.array([10.0, 0.5, 5.0])
            self.velocities[particle_idx][u] = np.clip(
                self.velocities[particle_idx][u], -v_max, v_max
            )
    
    def _diversity_boost(self):
        """
        HDPLO Improvement 4: Diversity Control.
        
        When diversity drops too low, reinitialize worst particles
        with random perturbation to prevent swarm collapse.
        """
        diversity = self._compute_diversity()
        
        if diversity >= self.diversity_threshold:
            return
        
        # Reinitialize bottom 20% of particles
        costs = []
        for p in range(self.num_particles):
            cost, _, _ = self.cost_fn.evaluate_cooperative(self.particles_cart[p])
            costs.append(cost)
        
        sorted_indices = np.argsort(costs)
        num_reinit = max(2, self.num_particles // 5)
        
        for rank in range(self.num_particles - num_reinit, self.num_particles):
            p_idx = sorted_indices[rank]
            
            for u, uav in enumerate(self.uavs):
                # Create new random path
                cart_path = utils.generate_random_path(
                    uav.start, uav.goal, self.num_waypoints,
                    self.env.bounds, self.rng
                )
                
                # Blend with gbest to keep some good information
                blend = self.rng.uniform(0.2, 0.5)
                cart_path[1:-1] = blend * cart_path[1:-1] + (1 - blend) * self.gbest_position_cart[u][1:-1]
                
                cyl_path = self._cart_to_cyl_path(cart_path, uav.start, uav.goal)
                
                self.particles[p_idx][u] = cyl_path
                self.particles_cart[p_idx][u] = cart_path
                
                # Reset velocity
                vel = np.zeros_like(cyl_path)
                vel[1:-1, 0] = self.rng.uniform(-5, 5, self.num_waypoints)
                vel[1:-1, 1] = self.rng.uniform(-0.3, 0.3, self.num_waypoints)
                vel[1:-1, 2] = self.rng.uniform(-3, 3, self.num_waypoints)
                self.velocities[p_idx][u] = vel
    
    def _gene_targeting_with_cma(self, uav_index, iteration):
        """
        Enhanced Gene Targeting using CMA-ES instead of simple replacement.
        """
        bottleneck_idx, wp_costs = self._detect_bottleneck(uav_index)
        
        self.bottleneck_history.append({
            'iteration': iteration,
            'uav': uav_index,
            'bottleneck_idx': bottleneck_idx,
            'bottleneck_cost': wp_costs[bottleneck_idx],
            'all_wp_costs': wp_costs.tolist(),
            'method': 'CMA-ES'
        })
        
        # Use CMA-ES for local refinement around bottleneck
        improved = self._cma_es_local_refinement(uav_index, bottleneck_idx)
        
        return improved
    
    def optimize(self, verbose=True):
        """Run GTCPSO + HDPLO optimization."""
        self._initialize()
        
        for it in range(self.max_iterations):
            # Compute direction prediction
            direction_pred = self._compute_direction_prediction()
            
            # Standard PSO update with direction guidance
            for p in range(self.num_particles):
                self._update_velocity_with_direction(p, it, direction_pred)
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
            
            # Store gbest trajectory for direction prediction
            self.gbest_trajectory.append([p.copy() for p in self.gbest_position])
            if len(self.gbest_trajectory) > self.direction_history_len + 2:
                self.gbest_trajectory.pop(0)
            
            # HDPLO Improvement 1: Adaptive Crossover
            if it % 5 == 0:  # Every 5 iterations
                self._adaptive_crossover(it)
            
            # HDPLO Improvement 2+: Enhanced Gene Targeting with CMA-ES
            if self.rng.random() < self.gt_probability:
                for u in range(self.num_uavs):
                    self._gene_targeting_with_cma(u, it)
            
            # HDPLO Improvement 4: Diversity Control
            if it % 10 == 0:
                self._diversity_boost()
            
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
                div = self.diversity_values[-1] if self.diversity_values else 0
                print(f"  HDPLO  Iter {it:4d}/{self.max_iterations}: "
                      f"Best Cost = {self.gbest_cost:.4f} | "
                      f"Diversity = {div:.4f}")
        
        return self.gbest_position_cart, self.gbest_cost
    
    def get_results(self):
        """Return results dict with HDPLO-specific metrics."""
        base_results = super().get_results()
        base_results['algorithm'] = 'GTCPSO + HDPLO'
        base_results['diversity_history'] = self.diversity_values
        base_results['crossover_rate_history'] = self.crossover_rates
        base_results['cma_improvements'] = self.cma_improvements
        return base_results
