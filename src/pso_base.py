"""
Standard Particle Swarm Optimization (PSO) baseline for UAV path planning.
Operates in Cartesian coordinates without gene targeting.
"""

import numpy as np
from . import utils
from .cost_function import CostFunction


class StandardPSO:
    """
    Standard PSO algorithm for cooperative multi-UAV path planning.
    Serves as the baseline comparison.
    """
    
    def __init__(self, environment, uavs, cost_function,
                 num_particles=50,
                 num_waypoints=10,
                 max_iterations=200,
                 w_start=0.9,        # Inertia weight start
                 w_end=0.4,          # Inertia weight end
                 c1=2.0,             # Cognitive coefficient
                 c2=2.0,             # Social coefficient
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
        self.rng = np.random.default_rng(seed)
        
        self.num_uavs = len(uavs)
        # Each particle: list of paths (one per UAV), each path is (num_waypoints+2, 3)
        self.particles = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_costs = []
        self.gbest_position = None
        self.gbest_cost = float('inf')
        
        self.convergence_history = []
        self.cost_breakdown_history = []
    
    def _initialize(self):
        """Initialize swarm with random feasible paths."""
        self.particles = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_costs = []
        
        for p in range(self.num_particles):
            particle_paths = []
            particle_velocities = []
            
            for uav in self.uavs:
                path = utils.generate_random_path(
                    uav.start, uav.goal, self.num_waypoints,
                    self.env.bounds, self.rng
                )
                particle_paths.append(path.copy())
                
                # Initialize velocity
                vel = np.zeros_like(path)
                scale = np.array([
                    (self.env.bounds['x'][1] - self.env.bounds['x'][0]) * 0.05,
                    (self.env.bounds['y'][1] - self.env.bounds['y'][0]) * 0.05,
                    (self.env.bounds['z'][1] - self.env.bounds['z'][0]) * 0.05,
                ])
                vel[1:-1] = self.rng.uniform(-1, 1, (self.num_waypoints, 3)) * scale
                particle_velocities.append(vel)
            
            self.particles.append(particle_paths)
            self.velocities.append(particle_velocities)
            self.pbest_positions.append([p.copy() for p in particle_paths])
            
            cost, _, _ = self.cost_fn.evaluate_cooperative(particle_paths)
            self.pbest_costs.append(cost)
            
            if cost < self.gbest_cost:
                self.gbest_cost = cost
                self.gbest_position = [p.copy() for p in particle_paths]
    
    def _update_velocity(self, particle_idx, iteration):
        """Standard PSO velocity update with linearly decreasing inertia."""
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
            v_max = np.array([
                (self.env.bounds['x'][1] - self.env.bounds['x'][0]) * 0.1,
                (self.env.bounds['y'][1] - self.env.bounds['y'][0]) * 0.1,
                (self.env.bounds['z'][1] - self.env.bounds['z'][0]) * 0.1,
            ])
            self.velocities[particle_idx][u] = np.clip(
                self.velocities[particle_idx][u], -v_max, v_max
            )
    
    def _update_position(self, particle_idx):
        """Update particle position (only intermediate waypoints)."""
        for u in range(self.num_uavs):
            self.particles[particle_idx][u][1:-1] += self.velocities[particle_idx][u][1:-1]
            
            # Clip to bounds
            for i in range(1, self.num_waypoints + 1):
                self.particles[particle_idx][u][i, 0] = np.clip(
                    self.particles[particle_idx][u][i, 0],
                    self.env.bounds['x'][0], self.env.bounds['x'][1]
                )
                self.particles[particle_idx][u][i, 1] = np.clip(
                    self.particles[particle_idx][u][i, 1],
                    self.env.bounds['y'][0], self.env.bounds['y'][1]
                )
                self.particles[particle_idx][u][i, 2] = np.clip(
                    self.particles[particle_idx][u][i, 2],
                    self.env.bounds['z'][0], self.env.bounds['z'][1]
                )
    
    def optimize(self, verbose=True):
        """Run the optimization."""
        self._initialize()
        
        for it in range(self.max_iterations):
            for p in range(self.num_particles):
                self._update_velocity(p, it)
                self._update_position(p)
                
                cost, breakdowns, coll = self.cost_fn.evaluate_cooperative(self.particles[p])
                
                if cost < self.pbest_costs[p]:
                    self.pbest_costs[p] = cost
                    self.pbest_positions[p] = [path.copy() for path in self.particles[p]]
                
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest_position = [path.copy() for path in self.particles[p]]
            
            self.convergence_history.append(self.gbest_cost)
            
            # Store cost breakdown for gbest
            _, breakdowns, coll = self.cost_fn.evaluate_cooperative(self.gbest_position)
            self.cost_breakdown_history.append({
                'iteration': it,
                'total': self.gbest_cost,
                'breakdowns': breakdowns,
                'collision_penalty': coll
            })
            
            if verbose and (it % 20 == 0 or it == self.max_iterations - 1):
                print(f"  PSO Iter {it:4d}/{self.max_iterations}: Best Cost = {self.gbest_cost:.4f}")
        
        return self.gbest_position, self.gbest_cost
    
    def get_results(self):
        """Return results dict."""
        _, breakdowns, coll = self.cost_fn.evaluate_cooperative(self.gbest_position)
        
        return {
            'algorithm': 'Standard PSO',
            'best_cost': self.gbest_cost,
            'best_paths': self.gbest_position,
            'convergence': self.convergence_history,
            'cost_breakdowns': breakdowns,
            'collision_penalty': coll,
            'path_lengths': [utils.path_length(p) for p in self.gbest_position],
        }
