"""
Continuous Online Receding Horizon Control (RHC) Manager for D-GTCPSO.
"""

import numpy as np
import time

class RHCManager:
    def __init__(self, environment, cost_function_class, algorithm_class, algorithm_params, dt=1.0):
        self.env = environment
        self.cost_function_class = cost_function_class
        self.algorithm_class = algorithm_class
        self.algorithm_params = algorithm_params
        self.dt = dt
        
        self.active_uavs = []
        self.events = []
        self.current_time = 0.0
        self.executed_paths = {} # Dict uav_id -> list of positions
        self.previous_best_paths = None
        
    def add_event(self, event_time, event_type, data):
        """
        data: For UAV_ARRIVE -> {'uav': UAV()}
        """
        self.events.append({'time': event_time, 'type': event_type, 'data': data})
        # Sort events by time
        self.events = sorted(self.events, key=lambda x: x['time'])
        
    def _process_events(self):
        new_uav_arrived = False
        while self.events and self.events[0]['time'] <= self.current_time:
            ev = self.events.pop(0)
            if ev['type'] == 'UAV_ARRIVE':
                uav = ev['data']['uav']
                self.active_uavs.append(uav)
                self.executed_paths[uav.uav_id] = [uav.start.copy()]
                new_uav_arrived = True
            elif ev['type'] == 'THREAT_ARRIVE':
                threat = ev['data']['threat']
                self.env.add_threat(threat.center, threat.radius, threat.strength)
                print(f"[{self.current_time:04.1f}s] Event: Threat arrived at {threat.center}")
        return new_uav_arrived

    def _prepare_warm_start(self, new_uav_arrived):
        if not self.previous_best_paths:
            return None
            
        warm_start_population = []
        num_particles = self.algorithm_params.get('num_particles', 50)
        
        for p in range(num_particles):
            particle_paths = []
            for i, uav in enumerate(self.active_uavs):
                if i < len(self.previous_best_paths) and not new_uav_arrived:
                    old_path = self.previous_best_paths[i].copy()
                    
                    # Shift previous path forward
                    shifted_path = np.zeros_like(old_path)
                    
                    # If we reached the end of the old path, just stay there
                    if len(old_path) > 1:
                        shifted_path[0] = old_path[1]
                        uav.start = shifted_path[0].copy()
                        for w in range(1, len(old_path) - 1):
                            shifted_path[w] = old_path[w+1]
                        shifted_path[-1] = old_path[-1]
                    else:
                        shifted_path = old_path
                    
                    particle_paths.append(shifted_path)
                else:
                    break # Re-init fully if structure changed
            if len(particle_paths) == len(self.active_uavs) and not new_uav_arrived:
               warm_start_population.append(particle_paths)
        
        if not warm_start_population: return None
        # Add random scatter to warm start so we don't collapse diversity
        for p_idx in range(1, len(warm_start_population)):
             for u in range(len(warm_start_population[p_idx])):
                 noise = np.random.normal(0, 1.0, size=warm_start_population[p_idx][u].shape)
                 noise[0] = 0 # Don't shift start
                 noise[-1] = 0 # Don't shift end
                 warm_start_population[p_idx][u] += noise

        return warm_start_population

    def run_simulation(self, max_time=100.0, goal_tolerance=5.0):
        print(f"Starting D-GTCPSO RHC Simulation (Max Time: {max_time}s)...")
        while self.current_time <= max_time:
            # 1. Process events
            strukt_change = self._process_events()
            
            if not self.active_uavs:
                self.current_time += self.dt
                continue

            # Check for generic completions
            all_arrived = True
            for uav in self.active_uavs:
                if np.linalg.norm(uav.start - uav.goal) > goal_tolerance:
                    all_arrived = False
                    break
            
            if all_arrived:
                # Goal marker visualization fix: force literal target coordinates onto executed paths 
                for uav in self.active_uavs:
                    if len(self.executed_paths[uav.uav_id]) > 0:
                        if np.linalg.norm(self.executed_paths[uav.uav_id][-1] - uav.goal) > 1e-4:
                            self.executed_paths[uav.uav_id].append(uav.goal.copy())
                            
                print(f"[{self.current_time:04.1f}s] All UAVs reached their destinations. Ending early.")
                break
                
            # 2. Re-instantiate optimization components with Active UAVs
            cost_fn = self.cost_function_class(self.env, self.active_uavs)
            
            warm_start_population = None 
            if not strukt_change and self.previous_best_paths:
                warm_start_population = self._prepare_warm_start(False)
            
            algo = self.algorithm_class(
                environment=self.env,
                uavs=self.active_uavs,
                cost_function=cost_fn,
                **self.algorithm_params
            )
            
            # 3. Optimize horizon
            prev_max_iters = algo.max_iterations
            if warm_start_population:
                # Fast online refinement (Warm Started)
                algo.max_iterations = max(5, prev_max_iters // 4) 
            
            best_paths, best_cost = algo.optimize(
                warm_start_particles=warm_start_population, 
                verbose=False
            )
            algo.max_iterations = prev_max_iters
            self.previous_best_paths = best_paths
            
            # 4. Execute next step and update UAV start positions
            for i, uav in enumerate(self.active_uavs):
                path = best_paths[i]
                if len(path) > 1 and np.linalg.norm(uav.start - uav.goal) > goal_tolerance:
                    # Constant velocity interpolation
                    speed = uav.min_speed
                    distance_to_travel = speed * self.dt
                    current_pos = uav.start.copy()
                    next_pos = uav.start.copy()
                    
                    for wp_idx in range(len(path) - 1):
                        seg_start = current_pos if wp_idx == 0 else path[wp_idx]
                        seg_end = path[wp_idx + 1]
                        seg_vec = seg_end - seg_start
                        seg_len = np.linalg.norm(seg_vec)
                        
                        if seg_len < 1e-6:
                            continue
                            
                        if distance_to_travel <= seg_len:
                            dir_vec = seg_vec / seg_len
                            next_pos = seg_start + dir_vec * distance_to_travel
                            distance_to_travel = 0
                            break
                        else:
                            distance_to_travel -= seg_len
                            current_pos = seg_end
                            
                    if distance_to_travel > 0:
                        next_pos = path[-1].copy()
                        
                    uav.start = next_pos.copy()
                    self.executed_paths[uav.uav_id].append(next_pos.copy())
            
            # 5. Advance time
            print(f"[{self.current_time:04.1f}s] RHC Horizon optimized. Gbest Cost: {best_cost:.2f}")
            self.current_time += self.dt

        return self.executed_paths
