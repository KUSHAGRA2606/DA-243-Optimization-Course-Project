"""
3D Environment model for UAV path planning.
Includes terrain, threats (radar/SAM sites), and no-fly zones.
"""

import numpy as np
import json


class Threat:
    """
    A threat zone modeled as a cylinder with exponential decay intensity.
    Threat cost: strength / max(distance, epsilon)^alpha
    """
    
    def __init__(self, center, radius, strength, alpha=2.0, height_range=None):
        """
        center: (x, y, z) center of threat
        radius: effective radius
        strength: threat intensity
        alpha: decay exponent
        height_range: (z_min, z_max) or None for full height
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.strength = float(strength)
        self.alpha = float(alpha)
        self.height_range = height_range
    
    def cost_at(self, point):
        """Compute threat cost at a given 3D point."""
        point = np.array(point, dtype=float)
        
        # Check height range
        if self.height_range is not None:
            if point[2] < self.height_range[0] or point[2] > self.height_range[1]:
                return 0.0
        
        dist = np.linalg.norm(point[:2] - self.center[:2])
        
        if dist > self.radius * 3:  # Beyond influence range
            return 0.0
        
        # Threat cost with exponential decay
        epsilon = 1.0
        cost = self.strength / (max(dist, epsilon) ** self.alpha)
        return cost
    
    def is_inside(self, point):
        """Check if point is inside threat zone."""
        dist = np.linalg.norm(np.array(point[:2]) - self.center[:2])
        if self.height_range is not None:
            in_height = self.height_range[0] <= point[2] <= self.height_range[1]
            return dist <= self.radius and in_height
        return dist <= self.radius


class NoFlyZone:
    """A no-fly zone modeled as a rectangular prism or cylinder."""
    
    def __init__(self, center, size, shape='box'):
        """
        center: (x, y, z)
        size: (sx, sy, sz) for box or (radius, height) for cylinder
        shape: 'box' or 'cylinder'
        """
        self.center = np.array(center, dtype=float)
        self.size = np.array(size, dtype=float)
        self.shape = shape
    
    def is_inside(self, point):
        """Check if point is inside no-fly zone."""
        point = np.array(point, dtype=float)
        
        if self.shape == 'box':
            half = self.size / 2
            return all(abs(point[i] - self.center[i]) <= half[i] for i in range(3))
        else:  # cylinder
            dist_xy = np.linalg.norm(point[:2] - self.center[:2])
            dz = abs(point[2] - self.center[2])
            return dist_xy <= self.size[0] and dz <= self.size[1] / 2
    
    def penalty_at(self, point):
        """Return penalty if inside, 0 otherwise."""
        if self.is_inside(point):
            return 1000.0  # Large penalty
        return 0.0


class Terrain:
    """
    Terrain model using a heightmap grid.
    """
    
    def __init__(self, x_range, y_range, resolution=1.0, terrain_type='flat'):
        """
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)
        resolution: grid spacing
        terrain_type: 'flat', 'hills', 'mountains', 'rugged'
        """
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.terrain_type = terrain_type
        
        nx = int((x_range[1] - x_range[0]) / resolution) + 1
        ny = int((y_range[1] - y_range[0]) / resolution) + 1
        
        self.nx = nx
        self.ny = ny
        self.heightmap = self._generate_heightmap(nx, ny, terrain_type)
    
    def _generate_heightmap(self, nx, ny, terrain_type):
        """Generate terrain heightmap."""
        x = np.linspace(self.x_range[0], self.x_range[1], nx)
        y = np.linspace(self.y_range[0], self.y_range[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if terrain_type == 'flat':
            return np.zeros((nx, ny))
        
        elif terrain_type == 'hills':
            # Gentle rolling hills
            h = (5.0 * np.sin(X * 0.05) * np.cos(Y * 0.04) +
                 3.0 * np.sin(X * 0.08 + 1.0) * np.sin(Y * 0.06 + 0.5))
            return np.maximum(h, 0)
        
        elif terrain_type == 'mountains':
            # More pronounced peaks
            h = (15.0 * np.sin(X * 0.04) * np.cos(Y * 0.03) +
                 10.0 * np.exp(-((X - (self.x_range[0] + self.x_range[1])/2)**2 +
                                  (Y - (self.y_range[0] + self.y_range[1])/2)**2) / 2000) +
                 5.0 * np.sin(X * 0.1) * np.sin(Y * 0.08))
            return np.maximum(h, 0)
        
        elif terrain_type == 'rugged':
            # Very rough terrain
            rng = np.random.default_rng(42)
            h = (20.0 * np.sin(X * 0.03) * np.cos(Y * 0.025) +
                 12.0 * np.sin(X * 0.07 + 2) * np.sin(Y * 0.05 + 1) +
                 8.0 * np.cos(X * 0.12) * np.sin(Y * 0.1) +
                 15.0 * np.exp(-((X - self.x_range[1]*0.3)**2 +
                                  (Y - self.y_range[1]*0.6)**2) / 800) +
                 10.0 * np.exp(-((X - self.x_range[1]*0.7)**2 +
                                  (Y - self.y_range[1]*0.3)**2) / 600))
            return np.maximum(h, 0)
        
        return np.zeros((nx, ny))
    
    def height_at(self, x, y):
        """Get terrain height at (x, y) with bilinear interpolation."""
        # Convert to grid indices
        ix = (x - self.x_range[0]) / self.resolution
        iy = (y - self.y_range[0]) / self.resolution
        
        ix = np.clip(ix, 0, self.nx - 1.001)
        iy = np.clip(iy, 0, self.ny - 1.001)
        
        ix0 = int(np.floor(ix))
        iy0 = int(np.floor(iy))
        ix1 = min(ix0 + 1, self.nx - 1)
        iy1 = min(iy0 + 1, self.ny - 1)
        
        fx = ix - ix0
        fy = iy - iy0
        
        h = (self.heightmap[ix0, iy0] * (1 - fx) * (1 - fy) +
             self.heightmap[ix1, iy0] * fx * (1 - fy) +
             self.heightmap[ix0, iy1] * (1 - fx) * fy +
             self.heightmap[ix1, iy1] * fx * fy)
        
        return h


class Environment:
    """
    Complete 3D environment for UAV path planning.
    """
    
    def __init__(self, bounds, terrain_type='flat'):
        """
        bounds: dict with 'x': (min, max), 'y': (min, max), 'z': (min, max)
        """
        self.bounds = bounds
        self.terrain = Terrain(bounds['x'], bounds['y'], 
                               resolution=max(1.0, (bounds['x'][1] - bounds['x'][0]) / 100),
                               terrain_type=terrain_type)
        self.threats = []
        self.no_fly_zones = []
        self.min_safe_altitude = 5.0  # Above terrain
    
    def add_threat(self, center, radius, strength, alpha=2.0, height_range=None):
        """Add a threat to the environment."""
        self.threats.append(Threat(center, radius, strength, alpha, height_range))
    
    def add_no_fly_zone(self, center, size, shape='cylinder'):
        """Add a no-fly zone."""
        self.no_fly_zones.append(NoFlyZone(center, size, shape))
    
    def threat_cost_at(self, point):
        """Total threat cost at a point."""
        return sum(t.cost_at(point) for t in self.threats)
    
    def nfz_penalty_at(self, point):
        """Total no-fly zone penalty at a point."""
        return sum(nfz.penalty_at(point) for nfz in self.no_fly_zones)
    
    def terrain_clearance(self, point):
        """Check terrain clearance. Returns clearance (positive = safe)."""
        terrain_h = self.terrain.height_at(point[0], point[1])
        return point[2] - terrain_h - self.min_safe_altitude
    
    def is_valid_point(self, point):
        """Check if a point is within bounds and not in any no-fly zone."""
        # Check bounds
        if not (self.bounds['x'][0] <= point[0] <= self.bounds['x'][1] and
                self.bounds['y'][0] <= point[1] <= self.bounds['y'][1] and
                self.bounds['z'][0] <= point[2] <= self.bounds['z'][1]):
            return False
        
        # Check terrain clearance
        if self.terrain_clearance(point) < 0:
            return False
        
        # Check no-fly zones
        for nfz in self.no_fly_zones:
            if nfz.is_inside(point):
                return False
        
        return True
    
    def to_dict(self):
        """Serialize environment to dict."""
        return {
            'bounds': self.bounds,
            'terrain_type': self.terrain.terrain_type,
            'threats': [
                {
                    'center': t.center.tolist(),
                    'radius': t.radius,
                    'strength': t.strength,
                    'alpha': t.alpha,
                    'height_range': t.height_range
                }
                for t in self.threats
            ],
            'no_fly_zones': [
                {
                    'center': nfz.center.tolist(),
                    'size': nfz.size.tolist(),
                    'shape': nfz.shape
                }
                for nfz in self.no_fly_zones
            ]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Deserialize environment from dict."""
        env = cls(data['bounds'], data.get('terrain_type', 'flat'))
        for t in data.get('threats', []):
            env.add_threat(t['center'], t['radius'], t['strength'],
                          t.get('alpha', 2.0), t.get('height_range'))
        for nfz in data.get('no_fly_zones', []):
            env.add_no_fly_zone(nfz['center'], nfz['size'], nfz.get('shape', 'cylinder'))
        return env
