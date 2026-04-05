"""
UAV model with dynamic constraints for path planning.
"""

import numpy as np


class UAV:
    """
    Unmanned Aerial Vehicle model with physical constraints.
    """
    
    def __init__(self, uav_id, start, goal, 
                 max_turn_angle=np.pi/3,       # 60 degrees max turn
                 max_climb_angle=np.pi/6,      # 30 degrees max climb/dive
                 min_speed=10.0,
                 max_speed=50.0,
                 safe_distance=10.0,           # Inter-UAV minimum distance
                 fuel_capacity=1000.0):
        """
        uav_id: unique identifier
        start: (x, y, z) start position
        goal: (x, y, z) goal position
        """
        self.uav_id = uav_id
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.max_turn_angle = max_turn_angle
        self.max_climb_angle = max_climb_angle
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.safe_distance = safe_distance
        self.fuel_capacity = fuel_capacity
    
    def to_dict(self):
        return {
            'uav_id': self.uav_id,
            'start': self.start.tolist(),
            'goal': self.goal.tolist(),
            'max_turn_angle': self.max_turn_angle,
            'max_climb_angle': self.max_climb_angle,
            'min_speed': self.min_speed,
            'max_speed': self.max_speed,
            'safe_distance': self.safe_distance,
            'fuel_capacity': self.fuel_capacity
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            uav_id=data['uav_id'],
            start=data['start'],
            goal=data['goal'],
            max_turn_angle=data.get('max_turn_angle', np.pi/3),
            max_climb_angle=data.get('max_climb_angle', np.pi/6),
            min_speed=data.get('min_speed', 10.0),
            max_speed=data.get('max_speed', 50.0),
            safe_distance=data.get('safe_distance', 10.0),
            fuel_capacity=data.get('fuel_capacity', 1000.0)
        )
