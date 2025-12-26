"""
Lightweight Python simulator for GA training.

This module provides a ROS-independent simulation environment for training
genetic algorithms. It includes:
- Costmap generation and inflation
- Robot dynamics simulation
- Collision detection
- Navigation environment wrapper
"""

from .costmap import Costmap, generate_random_costmap
from .robot_model import RobotState, RobotModel
from .collision_checker import CollisionChecker
from .environment import NavigationEnvironment

__all__ = [
    'Costmap',
    'generate_random_costmap',
    'RobotState',
    'RobotModel',
    'CollisionChecker',
    'NavigationEnvironment'
]
