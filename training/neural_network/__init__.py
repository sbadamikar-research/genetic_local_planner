"""
Neural network training module for GA-based ROS local planner.

This module provides PyTorch model architecture, dataset loaders, and training
utilities for distilling GA-optimized trajectories into a fast neural network.

Components:
- model.py: CNN + MLP architecture (PlannerPolicy)
- dataset.py: Trajectory dataset loader with filtering
"""

from .model import PlannerPolicy, CostmapEncoder, StateEncoder, PolicyHead, create_model
from .dataset import TrajectoryDataset, create_synthetic_data

__all__ = [
    'PlannerPolicy',
    'CostmapEncoder',
    'StateEncoder',
    'PolicyHead',
    'create_model',
    'TrajectoryDataset',
    'create_synthetic_data'
]
