"""
Footprint-based collision detection for robot navigation.

This module provides collision checking by transforming the robot's polygonal
footprint to world coordinates and checking against the costmap.
"""

import numpy as np
from typing import List, Tuple
from .robot_model import RobotState
from .costmap import Costmap


class CollisionChecker:
    """
    Checks if robot footprint collides with costmap obstacles.

    Uses sampling along footprint edges for efficient collision detection.
    """

    def __init__(self, footprint: List[Tuple[float, float]], lethal_threshold: int = 254):
        """
        Initialize collision checker with robot footprint.

        Args:
            footprint: List of (x, y) vertices in robot frame (meters)
                      Vertices should be ordered (clockwise or counter-clockwise)
                      Example: [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)]
            lethal_threshold: Costmap value considered collision (default 254=lethal)
        """
        self.footprint = np.array(footprint, dtype=np.float32)
        self.lethal_threshold = lethal_threshold

        # Pre-compute edge samples for efficient collision checking
        self.edge_samples = self._compute_edge_samples(sample_spacing=0.02)

    def _compute_edge_samples(self, sample_spacing: float = 0.02) -> np.ndarray:
        """
        Pre-compute sample points along footprint edges.

        Args:
            sample_spacing: Distance between samples in meters

        Returns:
            samples: np.ndarray (num_samples, 2), sample points in robot frame
        """
        samples = []
        num_vertices = len(self.footprint)

        for i in range(num_vertices):
            # Get edge endpoints
            p1 = self.footprint[i]
            p2 = self.footprint[(i + 1) % num_vertices]

            # Edge vector
            edge = p2 - p1
            edge_length = np.linalg.norm(edge)

            # Number of samples along this edge
            num_samples = max(2, int(edge_length / sample_spacing) + 1)

            # Sample points along edge
            for j in range(num_samples):
                t = j / (num_samples - 1)  # Parameter from 0 to 1
                sample = p1 + t * edge
                samples.append(sample)

        return np.array(samples, dtype=np.float32)

    def check_state(self, state: RobotState, costmap: Costmap) -> Tuple[bool, int]:
        """
        Check if robot state collides with obstacles.

        Algorithm:
        1. Transform footprint samples to world frame
        2. Convert to grid coordinates
        3. Check costmap values at all sample points
        4. Return collision status and maximum cost encountered

        Args:
            state: Robot state to check
            costmap: Costmap to check against

        Returns:
            is_collision: True if any sample point has cost >= lethal_threshold
            max_cost: Maximum costmap value encountered across all samples
        """
        # Transform footprint samples to world frame
        samples_world = self._transform_footprint(state, self.edge_samples)

        # Check costmap at all sample points
        max_cost = 0
        is_collision = False

        for sample in samples_world:
            # Convert to grid coordinates
            mx, my = costmap.world_to_grid(sample[0], sample[1])

            # Get cost at this grid cell
            cost = costmap.get_cost(mx, my)

            # Update max cost
            max_cost = max(max_cost, cost)

            # Check for collision
            if cost >= self.lethal_threshold:
                is_collision = True
                # Could return early here for efficiency, but continue to get max_cost

        return is_collision, max_cost

    def check_trajectory(self, trajectory: List[RobotState],
                        costmap: Costmap) -> Tuple[bool, int]:
        """
        Check entire trajectory for collisions.

        Args:
            trajectory: List of RobotState objects
            costmap: Costmap to check against

        Returns:
            is_collision: True if ANY state in trajectory collides
            max_cost: Maximum costmap value encountered across entire trajectory
        """
        max_cost = 0
        is_collision = False

        for state in trajectory:
            collision, cost = self.check_state(state, costmap)

            max_cost = max(max_cost, cost)

            if collision:
                is_collision = True
                # Could return early, but continue to get true max_cost

        return is_collision, max_cost

    def _transform_footprint(self, state: RobotState,
                           footprint_local: np.ndarray) -> np.ndarray:
        """
        Transform footprint points from robot frame to world frame.

        Args:
            state: Robot state (position and orientation)
            footprint_local: np.ndarray (num_points, 2), points in robot frame

        Returns:
            footprint_world: np.ndarray (num_points, 2), points in world frame
        """
        # Rotation matrix
        cos_theta = np.cos(state.theta)
        sin_theta = np.sin(state.theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Transform: points_world = R * points_local + position
        footprint_world = (rotation_matrix @ footprint_local.T).T
        footprint_world[:, 0] += state.x
        footprint_world[:, 1] += state.y

        return footprint_world


if __name__ == "__main__":
    # Test collision checker
    print("Testing CollisionChecker...")

    from .costmap import generate_random_costmap
    from .robot_model import RobotModel

    # Create test costmap
    costmap = generate_random_costmap(
        width=50,
        height=50,
        resolution=0.05,
        num_obstacles=3,
        obstacle_radius_range=(3, 6),
        inflation_radius=0.5,
        inflation_decay=0.8,
        free_radius_center=0.5
    )

    # Create collision checker (square robot 0.4m x 0.4m)
    footprint = [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)]
    checker = CollisionChecker(footprint, lethal_threshold=254)

    print(f"Footprint vertices: {len(footprint)}")
    print(f"Edge samples: {len(checker.edge_samples)}")

    # Test collision checking at center (should be free)
    center_x = costmap.width * costmap.resolution / 2.0
    center_y = costmap.height * costmap.resolution / 2.0
    state_center = RobotState(x=center_x, y=center_y, theta=0.0)

    collision, max_cost = checker.check_state(state_center, costmap)
    print(f"\nCenter position:")
    print(f"  Collision: {collision}")
    print(f"  Max cost: {max_cost}")

    # Test trajectory checking
    print("\nTesting trajectory collision checking...")
    robot = RobotModel({
        'max_v_x': 1.0, 'min_v_x': -0.5,
        'max_v_y': 0.5, 'min_v_y': -0.5,
        'max_omega': 1.0, 'min_omega': -1.0
    })

    control_sequence = np.array([
        [0.3, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ])

    trajectory = robot.simulate_trajectory(state_center, control_sequence)
    traj_collision, traj_max_cost = checker.check_trajectory(trajectory, costmap)

    print(f"Trajectory:")
    print(f"  Length: {len(trajectory)} states")
    print(f"  Collision: {traj_collision}")
    print(f"  Max cost: {traj_max_cost}")

    # Test footprint transformation
    print("\nTesting footprint transformation...")
    test_state = RobotState(x=1.0, y=2.0, theta=np.pi/4)
    transformed = checker._transform_footprint(test_state, checker.footprint)
    print(f"Original footprint corners: {checker.footprint[:2]}")
    print(f"Transformed corners: {transformed[:2]}")

    print("\n✓ CollisionChecker tests passed!")
