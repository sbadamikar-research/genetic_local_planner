"""
High-level navigation simulation wrapper for GA training.

This module provides a unified interface for running navigation scenarios,
combining costmap, robot dynamics, and collision checking.
"""

import numpy as np
from typing import Dict, List, Tuple
from .robot_model import RobotState, RobotModel
from .collision_checker import CollisionChecker
from .costmap import Costmap


class NavigationEnvironment:
    """
    Complete navigation simulation environment.

    Integrates costmap, robot dynamics, and collision checking for
    evaluating navigation trajectories during GA training.
    """

    def __init__(self, config: dict):
        """
        Initialize environment with configuration.

        Args:
            config: Configuration dict with 'robot' section containing:
                - footprint: List of (x,y) vertices
                - max_v_x, min_v_x, max_v_y, min_v_y, max_omega, min_omega
        """
        # Extract robot configuration
        robot_config = config['robot']

        # Create robot model
        velocity_limits = {
            'max_v_x': robot_config['max_v_x'],
            'min_v_x': robot_config['min_v_x'],
            'max_v_y': robot_config['max_v_y'],
            'min_v_y': robot_config['min_v_y'],
            'max_omega': robot_config['max_omega'],
            'min_omega': robot_config['min_omega']
        }
        self.robot_model = RobotModel(velocity_limits)

        # Create collision checker
        footprint = robot_config['footprint']
        self.collision_checker = CollisionChecker(footprint, lethal_threshold=254)

        # Scenario state (set by reset)
        self.costmap = None
        self.start_state = None
        self.goal_x = None
        self.goal_y = None
        self.goal_theta = None  # Optional goal orientation

    def reset(self, costmap: Costmap, start_state: RobotState,
              goal_x: float, goal_y: float, goal_theta: float = 0.0):
        """
        Reset environment with new scenario.

        Args:
            costmap: Costmap instance for this scenario
            start_state: Initial robot state
            goal_x: Goal x position in world frame (meters)
            goal_y: Goal y position in world frame (meters)
            goal_theta: Goal orientation (radians, default 0.0)
        """
        self.costmap = costmap
        self.start_state = start_state.copy()
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_theta = goal_theta

    def simulate_control_sequence(self, control_sequence: np.ndarray) -> Dict:
        """
        Simulate control sequence and return trajectory metrics.

        Args:
            control_sequence: np.ndarray (num_steps, 3), [[v_x, v_y, omega], ...]

        Returns:
            metrics: Dict with:
                - trajectory: List[RobotState], simulated trajectory
                - collision: bool, True if any collision occurred
                - max_cost: int, maximum costmap value encountered
                - goal_distance: float, final distance to goal (meters)
                - goal_distance_reduction: float, start_dist - end_dist (positive is improvement)
                - smoothness: float, sum of squared control changes
                - path_length: float, total distance traveled (meters)
                - goal_reached: bool, True if within tolerance
        """
        # Simulate trajectory
        trajectory = self.robot_model.simulate_trajectory(
            self.start_state, control_sequence
        )

        # Check collisions
        collision, max_cost = self.collision_checker.check_trajectory(
            trajectory, self.costmap
        )

        # Compute goal distance (initial and final)
        initial_distance = self._compute_distance_to_goal(self.start_state)
        final_state = trajectory[-1]
        final_distance = self._compute_distance_to_goal(final_state)
        goal_distance_reduction = initial_distance - final_distance

        # Check if goal reached (within 0.1m and 0.2 rad)
        goal_reached = self._is_goal_reached(final_state, tolerance_pos=0.1, tolerance_theta=0.2)

        # Compute smoothness (sum of squared control changes)
        smoothness = self._compute_smoothness(control_sequence)

        # Compute path length
        path_length = self._compute_path_length(trajectory)

        return {
            'trajectory': trajectory,
            'collision': collision,
            'max_cost': max_cost,
            'goal_distance': final_distance,
            'goal_distance_reduction': goal_distance_reduction,
            'smoothness': smoothness,
            'path_length': path_length,
            'goal_reached': goal_reached
        }

    def get_relative_goal(self, state: RobotState) -> np.ndarray:
        """
        Compute goal position and orientation in robot frame.

        Args:
            state: Current robot state

        Returns:
            goal_relative: np.ndarray [dx, dy, dtheta], goal in robot frame
        """
        # Goal in world frame
        dx_world = self.goal_x - state.x
        dy_world = self.goal_y - state.y

        # Transform to robot frame
        cos_theta = np.cos(state.theta)
        sin_theta = np.sin(state.theta)

        dx_robot = cos_theta * dx_world + sin_theta * dy_world
        dy_robot = -sin_theta * dx_world + cos_theta * dy_world

        # Goal orientation difference
        dtheta = self.goal_theta - state.theta
        dtheta = RobotModel.normalize_angle(dtheta)

        return np.array([dx_robot, dy_robot, dtheta], dtype=np.float32)

    def _compute_distance_to_goal(self, state: RobotState) -> float:
        """
        Compute Euclidean distance from state to goal.

        Args:
            state: Robot state

        Returns:
            distance: Distance to goal (meters)
        """
        dx = self.goal_x - state.x
        dy = self.goal_y - state.y
        return np.sqrt(dx**2 + dy**2)

    def _is_goal_reached(self, state: RobotState, tolerance_pos: float = 0.1,
                        tolerance_theta: float = 0.2) -> bool:
        """
        Check if robot has reached goal.

        Args:
            state: Robot state
            tolerance_pos: Position tolerance (meters)
            tolerance_theta: Orientation tolerance (radians)

        Returns:
            reached: True if within tolerance
        """
        # Position tolerance
        distance = self._compute_distance_to_goal(state)
        if distance > tolerance_pos:
            return False

        # Orientation tolerance
        dtheta = self.goal_theta - state.theta
        dtheta = RobotModel.normalize_angle(dtheta)
        if abs(dtheta) > tolerance_theta:
            return False

        return True

    def _compute_smoothness(self, control_sequence: np.ndarray) -> float:
        """
        Compute smoothness penalty (sum of squared control changes).

        Measures how much controls vary between timesteps.
        Lower is smoother.

        Args:
            control_sequence: np.ndarray (num_steps, 3)

        Returns:
            smoothness: Sum of squared control changes
        """
        if len(control_sequence) <= 1:
            return 0.0

        # Compute differences between consecutive controls
        control_diffs = np.diff(control_sequence, axis=0)

        # Sum of squared differences
        smoothness = np.sum(control_diffs ** 2)

        return float(smoothness)

    def _compute_path_length(self, trajectory: List[RobotState]) -> float:
        """
        Compute total path length (sum of distances between consecutive states).

        Args:
            trajectory: List of RobotState

        Returns:
            path_length: Total distance traveled (meters)
        """
        if len(trajectory) <= 1:
            return 0.0

        path_length = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i].x - trajectory[i-1].x
            dy = trajectory[i].y - trajectory[i-1].y
            path_length += np.sqrt(dx**2 + dy**2)

        return path_length

    def get_costmap_window(self, state: RobotState, window_size: int = 50) -> np.ndarray:
        """
        Extract normalized costmap window around robot.

        Useful for NN training data preparation.

        Args:
            state: Robot state (center of window)
            window_size: Window size in pixels (default 50x50)

        Returns:
            window: np.ndarray (window_size, window_size), float32, [0,1]
        """
        # Get robot position in grid coordinates
        mx, my = self.costmap.world_to_grid(state.x, state.y)

        # Compute window bounds
        half_size = window_size // 2
        x_min = mx - half_size
        x_max = mx + half_size
        y_min = my - half_size
        y_max = my + half_size

        # Extract window (pad with zeros if out of bounds)
        window = np.zeros((window_size, window_size), dtype=np.uint8)

        # Compute valid region
        src_x_min = max(0, x_min)
        src_x_max = min(self.costmap.width, x_max)
        src_y_min = max(0, y_min)
        src_y_max = min(self.costmap.height, y_max)

        dst_x_min = src_x_min - x_min
        dst_x_max = dst_x_min + (src_x_max - src_x_min)
        dst_y_min = src_y_min - y_min
        dst_y_max = dst_y_min + (src_y_max - src_y_min)

        # Copy valid region
        if src_x_max > src_x_min and src_y_max > src_y_min:
            window[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = \
                self.costmap.data[src_y_min:src_y_max, src_x_min:src_x_max]

        # Normalize
        return window.astype(np.float32) / 254.0


if __name__ == "__main__":
    # Test navigation environment
    print("Testing NavigationEnvironment...")

    from .costmap import generate_random_costmap

    # Create test configuration
    config = {
        'robot': {
            'footprint': [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)],
            'max_v_x': 1.0,
            'min_v_x': -0.5,
            'max_v_y': 0.5,
            'min_v_y': -0.5,
            'max_omega': 1.0,
            'min_omega': -1.0
        }
    }

    # Create environment
    env = NavigationEnvironment(config)

    # Generate test costmap
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

    # Create start state (center of map)
    center_x = costmap.width * costmap.resolution / 2.0
    center_y = costmap.height * costmap.resolution / 2.0
    start_state = RobotState(x=center_x, y=center_y, theta=0.0)

    # Create goal (1.5m ahead)
    goal_x = center_x + 1.5
    goal_y = center_y
    goal_theta = 0.0

    # Reset environment
    env.reset(costmap, start_state, goal_x, goal_y, goal_theta)

    print(f"Start: ({start_state.x:.2f}, {start_state.y:.2f}, {start_state.theta:.2f})")
    print(f"Goal: ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})")
    print(f"Initial distance to goal: {env._compute_distance_to_goal(start_state):.2f}m")

    # Test relative goal computation
    goal_relative = env.get_relative_goal(start_state)
    print(f"\nGoal in robot frame: ({goal_relative[0]:.2f}, {goal_relative[1]:.2f}, {goal_relative[2]:.2f})")

    # Test control sequence (straight ahead)
    control_sequence = np.array([
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])

    print(f"\nSimulating control sequence ({len(control_sequence)} steps)...")
    metrics = env.simulate_control_sequence(control_sequence)

    print(f"\nSimulation Results:")
    print(f"  Trajectory length: {len(metrics['trajectory'])} states")
    print(f"  Collision: {metrics['collision']}")
    print(f"  Max cost: {metrics['max_cost']}")
    print(f"  Final goal distance: {metrics['goal_distance']:.3f}m")
    print(f"  Distance reduction: {metrics['goal_distance_reduction']:.3f}m")
    print(f"  Goal reached: {metrics['goal_reached']}")
    print(f"  Smoothness: {metrics['smoothness']:.3f}")
    print(f"  Path length: {metrics['path_length']:.3f}m")

    # Test costmap window extraction
    print(f"\nTesting costmap window extraction...")
    window = env.get_costmap_window(start_state, window_size=50)
    print(f"Window shape: {window.shape}")
    print(f"Window range: [{window.min():.3f}, {window.max():.3f}]")

    # Test turning control
    print(f"\nTesting turning trajectory...")
    turn_sequence = np.array([
        [0.3, 0.0, 0.5],
        [0.3, 0.0, 0.5],
        [0.3, 0.0, 0.5],
        [0.3, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ])

    turn_metrics = env.simulate_control_sequence(turn_sequence)
    final_state = turn_metrics['trajectory'][-1]
    print(f"Final position: ({final_state.x:.2f}, {final_state.y:.2f}, {final_state.theta:.2f})")
    print(f"Final goal distance: {turn_metrics['goal_distance']:.3f}m")
    print(f"Path length: {turn_metrics['path_length']:.3f}m")

    print("\n✓ NavigationEnvironment tests passed!")
