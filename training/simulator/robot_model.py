"""
Robot dynamics simulation matching C++ trajectory_generator.cpp.

This module provides forward simulation of robot dynamics using Euler integration.
CRITICAL: Implementation must match C++ trajectory_generator.cpp exactly for
consistent behavior between training and deployment.
"""

import numpy as np
from typing import List


class RobotState:
    """
    Complete robot state including position, velocities, and accelerations.

    Attributes:
        x: float, global x position (meters)
        y: float, global y position (meters)
        theta: float, orientation (radians, [-pi, pi])
        v_x: float, linear velocity x in robot frame (m/s)
        v_y: float, linear velocity y in robot frame (m/s)
        omega: float, angular velocity (rad/s)
        a_x: float, linear acceleration x (m/s^2)
        a_y: float, linear acceleration y (m/s^2)
        alpha: float, angular acceleration (rad/s^2)
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0,
                 v_x: float = 0.0, v_y: float = 0.0, omega: float = 0.0):
        """
        Initialize robot state.

        Args:
            x: Global x position (meters)
            y: Global y position (meters)
            theta: Orientation (radians)
            v_x: Linear velocity x in robot frame (m/s)
            v_y: Linear velocity y in robot frame (m/s)
            omega: Angular velocity (rad/s)
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v_x = v_x
        self.v_y = v_y
        self.omega = omega

        # Accelerations (computed during simulation)
        self.a_x = 0.0
        self.a_y = 0.0
        self.alpha = 0.0

    def to_array(self) -> np.ndarray:
        """
        Convert state to numpy array for neural network input.

        Returns:
            state_array: np.ndarray [x, y, theta, v_x, v_y, omega, a_x, a_y, alpha]
        """
        return np.array([
            self.x, self.y, self.theta,
            self.v_x, self.v_y, self.omega,
            self.a_x, self.a_y, self.alpha
        ], dtype=np.float32)

    def copy(self):
        """
        Create deep copy of robot state.

        Returns:
            state_copy: New RobotState instance
        """
        new_state = RobotState(self.x, self.y, self.theta,
                               self.v_x, self.v_y, self.omega)
        new_state.a_x = self.a_x
        new_state.a_y = self.a_y
        new_state.alpha = self.alpha
        return new_state

    def __repr__(self):
        return (f"RobotState(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f}, "
                f"v_x={self.v_x:.3f}, v_y={self.v_y:.3f}, omega={self.omega:.3f})")


class RobotModel:
    """
    Robot forward dynamics simulator using Euler integration.

    CRITICAL: This implementation MUST match trajectory_generator.cpp:50-94
    exactly to ensure training/deployment consistency.
    """

    def __init__(self, velocity_limits: dict):
        """
        Initialize robot model with velocity constraints.

        Args:
            velocity_limits: Dict with keys:
                - max_v_x: Maximum forward velocity (m/s)
                - min_v_x: Minimum forward velocity (m/s, typically negative)
                - max_v_y: Maximum lateral velocity (m/s, 0 for differential drive)
                - min_v_y: Minimum lateral velocity (m/s, 0 for differential drive)
                - max_omega: Maximum angular velocity (rad/s)
                - min_omega: Minimum angular velocity (rad/s, typically negative)
        """
        self.velocity_limits = velocity_limits

    def simulate_step(self, state: RobotState, v_x: float, v_y: float,
                     omega: float, dt: float) -> RobotState:
        """
        Simulate one timestep with Euler integration.

        CRITICAL: Must match C++ trajectory_generator.cpp:50-94 exactly!

        Algorithm (from C++ integrateStep_):
        1. Compute accelerations: a = (v_new - v_old) / dt
        2. Update orientation: next_theta = theta + omega * dt
        3. Normalize theta to [-pi, pi]
        4. Transform velocities to global frame:
           vx_global = v_x * cos(theta) - v_y * sin(theta)
           vy_global = v_x * sin(theta) + v_y * cos(theta)
        5. Update position: x += vx_global * dt, y += vy_global * dt
        6. Clamp velocities to limits

        Args:
            state: Current robot state
            v_x: Commanded linear velocity x (m/s)
            v_y: Commanded linear velocity y (m/s)
            omega: Commanded angular velocity (rad/s)
            dt: Time step (seconds)

        Returns:
            next_state: Robot state after dt seconds
        """
        # Clamp commanded velocities to limits
        v_x = np.clip(v_x, self.velocity_limits['min_v_x'], self.velocity_limits['max_v_x'])
        v_y = np.clip(v_y, self.velocity_limits['min_v_y'], self.velocity_limits['max_v_y'])
        omega = np.clip(omega, self.velocity_limits['min_omega'], self.velocity_limits['max_omega'])

        # Compute accelerations (finite difference)
        a_x = (v_x - state.v_x) / dt
        a_y = (v_y - state.v_y) / dt
        alpha = (omega - state.omega) / dt

        # Update orientation first (C++ line 70-71)
        next_theta = state.theta + omega * dt
        next_theta = self.normalize_angle(next_theta)

        # Transform velocities to global frame (C++ lines 75-79)
        # Robot velocities are in robot frame
        cos_theta = np.cos(state.theta)
        sin_theta = np.sin(state.theta)

        vx_global = v_x * cos_theta - v_y * sin_theta
        vy_global = v_x * sin_theta + v_y * cos_theta

        # Update position (C++ lines 81-82)
        next_x = state.x + vx_global * dt
        next_y = state.y + vy_global * dt

        # Create next state (C++ lines 85-93)
        next_state = RobotState(
            x=next_x,
            y=next_y,
            theta=next_theta,
            v_x=v_x,
            v_y=v_y,
            omega=omega
        )

        # Store accelerations
        next_state.a_x = a_x
        next_state.a_y = a_y
        next_state.alpha = alpha

        return next_state

    def simulate_trajectory(self, initial_state: RobotState,
                          control_sequence: np.ndarray) -> List[RobotState]:
        """
        Simulate complete trajectory from control sequence.

        Args:
            initial_state: Starting robot state
            control_sequence: np.ndarray (num_steps, 3), [[v_x, v_y, omega], ...]

        Returns:
            trajectory: List[RobotState], length = num_steps + 1 (includes initial)
        """
        trajectory = [initial_state.copy()]
        current_state = initial_state.copy()

        # Default dt (typically 0.1s for 10 Hz control)
        dt = 0.1

        for step in range(len(control_sequence)):
            v_x, v_y, omega = control_sequence[step]

            # Simulate one step
            next_state = self.simulate_step(current_state, v_x, v_y, omega, dt)

            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-pi, pi].

        Matches C++ normalizeAngle_() from trajectory_generator.cpp:96-102

        Args:
            angle: Angle in radians

        Returns:
            normalized_angle: Angle in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle


if __name__ == "__main__":
    # Test robot dynamics
    print("Testing RobotModel dynamics...")

    # Create robot model
    velocity_limits = {
        'max_v_x': 1.0,
        'min_v_x': -0.5,
        'max_v_y': 0.5,
        'min_v_y': -0.5,
        'max_omega': 1.0,
        'min_omega': -1.0
    }
    robot = RobotModel(velocity_limits)

    # Test single step
    initial_state = RobotState(x=0.0, y=0.0, theta=0.0, v_x=0.0, v_y=0.0, omega=0.0)
    print(f"Initial state: {initial_state}")

    next_state = robot.simulate_step(initial_state, v_x=0.5, v_y=0.0, omega=0.1, dt=0.1)
    print(f"After 0.1s with v_x=0.5, omega=0.1:")
    print(f"  Next state: {next_state}")
    print(f"  Position: ({next_state.x:.4f}, {next_state.y:.4f})")
    print(f"  Theta: {next_state.theta:.4f} rad ({np.degrees(next_state.theta):.2f}°)")

    # Test trajectory simulation
    print("\nTesting trajectory simulation...")
    control_sequence = np.array([
        [0.5, 0.0, 0.1],  # Forward with slight turn
        [0.5, 0.0, 0.1],
        [0.5, 0.0, 0.0],  # Straight
        [0.5, 0.0, 0.0],
    ])

    trajectory = robot.simulate_trajectory(initial_state, control_sequence)
    print(f"Trajectory length: {len(trajectory)} (expected: {len(control_sequence) + 1})")
    print(f"Final position: ({trajectory[-1].x:.4f}, {trajectory[-1].y:.4f})")
    print(f"Final theta: {trajectory[-1].theta:.4f} rad")

    # Test angle normalization
    print("\nTesting angle normalization...")
    test_angles = [0.0, np.pi, -np.pi, 2*np.pi, -2*np.pi, 3*np.pi, 7.0]
    for angle in test_angles:
        normalized = RobotModel.normalize_angle(angle)
        print(f"  {angle:6.3f} -> {normalized:6.3f} (in range: {-np.pi <= normalized <= np.pi})")

    print("\n✓ RobotModel tests passed!")
