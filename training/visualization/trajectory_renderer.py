"""
Trajectory rendering for Pygame visualization.

Provides trajectory polyline rendering with fitness-based coloring.
"""

import pygame
import numpy as np
from typing import List, Tuple
from .color_utils import fitness_to_color


class TrajectoryRenderer:
    """Render robot trajectories with fitness-based coloring."""

    def __init__(self, costmap_size: int = 800):
        """
        Initialize trajectory renderer.

        Args:
            costmap_size: Costmap render size (for coordinate transform)
        """
        self.costmap_size = costmap_size

    def render_trajectory(self, trajectory: List, fitness: float,
                         fitness_min: float, fitness_max: float,
                         costmap, surface: pygame.Surface,
                         line_width: int = 2):
        """
        Render trajectory as polyline colored by fitness.

        Converts RobotState positions to screen coordinates and draws.

        Args:
            trajectory: List of RobotState objects with .x, .y attributes
            fitness: Fitness value for this trajectory
            fitness_min: Minimum fitness in population
            fitness_max: Maximum fitness in population
            costmap: Costmap object (for coordinate transform)
            surface: Pygame surface to render to
            line_width: Line width in pixels (default: 2)
        """
        if len(trajectory) < 2:
            return  # Need at least 2 points for a line

        # Skip invalid fitness values
        if not np.isfinite(fitness):
            return

        # Get color based on fitness
        color = fitness_to_color(fitness, fitness_min, fitness_max)

        # Convert trajectory to screen coordinates
        points = []
        scale_factor = self.costmap_size // costmap.width

        for state in trajectory:
            # World → grid
            grid_x = int(state.x / costmap.resolution)
            grid_y = int(state.y / costmap.resolution)

            # Grid → screen
            screen_x = grid_x * scale_factor
            screen_y = grid_y * scale_factor

            # Clamp to screen bounds
            screen_x = np.clip(screen_x, 0, self.costmap_size - 1)
            screen_y = np.clip(screen_y, 0, self.costmap_size - 1)

            points.append((screen_x, screen_y))

        # Draw polyline
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, line_width)

    def render_robot_footprint(self, state, costmap, surface: pygame.Surface,
                              footprint_vertices: List[Tuple[float, float]],
                              color: Tuple[int, int, int] = (0, 255, 255)):
        """
        Render robot footprint at given state.

        Args:
            state: RobotState with .x, .y, .theta
            costmap: Costmap object
            surface: Pygame surface
            footprint_vertices: List of (x, y) vertices in robot frame
            color: RGB color tuple (default: cyan)
        """
        # Transform footprint vertices to world frame
        cos_theta = np.cos(state.theta)
        sin_theta = np.sin(state.theta)

        world_vertices = []
        for vx, vy in footprint_vertices:
            # Rotate and translate
            wx = state.x + vx * cos_theta - vy * sin_theta
            wy = state.y + vx * sin_theta + vy * cos_theta
            world_vertices.append((wx, wy))

        # Convert to screen coordinates
        scale_factor = self.costmap_size // costmap.width
        screen_vertices = []

        for wx, wy in world_vertices:
            grid_x = int(wx / costmap.resolution)
            grid_y = int(wy / costmap.resolution)
            screen_x = grid_x * scale_factor
            screen_y = grid_y * scale_factor
            screen_x = np.clip(screen_x, 0, self.costmap_size - 1)
            screen_y = np.clip(screen_y, 0, self.costmap_size - 1)
            screen_vertices.append((screen_x, screen_y))

        # Draw polygon
        if len(screen_vertices) >= 3:
            pygame.draw.polygon(surface, color, screen_vertices, 2)
